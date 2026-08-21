"""Serialized primary-buffer output for the interactive REPL."""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from enum import StrEnum
from io import TextIOBase
from typing import Any, TextIO, cast

from axio_repl import _replay
from axio_repl._prompt_terminal import PromptToolkitInlineOutput
from axio_repl._terminal_ingress import (
    MAX_BATCH_CHARS,
    MAX_PENDING_CHARS,
    RESET,
    IngressDestination,
    OutputFrame,
    OutputStream,
    TerminalIngress,
)

__all__ = ["MAX_BATCH_CHARS", "MAX_PENDING_CHARS", "RESET", "OutputFrame", "TerminalPhase", "TerminalUI"]


class TerminalPhase(StrEnum):
    NEW = "new"
    RUNNING = "running"
    FAILED = "failed"
    DRAINING = "draining"
    CLOSED = "closed"


class _TerminalStream(TextIOBase):
    """Line-buffered TextIO facade whose writes become terminal frames."""

    def __init__(self, owner: TerminalUI, stream: OutputStream, fallback: TextIO) -> None:
        self._owner = owner
        self._stream = stream
        self._fallback = fallback
        self._lock = threading.RLock()
        self._buffers: dict[object, list[str]] = {}

    @property
    def encoding(self) -> str:
        return self._fallback.encoding or "utf-8"

    @encoding.setter
    def encoding(self, value: str) -> None:
        raise AttributeError("encoding is read-only")

    @property
    def errors(self) -> str | None:
        return self._fallback.errors

    @errors.setter
    def errors(self, value: str | None) -> None:
        raise AttributeError("errors is read-only")

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return self._fallback.isatty()

    def fileno(self) -> int:
        return self._fallback.fileno()

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            raise TypeError(f"write() argument must be str, not {type(data).__name__}")
        if not data:
            return 0
        writer = self._writer_key()
        with self._lock:
            buffered = self._buffers.setdefault(writer, [])
            if "\n" in data:
                before, after = data.rsplit("\n", 1)
                content = "".join([*buffered, before, "\n"])
                if after:
                    self._buffers[writer] = [after]
                else:
                    self._buffers.pop(writer, None)
                self._owner.submit(OutputFrame(content, self._stream))
            else:
                buffered.append(data)
        return len(data)

    def flush(self) -> None:
        self._flush_writers((self._writer_key(),))

    def flush_all(self) -> None:
        with self._lock:
            self._flush_writers(tuple(self._buffers))

    @staticmethod
    def _writer_key() -> object:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            return loop
        return ("thread", threading.get_ident())

    def _flush_writers(self, writers: tuple[object, ...]) -> None:
        with self._lock:
            for writer in writers:
                content = "".join(self._buffers.pop(writer, ()))
                if content:
                    self._owner.submit(OutputFrame(content, self._stream))


class TerminalUI:
    """The only interactive writer allowed to touch the prompt terminal.

    prompt_toolkit's inline layout is removed and redrawn around application
    output without detaching input or leaving raw mode. Producers see ordinary
    TextIO objects, but completed lines are serialized by one asyncio consumer
    instead of being written from background tasks or logging threads.
    """

    def __init__(self, prompt_session: Any) -> None:
        self._session = prompt_session
        self._output = prompt_session.app.output
        self._inline_output = PromptToolkitInlineOutput(prompt_session)
        self._reset = cast(str, getattr(prompt_session, "_axio_terminal_reset", RESET))
        self._loop: asyncio.AbstractEventLoop | None = None
        self._consumer: asyncio.Task[None] | None = None
        self._consumer_error: BaseException | None = None
        self._state_lock = threading.RLock()
        self._fallback_lock = threading.RLock()
        self._lifecycle_lock = asyncio.Lock()
        self._phase = TerminalPhase.NEW
        self._ingress: TerminalIngress | None = None
        self._wake: asyncio.Event | None = None
        self._drained: asyncio.Event | None = None
        self._failed: asyncio.Event | None = None
        self._initial_stdout = cast(TextIO, sys.stdout)
        self._initial_stderr = cast(TextIO, sys.stderr)
        self._original_stdout: TextIO | None = self._initial_stdout
        self._original_stderr: TextIO | None = self._initial_stderr
        self.stdout: _TerminalStream | None = None
        self.stderr: _TerminalStream | None = None

    @property
    def phase(self) -> TerminalPhase:
        with self._state_lock:
            return self._phase

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._phase is not TerminalPhase.NEW:
                raise RuntimeError(f"terminal UI cannot start from {self._phase.value} state")
            if sys.platform == "win32":
                raise RuntimeError("interactive axio-repl terminal UI requires POSIX")
            self._loop = asyncio.get_running_loop()
            self._original_stdout = cast(TextIO, sys.stdout)
            self._original_stderr = cast(TextIO, sys.stderr)
            self.stdout = _TerminalStream(self, "stdout", self._original_stdout)
            self.stderr = _TerminalStream(self, "stderr", self._original_stderr)
            ingress = TerminalIngress(reset=self._reset)
            self._ingress = ingress
            self._wake = asyncio.Event()
            self._drained = asyncio.Event()
            self._drained.set()
            self._failed = asyncio.Event()
            self._consumer_error = None
            rebound_handlers: list[tuple[Any, TextIO]] = []
            try:
                self._rebind_logging_recorded(
                    self._original_stdout,
                    cast(TextIO, self.stdout),
                    rebound_handlers,
                )
                self._rebind_logging_recorded(
                    self._original_stderr,
                    cast(TextIO, self.stderr),
                    rebound_handlers,
                )
                sys.stdout = self.stdout
                sys.stderr = self.stderr
                self._consumer = asyncio.create_task(self._consume(), name="axio-repl-terminal-output")
                with self._state_lock:
                    self._phase = TerminalPhase.RUNNING
            except BaseException:
                if sys.stdout is self.stdout:
                    sys.stdout = self._original_stdout
                if sys.stderr is self.stderr:
                    sys.stderr = self._original_stderr
                self._restore_recorded_logging(rebound_handlers)
                consumer = self._consumer
                if consumer is not None and not consumer.done():
                    consumer.cancel()
                    await asyncio.gather(consumer, return_exceptions=True)
                if ingress.phase.value == "open":
                    ingress.seal()
                ingress.close_late()
                with self._state_lock:
                    self._phase = TerminalPhase.CLOSED
                raise

    def submit(self, frame: OutputFrame) -> None:
        if not frame.content:
            return
        loop: asyncio.AbstractEventLoop | None
        ingress: TerminalIngress | None
        fallback: TextIO
        with self._state_lock:
            loop = self._loop
            ingress = self._ingress
            fallback = self._fallback_stream_locked(frame.stream)
            active = self._phase in {TerminalPhase.RUNNING, TerminalPhase.FAILED, TerminalPhase.DRAINING}
        if not active or ingress is None or loop is None:
            self._write_fallback(fallback, frame.content, frame.stream)
            return
        admission = ingress.submit(frame)
        if admission.destination is IngressDestination.FALLBACK:
            self._write_fallback(fallback, frame.content, frame.stream)
            return
        if not admission.wake_consumer:
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            self._notify_consumer()
        else:
            loop.call_soon_threadsafe(self._notify_consumer)

    async def drain(self) -> None:
        await asyncio.sleep(0)
        ingress = self._ingress
        drained = self._drained
        if ingress is None or drained is None:
            return
        while True:
            with self._state_lock:
                consumer_error = self._consumer_error
            if consumer_error is not None:
                raise RuntimeError("terminal output consumer failed") from consumer_error
            if ingress.idle:
                return
            drained.clear()
            if ingress.idle:
                drained.set()
                continue
            await drained.wait()

    async def wait_failed(self) -> None:
        failed = self._failed
        if failed is None:
            raise RuntimeError("terminal UI has not started")
        await failed.wait()
        with self._state_lock:
            consumer_error = self._consumer_error
        assert consumer_error is not None
        raise RuntimeError("terminal output consumer failed") from consumer_error

    @property
    def pending_char_count(self) -> int:
        ingress = self._ingress
        return 0 if ingress is None else ingress.pending_char_count

    async def close(self) -> None:
        async with self._lifecycle_lock:
            with self._state_lock:
                if self._phase is TerminalPhase.CLOSED:
                    return
                if self._phase is TerminalPhase.NEW:
                    self._phase = TerminalPhase.CLOSED
                    return
                self._phase = TerminalPhase.DRAINING
                consumer_error = self._consumer_error
                error: BaseException | None = None
                if consumer_error is not None:
                    error = RuntimeError("terminal output consumer failed")
                    error.__cause__ = consumer_error
            assert self.stdout is not None
            assert self.stderr is not None
            assert self._original_stdout is not None
            assert self._original_stderr is not None
            assert self._ingress is not None

            try:
                if sys.stdout is self.stdout:
                    sys.stdout = self._original_stdout
                if sys.stderr is self.stderr:
                    sys.stderr = self._original_stderr
                self._rebind_logging(self.stdout, self._original_stdout)
                self._rebind_logging(self.stderr, self._original_stderr)
                self.stdout.flush_all()
                self.stderr.flush_all()
            except BaseException as exc:
                if error is None:
                    error = exc

            if self._ingress.seal():
                self._notify_consumer()
            consumer = self._consumer
            if consumer is not None:
                try:
                    await consumer
                except BaseException as exc:
                    if error is None:
                        error = exc
            with self._state_lock:
                consumer_error = self._consumer_error
            if error is None and consumer_error is not None:
                error = RuntimeError("terminal output consumer failed")
                error.__cause__ = consumer_error
            try:
                self._restore_terminal()
            except BaseException as exc:
                if error is None:
                    error = exc
            try:
                self._flush_late_output(self._ingress)
            except BaseException as exc:
                if error is None:
                    error = exc
            finally:
                with self._state_lock:
                    self._phase = TerminalPhase.CLOSED
            if error is not None:
                raise error

    async def _consume(self) -> None:
        assert self._wake is not None
        assert self._drained is not None
        assert self._ingress is not None
        while True:
            await self._wake.wait()
            self._wake.clear()
            self._ingress.wake_delivered()
            while True:
                content = self._ingress.next_batch()
                if content is None:
                    self._drained.set()
                    if self._ingress.consumer_should_stop:
                        return
                    break
                try:
                    await self._write(content)
                except BaseException as exc:
                    self._record_failure(exc)
                    return
                finally:
                    self._ingress.finish_batch()

    def _notify_consumer(self) -> None:
        assert self._wake is not None
        assert self._drained is not None
        assert self._ingress is not None
        self._ingress.wake_delivered()
        self._drained.clear()
        self._wake.set()

    def _record_failure(self, exc: BaseException) -> None:
        assert self._failed is not None
        assert self._drained is not None
        with self._state_lock:
            if self._consumer_error is None:
                self._consumer_error = exc
            if self._phase is TerminalPhase.RUNNING:
                self._phase = TerminalPhase.FAILED
        assert self._ingress is not None
        self._ingress.fail()
        self._failed.set()
        self._drained.set()

    async def _write(self, content: str) -> None:
        def write_and_flush() -> None:
            self._output.enable_autowrap()
            self._output.write_raw(content)
            self._output.flush()

        await self._inline_output.write(write_and_flush)

    def _flush_late_output(self, ingress: TerminalIngress) -> None:
        assert self._original_stdout is not None
        assert self._original_stderr is not None
        with self._fallback_lock:
            late = ingress.close_late()
            for frame in late.frames:
                fallback = self._original_stderr if frame.stream == "stderr" else self._original_stdout
                fallback.write(frame.content)
                fallback.flush()
                _replay.record_terminal_fallback(self._output, frame.content, frame.stream, "late")
            if late.dropped_frames:
                marker = (
                    f"{self._reset}\n[late terminal output skipped: {late.dropped_frames} frame(s), "
                    f"{late.dropped_chars} character(s)]\n"
                )
                self._original_stderr.write(marker)
                self._original_stderr.flush()
                _replay.record_terminal_fallback(self._output, marker, "stderr", "late")

    def _fallback_stream_locked(self, stream: OutputStream) -> TextIO:
        if stream == "stderr":
            return self._original_stderr or self._initial_stderr
        return self._original_stdout or self._initial_stdout

    def _write_fallback(self, fallback: TextIO, content: str, stream: OutputStream) -> None:
        with self._fallback_lock:
            fallback.write(content)
            fallback.flush()
            _replay.record_terminal_fallback(self._output, content, stream, "fallback")

    def _restore_terminal(self) -> None:
        self._output.reset_attributes()
        self._output.enable_autowrap()
        self._output.show_cursor()
        self._output.flush()

    @staticmethod
    def _rebind_logging(previous: object, current: TextIO) -> None:
        root = logging.getLogger()
        loggers = [root]
        loggers.extend(
            logger for logger in logging.Logger.manager.loggerDict.values() if isinstance(logger, logging.Logger)
        )
        for logger in loggers:
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream is previous:
                    handler.setStream(current)

    @staticmethod
    def _rebind_logging_recorded(
        previous: object,
        current: TextIO,
        rebound_handlers: list[tuple[Any, TextIO]],
    ) -> None:
        root = logging.getLogger()
        loggers = [root]
        loggers.extend(
            logger for logger in logging.Logger.manager.loggerDict.values() if isinstance(logger, logging.Logger)
        )
        for logger in loggers:
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream is previous:
                    rebound_handlers.append((handler, cast(TextIO, previous)))
                    handler.setStream(current)

    @staticmethod
    def _restore_recorded_logging(rebound_handlers: list[tuple[Any, TextIO]]) -> None:
        for handler, previous in reversed(rebound_handlers):
            handler.acquire()
            try:
                handler.stream = previous
            finally:
                handler.release()
