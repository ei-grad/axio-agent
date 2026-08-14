"""Serialized primary-buffer output for the interactive REPL."""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from collections import deque
from dataclasses import dataclass
from io import TextIOBase
from typing import Any, Literal, TextIO, cast

from prompt_toolkit.application import run_in_terminal

RESET = "\033[0m"
MAX_PENDING_CHARS = 256 * 1024
MAX_BATCH_CHARS = 32 * 1024

type OutputStream = Literal["stdout", "stderr"]


@dataclass(frozen=True, slots=True)
class OutputFrame:
    """One atomic write accepted by the terminal owner."""

    content: str
    stream: OutputStream = "stdout"


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

    Application output is written through ``run_in_terminal`` so prompt_toolkit
    removes and redraws its inline layout on the primary screen. Producers see
    ordinary TextIO objects, but completed lines are serialized by one asyncio
    consumer instead of being written from background tasks or logging threads.
    """

    def __init__(self, prompt_session: Any) -> None:
        self._session = prompt_session
        self._output = prompt_session.app.output
        self._loop: asyncio.AbstractEventLoop | None = None
        self._consumer: asyncio.Task[None] | None = None
        self._consumer_error: BaseException | None = None
        self._state_lock = threading.RLock()
        self._pending: deque[OutputFrame] = deque()
        self._pending_chars = 0
        self._dropped_frames = 0
        self._dropped_chars = 0
        self._wake: asyncio.Event | None = None
        self._drained: asyncio.Event | None = None
        self._failed: asyncio.Event | None = None
        self._wake_scheduled = False
        self._writing = False
        self._closing = False
        self._active = False
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None
        self.stdout: _TerminalStream | None = None
        self.stderr: _TerminalStream | None = None

    async def start(self) -> None:
        if self._active:
            raise RuntimeError("terminal UI is already active")
        self._loop = asyncio.get_running_loop()
        self._original_stdout = cast(TextIO, sys.stdout)
        self._original_stderr = cast(TextIO, sys.stderr)
        self.stdout = _TerminalStream(self, "stdout", self._original_stdout)
        self.stderr = _TerminalStream(self, "stderr", self._original_stderr)
        self._wake = asyncio.Event()
        self._drained = asyncio.Event()
        self._drained.set()
        self._failed = asyncio.Event()
        self._active = True
        self._consumer = asyncio.create_task(self._consume(), name="axio-repl-terminal-output")
        self._rebind_logging(self._original_stdout, cast(TextIO, self.stdout))
        self._rebind_logging(self._original_stderr, cast(TextIO, self.stderr))
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def submit(self, frame: OutputFrame) -> None:
        if not frame.content:
            return
        loop = self._loop
        fallback: TextIO | None = None
        consumer_error: BaseException | None = None
        schedule_wake = False
        with self._state_lock:
            if not self._active or loop is None:
                fallback = self._original_stderr if frame.stream == "stderr" else self._original_stdout
            elif self._consumer_error is not None:
                consumer_error = self._consumer_error
            else:
                self._enqueue_locked(frame)
                if not self._wake_scheduled:
                    self._wake_scheduled = True
                    schedule_wake = True
        if fallback is not None:
            fallback.write(frame.content)
            fallback.flush()
            return
        if consumer_error is not None:
            raise RuntimeError("terminal output consumer failed") from consumer_error
        if not schedule_wake:
            return
        assert loop is not None
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
        assert self._drained is not None
        while True:
            with self._state_lock:
                consumer_error = self._consumer_error
                idle = (
                    not self._pending and not self._dropped_frames and not self._writing and not self._wake_scheduled
                )
            if consumer_error is not None:
                raise RuntimeError("terminal output consumer failed") from consumer_error
            if idle:
                return
            self._drained.clear()
            with self._state_lock:
                idle = (
                    not self._pending and not self._dropped_frames and not self._writing and not self._wake_scheduled
                )
            if idle:
                self._drained.set()
                continue
            await self._drained.wait()

    async def wait_failed(self) -> None:
        assert self._failed is not None
        await self._failed.wait()
        assert self._consumer_error is not None
        raise RuntimeError("terminal output consumer failed") from self._consumer_error

    @property
    def pending_char_count(self) -> int:
        with self._state_lock:
            return self._pending_chars

    async def close(self) -> None:
        if not self._active:
            return
        assert self.stdout is not None
        assert self.stderr is not None
        assert self._original_stdout is not None
        assert self._original_stderr is not None

        error: BaseException | None = None
        try:
            if sys.stdout is self.stdout:
                sys.stdout = self._original_stdout
            if sys.stderr is self.stderr:
                sys.stderr = self._original_stderr
            self._rebind_logging(self.stdout, self._original_stdout)
            self._rebind_logging(self.stderr, self._original_stderr)
            self.stdout.flush_all()
            self.stderr.flush_all()
            await self.drain()
        except BaseException as exc:
            error = exc
        finally:
            self._request_close()
            consumer = self._consumer
            if consumer is not None:
                try:
                    await consumer
                except BaseException as exc:
                    if error is None:
                        error = exc
            try:
                self._restore_terminal()
            except BaseException as exc:
                if error is None:
                    error = exc
            self._active = False
        if error is not None:
            raise error

    async def _consume(self) -> None:
        assert self._wake is not None
        assert self._drained is not None
        while True:
            await self._wake.wait()
            self._wake.clear()
            while True:
                content = self._next_batch()
                if content is None:
                    self._drained.set()
                    with self._state_lock:
                        closing = self._closing
                    if closing:
                        return
                    break
                try:
                    await self._write(content)
                except BaseException as exc:
                    self._record_failure(exc)
                    return

    def _enqueue_locked(self, frame: OutputFrame) -> None:
        size = len(frame.content)
        if self._dropped_frames or self._pending_chars + size > MAX_PENDING_CHARS:
            self._dropped_frames += 1
            self._dropped_chars += size
            return
        if (
            self._pending
            and self._pending[-1].stream == frame.stream
            and len(self._pending[-1].content) + size <= MAX_BATCH_CHARS
        ):
            previous = self._pending[-1]
            self._pending[-1] = OutputFrame(previous.content + frame.content, frame.stream)
        else:
            self._pending.append(frame)
        self._pending_chars += size

    def _notify_consumer(self) -> None:
        assert self._wake is not None
        assert self._drained is not None
        with self._state_lock:
            self._wake_scheduled = False
        self._drained.clear()
        self._wake.set()

    def _next_batch(self) -> str | None:
        with self._state_lock:
            if self._pending:
                batch = [self._pending.popleft()]
                size = len(batch[0].content)
                while self._pending and size + len(self._pending[0].content) <= MAX_BATCH_CHARS:
                    frame = self._pending.popleft()
                    batch.append(frame)
                    size += len(frame.content)
                self._pending_chars -= size
                self._writing = True
                return "".join(frame.content for frame in batch)
            if self._dropped_frames:
                dropped_frames = self._dropped_frames
                chars = self._dropped_chars
                self._dropped_frames = 0
                self._dropped_chars = 0
                self._writing = True
                return f"{RESET}\n[terminal output skipped: {dropped_frames} frame(s), {chars} character(s)]\n"
            self._writing = False
            return None

    def _record_failure(self, exc: BaseException) -> None:
        assert self._failed is not None
        assert self._drained is not None
        with self._state_lock:
            self._consumer_error = exc
            self._pending.clear()
            self._pending_chars = 0
            self._dropped_frames = 0
            self._dropped_chars = 0
            self._writing = False
        self._failed.set()
        self._drained.set()

    def _request_close(self) -> None:
        loop = self._loop
        if loop is None:
            return
        schedule_wake = False
        with self._state_lock:
            self._closing = True
            if not self._wake_scheduled:
                self._wake_scheduled = True
                schedule_wake = True
        if schedule_wake:
            self._notify_consumer()

    async def _write(self, content: str) -> None:
        def write_and_flush() -> None:
            self._output.enable_autowrap()
            self._output.write_raw(content)
            self._output.flush()

        app = self._session.app
        if app.is_running:
            context = app.context
            if context is None:
                await run_in_terminal(write_and_flush, in_executor=False)
            else:
                operation = context.copy().run(lambda: run_in_terminal(write_and_flush, in_executor=False))
                await operation
        else:
            write_and_flush()

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
