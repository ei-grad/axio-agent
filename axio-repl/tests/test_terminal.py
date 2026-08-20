from __future__ import annotations

import asyncio
import logging
import os
import sys
import termios
import threading
import time
from datetime import datetime
from io import TextIOWrapper
from typing import Any, cast

import pytest
from axio.events import SessionEndEvent, TextDelta
from axio.types import StopReason, Usage
from axio_tools_agents.runtime import ExecutionMode, TurnStarted
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output

from axio_repl import ReplRenderer, _panel
from axio_repl._terminal import MAX_PENDING_CHARS, RESET, OutputFrame, TerminalPhase, TerminalUI
from axio_repl._theme import DEFAULT_THEME, MONOCHROME_THEME, TerminalTheme


class _RecordingOutput(DummyOutput):
    def __init__(self) -> None:
        self.raw: list[str] = []

    def write_raw(self, data: str) -> None:
        self.raw.append(data)

    def reset_attributes(self) -> None:
        self.raw.append(RESET)

    def enable_autowrap(self) -> None:
        self.raw.append("\x1b[?7h")

    def show_cursor(self) -> None:
        self.raw.append("\x1b[?25h")


class _Session:
    def __init__(self, output: _RecordingOutput, *, reset: str = RESET) -> None:
        self.app = type("App", (), {"output": output, "is_running": False})()
        self._axio_terminal_reset = reset


class _FailingOutput(_RecordingOutput):
    def write_raw(self, data: str) -> None:
        raise OSError("terminal disconnected")


async def test_terminal_ui_serializes_prints_and_logging_through_one_sink() -> None:
    output = _RecordingOutput()
    terminal = TerminalUI(_Session(output))
    logger = logging.getLogger("axio-repl-terminal-test")
    previous_handlers = list(logger.handlers)
    previous_level = logger.level
    previous_propagate = logger.propagate
    handler = logging.StreamHandler(sys.stderr)
    logger.handlers[:] = [handler]
    logger.setLevel(logging.WARNING)
    logger.propagate = False

    await terminal.start()
    try:
        print("stdout line")
        logger.warning("logged line")
        await terminal.drain()
    finally:
        await terminal.close()
        logger.handlers[:] = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    rendered = "".join(output.raw)
    assert "stdout line\n" in rendered
    assert "logged line\n" in rendered
    assert sys.stdout is not terminal.stdout
    assert sys.stderr is not terminal.stderr


async def test_terminal_ui_keeps_concurrent_lines_atomic() -> None:
    output = _RecordingOutput()
    terminal = TerminalUI(_Session(output))

    await terminal.start()
    try:
        await asyncio.gather(
            asyncio.to_thread(print, "thread-one"),
            asyncio.to_thread(print, "thread-two"),
        )
        await terminal.drain()
    finally:
        await terminal.close()

    rendered = "".join(output.raw)
    assert "thread-one\n" in rendered
    assert "thread-two\n" in rendered
    assert "thread-onethread-two" not in rendered
    assert "thread-twothread-one" not in rendered


async def test_terminal_ui_coalesces_and_bounds_a_stalled_producer() -> None:
    output = _RecordingOutput()
    terminal = TerminalUI(_Session(output))

    await terminal.start()
    try:
        for index in range(5_000):
            print(f"{index:05d} {'x' * 60}")
        assert terminal.pending_char_count <= MAX_PENDING_CHARS
        await terminal.drain()
    finally:
        await terminal.close()

    rendered = "".join(output.raw)
    assert "00000 " in rendered
    assert "[terminal output skipped:" in rendered
    assert f"{RESET}\n[terminal output skipped:" in rendered
    assert len(output.raw) < 50


async def test_no_color_terminal_skip_marker_has_no_sgr() -> None:
    output = _RecordingOutput()
    terminal = TerminalUI(_Session(output, reset=""))

    await terminal.start()
    try:
        for index in range(5_000):
            print(f"{index:05d} {'x' * 60}")
        await terminal.drain()
    finally:
        await terminal.close()

    marker = next(chunk for chunk in output.raw if "[terminal output skipped:" in chunk)
    assert "\x1b[" not in marker


async def test_terminal_ui_surfaces_write_failure_and_restores_streams() -> None:
    output = _FailingOutput()
    terminal = TerminalUI(_Session(output))
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    await terminal.start()
    print("first line")
    with pytest.raises(RuntimeError, match="terminal output consumer failed"):
        await asyncio.wait_for(terminal.wait_failed(), timeout=1)
    with pytest.raises(RuntimeError, match="terminal output consumer failed"):
        await terminal.drain()
    print("later line")
    with pytest.raises(RuntimeError, match="terminal output consumer failed"):
        await terminal.close()

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


async def test_terminal_ui_is_single_use_even_when_closed_before_start() -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))

    await terminal.close()

    assert terminal.phase is TerminalPhase.CLOSED
    with pytest.raises(RuntimeError, match="cannot start from closed"):
        await terminal.start()


async def test_terminal_ui_rolls_back_partial_logging_rebind_when_start_fails() -> None:
    class FailingRebindHandler(logging.StreamHandler[Any]):
        def setStream(self, stream: Any) -> None:
            del stream
            raise OSError("logging rebind failed")

    output = _RecordingOutput()
    terminal = TerminalUI(_Session(output))
    logger = logging.getLogger("axio-repl-terminal-start-rollback")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    first = logging.StreamHandler(original_stdout)
    failing = FailingRebindHandler(original_stdout)
    previous_handlers = list(logger.handlers)
    previous_propagate = logger.propagate
    logger.handlers[:] = [first, failing]
    logger.propagate = False

    try:
        with pytest.raises(OSError, match="logging rebind failed"):
            await terminal.start()

        assert terminal.phase is TerminalPhase.CLOSED
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
        assert first.stream is original_stdout
        assert failing.stream is original_stdout
        assert terminal._consumer is None or terminal._consumer.done()
        await terminal.close()
    finally:
        logger.handlers[:] = previous_handlers
        logger.propagate = previous_propagate


async def test_terminal_ui_rejects_restart_while_running() -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))
    await terminal.start()
    try:
        with pytest.raises(RuntimeError, match="cannot start from running"):
            await terminal.start()
    finally:
        await terminal.close()


async def test_terminal_ui_fails_posix_check_before_mutating_process_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))
    stdout = sys.stdout
    stderr = sys.stderr
    monkeypatch.setattr(sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="requires POSIX"):
        await terminal.start()

    assert terminal.phase is TerminalPhase.NEW
    assert sys.stdout is stdout
    assert sys.stderr is stderr
    await terminal.close()


async def test_terminal_stream_facade_validates_writes_and_flushes_partial_text() -> None:
    output = _RecordingOutput()
    terminal = TerminalUI(_Session(output))
    await terminal.start()
    try:
        stream = terminal.stdout
        assert stream is not None
        assert stream.writable()
        assert stream.isatty() is False
        assert stream.encoding
        assert stream.errors is not None
        try:
            stream.fileno()
        except OSError:
            pass
        assert stream.write("") == 0
        with pytest.raises(TypeError, match="must be str"):
            stream.write(cast(Any, 1))
        with pytest.raises(AttributeError, match="read-only"):
            stream.encoding = "ascii"
        with pytest.raises(AttributeError, match="read-only"):
            stream.errors = "strict"
        stream.write("line\npartial")
        stream.flush()
        await terminal.drain()
    finally:
        await terminal.close()

    rendered = "".join(output.raw)
    assert "line\n" in rendered
    assert "partial" in rendered


async def test_drain_and_failure_wait_before_start_have_explicit_contracts() -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))

    await terminal.drain()
    assert terminal.pending_char_count == 0
    terminal.submit(OutputFrame(""))
    with pytest.raises(RuntimeError, match="has not started"):
        await terminal.wait_failed()


async def test_terminal_ui_concurrent_close_is_idempotent() -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))
    await terminal.start()
    print("before close")

    await asyncio.gather(terminal.close(), terminal.close())

    assert terminal.phase is TerminalPhase.CLOSED


async def test_submit_before_start_and_retained_wrapper_after_close_use_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))
    terminal.submit(OutputFrame("before start\n"))

    await terminal.start()
    retained = terminal.stdout
    assert retained is not None
    await terminal.close()
    retained.write("after close\n")
    retained.flush()

    captured = capsys.readouterr().out
    assert "before start\n" in captured
    assert "after close\n" in captured


async def test_write_through_retained_wrapper_during_close_is_flushed_after_restore(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))
    await terminal.start()
    retained = terminal.stdout
    assert retained is not None
    restore = terminal._restore_terminal

    def write_during_restore() -> None:
        retained.write("late during close\n")
        restore()

    monkeypatch.setattr(terminal, "_restore_terminal", write_during_restore)
    await terminal.close()

    assert "late during close\n" in capsys.readouterr().out


async def test_late_output_overflow_is_reported_after_terminal_restore(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))
    await terminal.start()
    retained = terminal.stderr
    assert retained is not None
    restore = terminal._restore_terminal

    def overflow_during_restore() -> None:
        retained.write("x" * (64 * 1024 + 1) + "\n")
        restore()

    monkeypatch.setattr(terminal, "_restore_terminal", overflow_during_restore)
    await terminal.close()

    error = capsys.readouterr().err
    assert "late terminal output skipped: 1 frame" in error
    assert error.startswith(RESET)


async def test_no_color_late_output_skip_marker_has_no_sgr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = TerminalUI(_Session(_RecordingOutput(), reset=""))
    await terminal.start()
    retained = terminal.stderr
    assert retained is not None
    restore = terminal._restore_terminal

    def overflow_during_restore() -> None:
        retained.write("x" * (64 * 1024 + 1) + "\n")
        restore()

    monkeypatch.setattr(terminal, "_restore_terminal", overflow_during_restore)
    await terminal.close()

    error = capsys.readouterr().err
    assert "late terminal output skipped: 1 frame" in error
    assert "\x1b[" not in error


async def test_restore_failure_still_closes_ingress_and_process_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))
    stdout = sys.stdout
    stderr = sys.stderr
    await terminal.start()

    def fail_restore() -> None:
        raise OSError("restore failed")

    monkeypatch.setattr(terminal, "_restore_terminal", fail_restore)
    with pytest.raises(OSError, match="restore failed"):
        await terminal.close()

    assert terminal.phase is TerminalPhase.CLOSED
    assert sys.stdout is stdout
    assert sys.stderr is stderr


async def test_flush_failure_still_drains_and_closes_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))
    await terminal.start()
    assert terminal.stdout is not None

    def fail_flush() -> None:
        raise OSError("flush failed")

    monkeypatch.setattr(terminal.stdout, "flush_all", fail_flush)
    with pytest.raises(OSError, match="flush failed"):
        await terminal.close()

    assert terminal.phase is TerminalPhase.CLOSED


async def test_late_flush_failure_still_marks_terminal_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))
    await terminal.start()

    def fail_late_flush(_ingress: object) -> None:
        raise OSError("late flush failed")

    monkeypatch.setattr(terminal, "_flush_late_output", fail_late_flush)
    with pytest.raises(OSError, match="late flush failed"):
        await terminal.close()

    assert terminal.phase is TerminalPhase.CLOSED


async def test_close_does_not_overwrite_streams_replaced_by_the_application() -> None:
    terminal = TerminalUI(_Session(_RecordingOutput()))
    stdout = sys.stdout
    stderr = sys.stderr
    await terminal.start()
    sys.stdout = stdout
    sys.stderr = stderr

    await terminal.close()

    assert sys.stdout is stdout
    assert sys.stderr is stderr


async def test_consumer_task_failure_is_propagated_while_process_state_is_restored() -> None:
    class CrashingTerminal(TerminalUI):
        async def _consume(self) -> None:
            raise RuntimeError("consumer task crashed")

    terminal = CrashingTerminal(_Session(_RecordingOutput()))
    stdout = sys.stdout
    stderr = sys.stderr
    await terminal.start()
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="consumer task crashed"):
        await terminal.close()

    assert terminal.phase is TerminalPhase.CLOSED
    assert sys.stdout is stdout
    assert sys.stderr is stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal and termios test")
@pytest.mark.parametrize("powerline", [False, True])
@pytest.mark.parametrize("theme", [DEFAULT_THEME, MONOCHROME_THEME], ids=("default", "monochrome"))
async def test_terminal_ui_preserves_primary_buffer_and_restores_termios_in_both_prompt_styles(
    monkeypatch: pytest.MonkeyPatch,
    powerline: bool,
    theme: TerminalTheme,
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    master_fd, slave_fd = os.openpty()
    before = termios.tcgetattr(slave_fd)
    reader = TextIOWrapper(os.fdopen(os.dup(slave_fd), "rb", buffering=0), encoding="utf-8", newline="")
    writer = TextIOWrapper(os.fdopen(os.dup(slave_fd), "wb", buffering=0), encoding="utf-8", newline="")
    terminal_input = create_input(reader)
    terminal_output = Vt100_Output(
        writer,
        get_size=lambda: Size(rows=12, columns=48),
        term="xterm-256color",
        enable_bell=False,
        enable_cpr=False,
    )

    try:
        with create_app_session(input=terminal_input, output=terminal_output):
            session: Any = _panel.make_session(lambda: "temporary status", theme=theme)
            terminal = TerminalUI(session)
            await terminal.start()
            prompt_factory = _panel.make_prompt_factory(
                "tester",
                powerline=powerline,
                theme=theme,
                now_provider=lambda: datetime(2026, 8, 20, 9, 7),
            )
            prompt = asyncio.create_task(session.prompt_async(prompt_factory()))
            try:
                await asyncio.sleep(0.05)
                assert session.app.is_running
                os.write(master_fd, b"par")
                await asyncio.sleep(0.05)
                if powerline:
                    renderer = ReplRenderer(main_agent_name="axio-repl", powerline=True, theme=theme)
                    await renderer.start_turn(
                        "main",
                        TurnStarted(prompt="inspect"),
                        run_id="main-run",
                        turn_id="main-turn",
                        execution_mode=ExecutionMode.FOREGROUND,
                    )
                    await renderer.render(
                        "main",
                        TextDelta(index=0, delta="ordinary main response"),
                        run_id="main-run",
                        turn_id="main-turn",
                        execution_mode=ExecutionMode.FOREGROUND,
                    )
                    await renderer.render(
                        "main",
                        SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=Usage(1, 1)),
                        run_id="main-run",
                        turn_id="main-turn",
                        execution_mode=ExecutionMode.FOREGROUND,
                    )
                else:
                    print("asynchronous output")
                await terminal.drain()
                os.write(master_fd, b"tial\r")
                assert await asyncio.wait_for(prompt, timeout=2) == "partial"
            finally:
                if not prompt.done():
                    prompt.cancel()
                    await asyncio.gather(prompt, return_exceptions=True)
                await terminal.close()

        after = termios.tcgetattr(slave_fd)
        os.set_blocking(master_fd, False)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(master_fd, 64 * 1024)
            except BlockingIOError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        rendered = b"".join(chunks).decode("utf-8", errors="replace")

        if powerline:
            assert " 09:07 tester " in rendered
            assert "\ue0b0" in rendered
            assert "\ue0b2" not in rendered
            assert "tester>" not in rendered
            assert "ordinary main response" in rendered
            assert "agent axio-repl (main)" not in rendered
        else:
            assert "asynchronous output" in rendered
            assert "09:07 tester> " in rendered
        assert "\x1b[J" in rendered
        assert "\x1b[?1049h" not in rendered
        assert after[1] == before[1]
        assert termios.ICANON & after[3] == termios.ICANON & before[3]
        assert termios.ECHO & after[3] == termios.ECHO & before[3]
    finally:
        terminal_input.close()
        reader.close()
        writer.close()
        os.close(master_fd)
        os.close(slave_fd)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal layout test")
async def test_tall_terminal_input_window_is_compact_and_grows_with_multiline_wrapped_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    master_fd, slave_fd = os.openpty()
    reader = TextIOWrapper(os.fdopen(os.dup(slave_fd), "rb", buffering=0), encoding="utf-8", newline="")
    writer = TextIOWrapper(os.fdopen(os.dup(slave_fd), "wb", buffering=0), encoding="utf-8", newline="")
    terminal_input = create_input(reader)
    terminal_output = Vt100_Output(
        writer,
        get_size=lambda: Size(rows=80, columns=120),
        term="xterm-256color",
        enable_bell=False,
        enable_cpr=False,
    )

    try:
        with create_app_session(input=terminal_input, output=terminal_output):
            session: Any = _panel.make_session(lambda: "status")
            prompt = asyncio.create_task(session.prompt_async(_panel.prompt_message("09:07 tester")))
            try:
                await asyncio.sleep(0.05)
                compact_height = session.app.renderer._last_screen.height
                assert compact_height <= 3

                content = "first line\n" + ("wrapped " * 40)
                session.default_buffer.text = content
                session.app.invalidate()
                await asyncio.sleep(0.05)
                expanded_height = session.app.renderer._last_screen.height

                assert compact_height < expanded_height < 20
                os.write(master_fd, b"\r")
                assert await asyncio.wait_for(prompt, timeout=2) == content
            finally:
                if not prompt.done():
                    prompt.cancel()
                    await asyncio.gather(prompt, return_exceptions=True)
    finally:
        terminal_input.close()
        reader.close()
        writer.close()
        os.close(master_fd)
        os.close(slave_fd)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal and termios test")
async def test_escape_during_redraw_stays_raw_without_submitting_editor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    master_fd, slave_fd = os.openpty()
    before = termios.tcgetattr(slave_fd)
    reader = TextIOWrapper(os.fdopen(os.dup(slave_fd), "rb", buffering=0), encoding="utf-8", newline="")
    writer = TextIOWrapper(os.fdopen(os.dup(slave_fd), "wb", buffering=0), encoding="utf-8", newline="")
    terminal_input = create_input(reader)
    write_started = threading.Event()
    input_modes: list[list[int | list[bytes | int]]] = []

    class _SlowOutput(Vt100_Output):
        def write_raw(self, data: str) -> None:
            if "redraw marker" in data:
                write_started.set()
                time.sleep(0.15)
            super().write_raw(data)

    terminal_output = _SlowOutput(
        writer,
        get_size=lambda: Size(rows=12, columns=48),
        term="xterm-256color",
        enable_bell=False,
        enable_cpr=False,
    )
    interrupts: list[int] = []
    finish_turn = asyncio.Event()

    async def simulated_turn() -> None:
        print("redraw marker")
        await terminal.drain()
        await finish_turn.wait()

    def send_escape_during_write() -> None:
        if write_started.wait(timeout=1):
            input_modes.append(termios.tcgetattr(slave_fd))
            os.write(master_fd, b"\x1b")

    try:
        with create_app_session(input=terminal_input, output=terminal_output):
            session: Any = _panel.make_session(lambda: "temporary status", on_interrupt=lambda: interrupts.append(1))
            terminal = TerminalUI(session)
            await terminal.start()
            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            sender = threading.Thread(target=send_escape_during_write)
            sender.start()
            turn: asyncio.Task[None] | None = None
            try:
                await asyncio.sleep(0.05)
                os.write(master_fd, b"queued message")
                await asyncio.sleep(0.05)
                turn = asyncio.create_task(simulated_turn())
                for _ in range(100):
                    if interrupts:
                        break
                    await asyncio.sleep(0.01)
                assert interrupts == [1]
                assert prompt.done() is False
                assert not turn.done()
                os.write(master_fd, b"\r")
                assert await asyncio.wait_for(prompt, timeout=1) == "queued message"
                finish_turn.set()
                await turn
            finally:
                finish_turn.set()
                sender.join(timeout=1)
                if turn is not None and not turn.done():
                    turn.cancel()
                    await asyncio.gather(turn, return_exceptions=True)
                if not prompt.done():
                    prompt.cancel()
                    await asyncio.gather(prompt, return_exceptions=True)
                await terminal.close()

        assert len(input_modes) == 1
        during_redraw = input_modes[0]
        local_flags = cast(int, during_redraw[3])
        assert termios.ICANON & local_flags == 0
        assert termios.ECHO & local_flags == 0
        assert during_redraw[1] == before[1]

        os.set_blocking(master_fd, False)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(master_fd, 64 * 1024)
            except BlockingIOError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        assert b"^[" not in b"".join(chunks)
    finally:
        terminal_input.close()
        reader.close()
        writer.close()
        os.close(master_fd)
        os.close(slave_fd)
