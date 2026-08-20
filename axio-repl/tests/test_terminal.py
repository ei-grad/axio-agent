from __future__ import annotations

import asyncio
import logging
import os
import sys
import termios
import threading
import time
from datetime import UTC, datetime
from io import TextIOWrapper
from typing import Any, cast

import pytest
from axio.events import Error, SessionEndEvent, TextDelta, ToolInputDelta, ToolResult, ToolUseStart
from axio.types import StopReason, Usage
from axio_tools_agents.runtime import ExecutionMode, TurnStarted
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output

from axio_repl import ReplRenderer, _panel, _read_input_async
from axio_repl._input import InputSubmitted, SubmissionDisposition
from axio_repl._terminal import MAX_PENDING_CHARS, RESET, OutputFrame, TerminalPhase, TerminalUI
from axio_repl._terminal_sanitizer import sanitize_terminal_text
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


async def test_no_color_active_tool_argument_frames_have_no_sgr() -> None:
    from axio_repl._theme import NO_COLOR_THEME

    output = _RecordingOutput()
    terminal = TerminalUI(_Session(output, reset=""))
    renderer = ReplRenderer(theme=NO_COLOR_THEME)

    await terminal.start()
    try:
        renderer.set_input_active(True)
        await renderer.render("main", ToolUseStart(index=0, tool_use_id="call", name="write_file"))
        await terminal.drain()
        output.raw.clear()

        await renderer.render(
            "main",
            ToolInputDelta(index=0, tool_use_id="call", partial_json='{"path":"/tmp/stream-'),
        )
        await terminal.drain()
        assert not any("/tmp/stream-" in chunk for chunk in output.raw)
        await renderer.render(
            "main",
            ToolInputDelta(index=0, tool_use_id="call", partial_json='demo"'),
        )
        await terminal.drain()
        first_frame = next(chunk for chunk in output.raw if "/tmp/stream-" in chunk)
        assert "\x1b[" not in first_frame

        output.raw.clear()
        await renderer.render(
            "main",
            ToolInputDelta(index=0, tool_use_id="call", partial_json=',"content":"value"}'),
        )
        await terminal.drain()
        second_frame = next(chunk for chunk in output.raw if "content" in chunk and "value" in chunk)
        assert "\x1b[" not in second_frame
    finally:
        renderer.set_input_active(False)
        await terminal.close()


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
            assert " tester " in rendered
            assert "\ue0b0" in rendered
            assert "\ue0b2" not in rendered
            assert "tester>" not in rendered
            assert "ordinary main response" in rendered
            assert "agent axio-repl (main)" not in rendered
            if theme is DEFAULT_THEME:
                assert "\x1b[0;30;107;1m tester \x1b[0;97m\ue0b0\x1b[0m par" in rendered
        else:
            assert "asynchronous output" in rendered
            assert "tester> " in rendered
            if theme is DEFAULT_THEME:
                assert "\x1b[0;30;107;1mtester> \x1b[0mpar" in rendered
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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal submission test")
async def test_enter_replaces_the_temporary_prompt_with_one_timestamped_powerline_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    master_fd, slave_fd = os.openpty()
    reader = TextIOWrapper(os.fdopen(os.dup(slave_fd), "rb", buffering=0), encoding="utf-8", newline="")
    writer = TextIOWrapper(os.fdopen(os.dup(slave_fd), "wb", buffering=0), encoding="utf-8", newline="")
    terminal_input = create_input(reader)
    terminal_output = Vt100_Output(
        writer,
        get_size=lambda: Size(rows=20, columns=80),
        term="xterm-256color",
        enable_bell=False,
        enable_cpr=False,
    )
    accepted_time = datetime(2026, 8, 20, 12, 41, tzinfo=UTC)
    admission_started = asyncio.Event()
    release_admission = asyncio.Event()
    clock_calls: list[datetime] = []

    def now() -> datetime:
        clock_calls.append(accepted_time)
        return accepted_time

    async def admit(text: str, target_agent_id: str, reserved_seq: int | None) -> InputSubmitted:
        admission_started.set()
        await release_admission.wait()
        return InputSubmitted(
            text=text,
            target_agent_id=target_agent_id,
            disposition=SubmissionDisposition.PENDING,
            input_id="input-1",
            arrival_seq=1,
        )

    try:
        with create_app_session(input=terminal_input, output=terminal_output):
            session: Any = _panel.make_session(
                lambda: "temporary status",
                reserve_sequence=lambda: 1,
                accepted_at_provider=now,
                theme=DEFAULT_THEME,
            )
            setattr(session, "_axio_terminal_reset", DEFAULT_THEME.reset)
            terminal = TerminalUI(session)
            renderer = ReplRenderer(
                powerline=True,
                theme=DEFAULT_THEME,
                effective_username="tester",
            )
            await terminal.start()
            input_task = asyncio.create_task(
                _read_input_async(
                    session,
                    renderer,
                    lambda: None,
                    admit,
                    prompt_factory=_panel.make_prompt_factory(
                        "tester",
                        powerline=True,
                        theme=DEFAULT_THEME,
                    ),
                )
            )
            try:
                await asyncio.sleep(0.05)
                await renderer.render("main", TextDelta(index=0, delta="partial model output"))
                os.write(master_fd, b"ping\r")
                await asyncio.wait_for(admission_started.wait(), timeout=1)
                assert clock_calls == [accepted_time]
                assert not input_task.done()

                release_admission.set()
                submitted = await asyncio.wait_for(input_task, timeout=1)
                await renderer.render("main", TextDelta(index=0, delta="remaining model output"))
                await terminal.drain()
                assert submitted.text == "ping"
            finally:
                release_admission.set()
                if not input_task.done():
                    input_task.cancel()
                    await asyncio.gather(input_task, return_exceptions=True)
                await terminal.close()

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

        assert rendered.count("12:41 tester") == 1
        assert rendered.index("partial model output") < rendered.index("12:41 tester")
        assert rendered.index("12:41 tester") < rendered.index("remaining model output")
        assert "\x1b[1;30;107m 12:41 tester \x1b[22;97;49m\ue0b0\x1b[0m ping\r\n" in rendered
        assert "\x1b[?1049h" not in rendered
    finally:
        terminal_input.close()
        reader.close()
        writer.close()
        os.close(master_fd)
        os.close(slave_fd)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal argument streaming test")
@pytest.mark.parametrize("powerline", [False, True])
async def test_tool_arguments_reach_the_terminal_only_at_field_and_line_boundaries_with_prompt_open(
    monkeypatch: pytest.MonkeyPatch,
    powerline: bool,
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    master_fd, slave_fd = os.openpty()
    reader = TextIOWrapper(os.fdopen(os.dup(slave_fd), "rb", buffering=0), encoding="utf-8", newline="")
    writer = TextIOWrapper(os.fdopen(os.dup(slave_fd), "wb", buffering=0), encoding="utf-8", newline="")
    terminal_input = create_input(reader)
    terminal_output = Vt100_Output(
        writer,
        get_size=lambda: Size(rows=24, columns=100),
        term="xterm-256color",
        enable_bell=False,
        enable_cpr=False,
    )
    os.set_blocking(master_fd, False)

    async def read_stage() -> str:
        await asyncio.sleep(0.02)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(master_fd, 64 * 1024)
            except BlockingIOError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")

    try:
        with create_app_session(input=terminal_input, output=terminal_output):
            session: Any = _panel.make_session(lambda: "temporary status", theme=DEFAULT_THEME)
            setattr(session, "_axio_terminal_reset", DEFAULT_THEME.reset)
            terminal = TerminalUI(session)
            renderer = ReplRenderer(
                powerline=powerline,
                theme=DEFAULT_THEME,
                effective_username="tester",
            )
            await terminal.start()
            prompt = asyncio.create_task(
                session.prompt_async(_panel.make_prompt_factory("tester", powerline=powerline)())
            )
            try:
                await asyncio.sleep(0.05)
                await read_stage()
                renderer.set_input_active(True)
                await renderer.render("main", ToolUseStart(index=0, tool_use_id="call", name="write_file"))
                await terminal.drain()
                title_stage = await read_stage()

                await renderer.render(
                    "main",
                    ToolInputDelta(index=0, tool_use_id="call", partial_json='{"path":"/tmp/stream-'),
                )
                await terminal.drain()
                path_partial_stage = await read_stage()

                await renderer.render(
                    "main",
                    ToolInputDelta(
                        index=0,
                        tool_use_id="call",
                        partial_json='demo.txt","content":"alpha-one',
                    ),
                )
                await terminal.drain()
                path_complete_stage = await read_stage()

                await renderer.render(
                    "main",
                    ToolInputDelta(index=0, tool_use_id="call", partial_json="\\nbeta"),
                )
                await terminal.drain()
                first_line_stage = await read_stage()

                await renderer.render(
                    "main",
                    ToolInputDelta(index=0, tool_use_id="call", partial_json="-two\\ngamma"),
                )
                await terminal.drain()
                second_line_stage = await read_stage()

                await renderer.submitted(
                    "queued input",
                    datetime(2026, 8, 20, 12, 41, tzinfo=UTC),
                )
                await terminal.drain()
                submitted_stage = await read_stage()

                await renderer.render(
                    "main",
                    ToolInputDelta(index=0, tool_use_id="call", partial_json=" after"),
                )
                await terminal.drain()
                incomplete_after_insert_stage = await read_stage()
                await renderer.incoming("peer body", agent_id="child", agent_name="peer")
                await terminal.drain()
                incoming_stage = await read_stage()

                await renderer.render(
                    "main",
                    ToolInputDelta(index=0, tool_use_id="call", partial_json='","long":"'),
                )
                long_value = "long-" + ("x" * 400)
                for character in long_value:
                    await renderer.render(
                        "main",
                        ToolInputDelta(index=0, tool_use_id="call", partial_json=character),
                    )
                await terminal.drain()
                long_partial_stage = await read_stage()
                await renderer.render(
                    "main",
                    ToolInputDelta(index=0, tool_use_id="call", partial_json='"}'),
                )
                await terminal.drain()
                long_complete_stage = await read_stage()

                await renderer.render(
                    "main",
                    ToolResult(
                        tool_use_id="call",
                        name="write_file",
                        is_error=False,
                        content="Wrote 14719 bytes to /tmp/stream-demo.txt",
                    ),
                )
                await terminal.drain()
                result_stage = await read_stage()

                await renderer.render(
                    "main",
                    ToolUseStart(index=1, tool_use_id="patch-call", name="patch_file"),
                )
                await renderer.render(
                    "main",
                    ToolInputDelta(
                        index=1,
                        tool_use_id="patch-call",
                        partial_json=(
                            '{"path":"src/service.py","from_line":3,"to_line":3,"content":"        return 2\\n"}'
                        ),
                    ),
                )
                await terminal.drain()
                patch_argument_stage = await read_stage()
                await renderer.render(
                    "main",
                    ToolResult(
                        tool_use_id="patch-call",
                        name="patch_file",
                        is_error=False,
                        content=(
                            "+1 -1\n@@ -1,3 +1,3 @@ Service.run\n class Service:\n-    return 1\n+    return 2\n"
                        ),
                    ),
                )
                await terminal.drain()
                patch_result_stage = await read_stage()

                await renderer.render(
                    "main",
                    ToolUseStart(index=2, tool_use_id="tool-result-close", name="list_files"),
                )
                await terminal.drain()
                await read_stage()
                await renderer.render(
                    "main",
                    ToolInputDelta(
                        index=2,
                        tool_use_id="tool-result-close",
                        partial_json='{"path":"partial-before-result',
                    ),
                )
                await terminal.drain()
                result_partial_stage = await read_stage()
                await renderer.render(
                    "main",
                    ToolResult(
                        tool_use_id="tool-result-close",
                        name="list_files",
                        is_error=False,
                        content="partial result",
                    ),
                )
                await terminal.drain()
                forced_result_stage = await read_stage()

                await renderer.render(
                    "main",
                    ToolUseStart(index=3, tool_use_id="error-call", name="list_files"),
                )
                await terminal.drain()
                await read_stage()
                await renderer.render(
                    "main",
                    ToolInputDelta(index=3, tool_use_id="error-call", partial_json='{"path":"'),
                )
                dsml_value = "<|DSML|>\nsecond natural line\nunterminated-tail"
                for character in dsml_value.replace("\n", "\\n"):
                    await renderer.render(
                        "main",
                        ToolInputDelta(index=3, tool_use_id="error-call", partial_json=character),
                    )
                await terminal.drain()
                dsml_streaming_stage = await read_stage()
                await renderer.render(
                    "main",
                    ToolInputDelta(index=3, tool_use_id="error-call", partial_json="\x1b[3"),
                )
                await renderer.render(
                    "main",
                    ToolInputDelta(index=3, tool_use_id="error-call", partial_json="1m"),
                )
                await terminal.drain()
                sanitizer_only_stage = await read_stage()
                await renderer.render("main", Error(exception=RuntimeError("provider stream failed")))
                await terminal.drain()
                error_stage = await read_stage()
            finally:
                renderer.set_input_active(False)
                if not prompt.done():
                    prompt.cancel()
                    await asyncio.gather(prompt, return_exceptions=True)
                await terminal.close()
                late_stage = await read_stage()

        assert "write_file" not in title_stage
        assert path_partial_stage == ""
        assert "/tmp/stream-demo.txt" in path_complete_stage
        assert "content" not in path_complete_stage and "alpha-one" not in path_complete_stage
        assert "content" in first_line_stage and "alpha-one" in first_line_stage
        assert "beta" not in first_line_stage
        assert "beta-two" in second_line_stage and "gamma" not in second_line_stage
        submitted = sanitize_terminal_text(submitted_stage)
        assert submitted.index("gamma") < submitted.index("12:41 tester") < submitted.index("queued input")
        assert incomplete_after_insert_stage == ""
        incoming = sanitize_terminal_text(incoming_stage)
        assert incoming.index(" after") < incoming.index("peer body")
        assert long_partial_stage == ""
        assert long_value in sanitize_terminal_text(long_complete_stage)
        assert "Wrote 14719 bytes" not in result_stage
        assert patch_argument_stage.count("src/service.py") == 1
        assert "src/service.py" not in patch_result_stage
        assert "+1 -1" in patch_result_stage
        assert "Service.run" in patch_result_stage
        assert "return 1" in patch_result_stage and "return 2" in patch_result_stage
        assert result_partial_stage == ""
        forced_result = sanitize_terminal_text(forced_result_stage)
        assert forced_result.index("path: partial-before-result") < forced_result.index("partial result")
        dsml_streaming = sanitize_terminal_text(dsml_streaming_stage)
        assert dsml_streaming.count("path:") == 1
        assert "<|DSML|>\nsecond natural line\n" in dsml_streaming
        assert "unterminated-tail" not in dsml_streaming
        assert sanitizer_only_stage == ""
        assert "unterminated-tail" in sanitize_terminal_text(error_stage)
        assert "provider stream failed" in error_stage
        assert "(continued)" not in error_stage
        assert "(continued)" not in late_stage

        combined = "".join(
            (
                title_stage,
                path_complete_stage,
                first_line_stage,
                second_line_stage,
                submitted_stage,
                incoming_stage,
                long_complete_stage,
                result_stage,
                patch_argument_stage,
                patch_result_stage,
                forced_result_stage,
                dsml_streaming_stage,
                error_stage,
                late_stage,
            )
        )
        sanitized = sanitize_terminal_text(combined)
        assert sanitized.count("content:") == 2
        assert sanitized.count("long:") == 1
        assert sanitized.count("partial-before-result") == 1
        assert sanitized.count("<|DSML|>") == 1
        assert sanitized.count("src/service.py") == 1
        assert "(continued)" not in sanitized
        assert sanitized.index("/tmp/stream-demo.txt") < sanitized.index("alpha-one")
        assert sanitized.index("alpha-one") < sanitized.index("beta-two") < sanitized.index("gamma")
        assert "\x1b[?1049h" not in combined
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
            prompt = asyncio.create_task(session.prompt_async(_panel.prompt_message("tester")))
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
