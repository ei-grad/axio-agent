from __future__ import annotations

import asyncio
import logging
import os
import sys
import termios
import threading
import time
from io import TextIOWrapper
from typing import Any, cast

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output

from axio_repl import _panel
from axio_repl._terminal import MAX_PENDING_CHARS, RESET, TerminalUI


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
    def __init__(self, output: _RecordingOutput) -> None:
        self.app = type("App", (), {"output": output, "is_running": False})()


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
    assert len(output.raw) < 50


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
        print("later line")
    with pytest.raises(RuntimeError, match="terminal output consumer failed"):
        await terminal.close()

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


async def test_write_above_prompt_releases_waiters_when_redraw_fails() -> None:
    class _Renderer:
        def erase(self) -> None:
            pass

        def reset(self) -> None:
            raise OSError("redraw failed")

    app = type(
        "App",
        (),
        {
            "_running_in_terminal_f": None,
            "_running_in_terminal": False,
            "_request_absolute_cursor_position": lambda self: None,
            "_redraw": lambda self: None,
            "is_running": True,
            "output": type("Output", (), {"responds_to_cpr": False})(),
            "renderer": _Renderer(),
        },
    )()

    with pytest.raises(OSError, match="redraw failed"):
        await TerminalUI._write_above_prompt(app, lambda: None)

    assert app._running_in_terminal is False
    assert app._running_in_terminal_f.done()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal and termios test")
async def test_terminal_ui_preserves_primary_buffer_and_restores_termios(
    monkeypatch: pytest.MonkeyPatch,
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
            session: Any = _panel.make_session(lambda: "temporary status")
            terminal = TerminalUI(session)
            await terminal.start()
            prompt = asyncio.create_task(session.prompt_async("repl> "))
            try:
                await asyncio.sleep(0.05)
                assert session.app.is_running
                os.write(master_fd, b"par")
                await asyncio.sleep(0.05)
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

        assert "asynchronous output" in rendered
        assert "\x1b[J" in rendered
        assert "\x1b[?1049h" not in rendered
        assert termios.ICANON & after[3] == termios.ICANON & before[3]
        assert termios.ECHO & after[3] == termios.ECHO & before[3]
    finally:
        terminal_input.close()
        reader.close()
        writer.close()
        os.close(master_fd)
        os.close(slave_fd)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-terminal and termios test")
async def test_escape_during_redraw_stays_raw_and_submits_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    master_fd, slave_fd = os.openpty()
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
            prompt = asyncio.create_task(session.prompt_async("repl> "))
            sender = threading.Thread(target=send_escape_during_write)
            sender.start()
            turn: asyncio.Task[None] | None = None
            try:
                await asyncio.sleep(0.05)
                os.write(master_fd, b"queued message")
                await asyncio.sleep(0.05)
                turn = asyncio.create_task(simulated_turn())
                assert await asyncio.wait_for(prompt, timeout=1) == "queued message"
                assert not turn.done()
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

        assert len(interrupts) == 1
        assert len(input_modes) == 1
        during_redraw = input_modes[0]
        local_flags = cast(int, during_redraw[3])
        assert termios.ICANON & local_flags == 0
        assert termios.ECHO & local_flags == 0

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
