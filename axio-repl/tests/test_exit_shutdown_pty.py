from __future__ import annotations

import fcntl
import os
import signal
import struct
import sys
import termios
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from axio_repl._history import project_history_path
from axio_repl._journal import SEMANTIC_FILENAME, read_journal


@dataclass(frozen=True, slots=True)
class _Harness:
    pid: int
    master_fd: int
    slave_fd: int
    original_termios: list[object]
    started: Path
    finalized: Path
    prompt_count: Path
    call_count: Path
    journal_root: Path
    project_history: Path
    legacy_history: Path


def _spawn_harness(tmp_path: Path, *, mode: str) -> _Harness:
    master_fd, slave_fd = os.openpty()
    terminal_size = struct.pack("HHHH", 24, 100, 0, 0)
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, terminal_size)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, terminal_size)
    original_termios = termios.tcgetattr(slave_fd)
    started = tmp_path / "transport-started"
    finalized = tmp_path / "transport-finalized"
    prompt_count = tmp_path / "prompt-count"
    call_count = tmp_path / "call-count"
    journal_root = tmp_path / "journals"
    state_home = tmp_path / "state"
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    legacy_history = fake_home / ".axio_repl_history"
    legacy_history.write_text("sentinel legacy history\n")
    project_history = project_history_path(tmp_path, state_home=state_home)
    harness_script = Path(__file__).with_name("_exit_harness.py")
    environment = os.environ.copy()
    environment.pop("AXIO_REPL_AGENT", None)
    environment.pop("AXIO_CONFIG_DIR", None)
    environment.update(
        {
            "TERM": "xterm-256color",
            "PYTHONUNBUFFERED": "1",
            "AXIO_EXIT_HARNESS_STARTED": str(started),
            "AXIO_EXIT_HARNESS_FINALIZED": str(finalized),
            "AXIO_EXIT_HARNESS_JOURNAL_ROOT": str(journal_root),
            "AXIO_EXIT_HARNESS_CONFIG_ROOT": str(tmp_path / "config"),
            "AXIO_EXIT_HARNESS_PEER_ROOT": str(tmp_path / "peers"),
            "AXIO_EXIT_HARNESS_PROMPT_COUNT": str(prompt_count),
            "AXIO_EXIT_HARNESS_CALL_COUNT": str(call_count),
            "AXIO_EXIT_HARNESS_MODE": mode,
            "HOME": str(fake_home),
            "XDG_STATE_HOME": str(state_home),
        }
    )
    pid = os.fork()
    if pid == 0:
        try:
            os.setsid()
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
            os.tcsetpgrp(slave_fd, os.getpgrp())
            for target_fd in (0, 1, 2):
                os.dup2(slave_fd, target_fd)
            os.close(master_fd)
            if slave_fd > 2:
                os.close(slave_fd)
            os.chdir(tmp_path)
            os.execve(sys.executable, [sys.executable, str(harness_script)], environment)
        except BaseException:
            os._exit(127)
    os.set_blocking(master_fd, False)
    return _Harness(
        pid=pid,
        master_fd=master_fd,
        slave_fd=slave_fd,
        original_termios=original_termios,
        started=started,
        finalized=finalized,
        prompt_count=prompt_count,
        call_count=call_count,
        journal_root=journal_root,
        project_history=project_history,
        legacy_history=legacy_history,
    )


def _read_available(harness: _Harness, output: bytearray) -> None:
    while True:
        try:
            chunk = os.read(harness.master_fd, 64 * 1024)
        except BlockingIOError:
            return
        except OSError:
            return
        if not chunk:
            return
        output.extend(chunk)
        for _ in range(chunk.count(b"\x1b[6n")):
            os.write(harness.master_fd, b"\x1b[1;1R")


def _prompt_count(harness: _Harness) -> int:
    try:
        return int(harness.prompt_count.read_text())
    except (FileNotFoundError, ValueError):
        return 0


def _transport_call_count(harness: _Harness) -> int:
    try:
        return int(harness.call_count.read_text())
    except (FileNotFoundError, ValueError):
        return 0


def _wait_until(
    harness: _Harness,
    output: bytearray,
    predicate: Callable[[], bool],
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _read_available(harness, output)
        if predicate():
            return
        time.sleep(0.01)
    _read_available(harness, output)
    raise AssertionError(output.decode("utf-8", errors="replace"))


def _wait_for_exit(harness: _Harness, output: bytearray, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _read_available(harness, output)
        waited_pid, status = os.waitpid(harness.pid, os.WNOHANG)
        if waited_pid == harness.pid:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.01)
    raise AssertionError(output.decode("utf-8", errors="replace"))


def _assert_journal_shutdown(harness: _Harness, *, reason: str = "eof") -> None:
    semantic_paths = list(harness.journal_root.rglob(SEMANTIC_FILENAME))
    assert len(semantic_paths) == 1
    records = read_journal(semantic_paths[0]).records
    kinds = [record["kind"] for record in records]
    assert "agent_stopped" in kinds
    assert kinds[-1] == "session_end"
    shutdown = next(record for record in records if record["kind"] == "shutdown_recorded")
    payload = shutdown["payload"]
    assert isinstance(payload, dict)
    event = payload["event"]
    assert isinstance(event, dict)
    assert event["reason"] == reason
    assert not any("Draining active/pending work" in repr(record) for record in records)


def _assert_history_isolated(harness: _Harness) -> None:
    assert harness.legacy_history.read_text() == "sentinel legacy history\n"
    assert "run forever" not in harness.legacy_history.read_text()
    assert "run forever" in harness.project_history.read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX controlling-terminal test")
@pytest.mark.parametrize(
    "mode",
    ["provider", "provider_resistant", "tool", "detached"],
    ids=("active-provider", "provider-resistant-cleanup", "between-provider-iterations", "detached-tool"),
)
def test_single_eof_cancels_active_work_and_restores_terminal(tmp_path: Path, mode: str) -> None:
    harness = _spawn_harness(tmp_path, mode=mode)
    output = bytearray()
    waited = False
    try:
        _wait_until(harness, output, lambda: b"exit-test>" in output)
        os.write(harness.master_fd, b"run forever\r")
        _wait_until(harness, output, harness.started.exists)
        if mode == "detached":
            _wait_until(harness, output, lambda: _transport_call_count(harness) == 2)
        _wait_until(
            harness,
            output,
            lambda: _prompt_count(harness) >= 2 and termios.tcgetattr(harness.slave_fd) != harness.original_termios,
        )

        os.write(harness.master_fd, b"\x04")
        exit_code = _wait_for_exit(harness, output)
        waited = True
        assert exit_code == 0
        assert harness.finalized.exists()
        assert harness.call_count.read_text() == ("2" if mode == "detached" else "1")
        assert termios.tcgetattr(harness.slave_fd) == harness.original_termios
        _assert_journal_shutdown(harness)
        _assert_history_isolated(harness)
        assert b"Press Ctrl-D again" not in output
        assert b"Main turn interrupted" not in output
        records = read_journal(next(harness.journal_root.rglob(SEMANTIC_FILENAME))).records
        assert not any(str(record["kind"]).startswith("interruption_") for record in records)
        if mode == "tool":
            assert sum(record["kind"] == "tool_result" for record in records) == 1
            tool_result = next(record for record in records if record["kind"] == "tool_result")
            assert "[cancelled: eof shutdown]" in repr(tool_result)
            assert "interrupted by user" not in repr(tool_result)
            assert not any(
                record["kind"] == "input_received" and "deferred-tool" in repr(record) for record in records
            )
            shutdown = next(record for record in records if record["kind"] == "shutdown_recorded")
            payload = shutdown["payload"]
            assert isinstance(payload, dict)
            event = payload["event"]
            assert isinstance(event, dict)
            assert event["deferred_tool_use_ids"] == []
    finally:
        if not waited:
            waited_pid, _ = os.waitpid(harness.pid, os.WNOHANG)
            if waited_pid == 0:
                os.kill(harness.pid, signal.SIGKILL)
                os.waitpid(harness.pid, 0)
        os.close(harness.master_fd)
        os.close(harness.slave_fd)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX controlling-terminal test")
def test_single_eof_exits_idle_repl_and_restores_terminal(tmp_path: Path) -> None:
    harness = _spawn_harness(tmp_path, mode="provider")
    output = bytearray()
    waited = False
    try:
        _wait_until(
            harness,
            output,
            lambda: b"exit-test>" in output and termios.tcgetattr(harness.slave_fd) != harness.original_termios,
        )

        os.write(harness.master_fd, b"\x04")
        exit_code = _wait_for_exit(harness, output)
        waited = True
        assert exit_code == 0
        assert not harness.started.exists()
        assert not harness.finalized.exists()
        assert not harness.call_count.exists()
        assert termios.tcgetattr(harness.slave_fd) == harness.original_termios
        _assert_journal_shutdown(harness)
        assert b"Press Ctrl-D again" not in output
    finally:
        if not waited:
            waited_pid, _ = os.waitpid(harness.pid, os.WNOHANG)
            if waited_pid == 0:
                os.kill(harness.pid, signal.SIGKILL)
                os.waitpid(harness.pid, 0)
        os.close(harness.master_fd)
        os.close(harness.slave_fd)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX controlling-terminal test")
@pytest.mark.parametrize("interrupt", ["signal", "terminal"], ids=("external-sigint", "raw-ctrl-c"))
def test_sigint_still_cancels_active_provider_and_restores_terminal(tmp_path: Path, interrupt: str) -> None:
    harness = _spawn_harness(tmp_path, mode="provider")
    output = bytearray()
    waited = False
    try:
        _wait_until(harness, output, lambda: b"exit-test>" in output)
        os.write(harness.master_fd, b"run forever\r")
        _wait_until(harness, output, harness.started.exists)
        _wait_until(
            harness,
            output,
            lambda: _prompt_count(harness) >= 2 and termios.tcgetattr(harness.slave_fd) != harness.original_termios,
        )

        if interrupt == "signal":
            os.kill(harness.pid, signal.SIGINT)
        else:
            os.write(harness.master_fd, b"\x03")

        exit_code = _wait_for_exit(harness, output)
        waited = True
        assert exit_code == 0
        assert harness.finalized.exists()
        assert harness.call_count.read_text() == "1"
        assert termios.tcgetattr(harness.slave_fd) == harness.original_termios
        _assert_journal_shutdown(harness, reason="sigint")
        _assert_history_isolated(harness)
    finally:
        if not waited:
            waited_pid, _ = os.waitpid(harness.pid, os.WNOHANG)
            if waited_pid == 0:
                os.kill(harness.pid, signal.SIGKILL)
                os.waitpid(harness.pid, 0)
        os.close(harness.master_fd)
        os.close(harness.slave_fd)
