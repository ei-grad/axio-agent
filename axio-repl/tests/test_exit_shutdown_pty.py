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
    journal_root: Path
    project_history: Path
    legacy_history: Path


def _spawn_harness(tmp_path: Path) -> _Harness:
    master_fd, slave_fd = os.openpty()
    terminal_size = struct.pack("HHHH", 24, 100, 0, 0)
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, terminal_size)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, terminal_size)
    original_termios = termios.tcgetattr(slave_fd)
    started = tmp_path / "transport-started"
    finalized = tmp_path / "transport-finalized"
    prompt_count = tmp_path / "prompt-count"
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


def _assert_journal_shutdown(harness: _Harness) -> None:
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
    assert event["reason"] == "sigint"
    assert not any("Draining active/pending work" in repr(record) for record in records)


def _assert_history_isolated(harness: _Harness) -> None:
    assert harness.legacy_history.read_text() == "sentinel legacy history\n"
    assert "run forever" not in harness.legacy_history.read_text()
    assert "run forever" in harness.project_history.read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX controlling-terminal test")
@pytest.mark.parametrize("interrupt", ["signal", "terminal"], ids=("external-sigint", "raw-ctrl-c"))
def test_sigint_after_double_eof_cancels_drain_and_restores_terminal(tmp_path: Path, interrupt: str) -> None:
    harness = _spawn_harness(tmp_path)
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

        os.write(harness.master_fd, b"\x04")
        time.sleep(0.05)
        os.write(harness.master_fd, b"\x04")
        _wait_until(
            harness,
            output,
            lambda: b"Ctrl-C cancels." in output and termios.tcgetattr(harness.slave_fd) == harness.original_termios,
        )
        assert os.waitpid(harness.pid, os.WNOHANG) == (0, 0)

        if interrupt == "signal":
            os.kill(harness.pid, signal.SIGINT)
        else:
            os.write(harness.master_fd, b"\x03")

        exit_code = _wait_for_exit(harness, output)
        waited = True
        assert exit_code == 0
        assert harness.finalized.exists()
        assert termios.tcgetattr(harness.slave_fd) == harness.original_termios
        _assert_journal_shutdown(harness)
        _assert_history_isolated(harness)
    finally:
        if not waited:
            waited_pid, _ = os.waitpid(harness.pid, os.WNOHANG)
            if waited_pid == 0:
                os.kill(harness.pid, signal.SIGKILL)
                os.waitpid(harness.pid, 0)
        os.close(harness.master_fd)
        os.close(harness.slave_fd)
