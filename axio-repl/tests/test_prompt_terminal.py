from __future__ import annotations

import asyncio
from typing import Any

import pytest
from prompt_toolkit.data_structures import Size

from axio_repl._prompt_terminal import PromptToolkitCompatibilityError, PromptToolkitInlineOutput


class _Output:
    responds_to_cpr = False

    def __init__(self, operations: list[str], *, columns: int = 10) -> None:
        self._operations = operations
        self._columns = columns

    def hide_cursor(self) -> None:
        self._operations.append("hide")

    def show_cursor(self) -> None:
        self._operations.append("show")

    def flush(self) -> None:
        self._operations.append("flush")

    def enable_autowrap(self) -> None:
        self._operations.append("autowrap")

    def write_raw(self, content: str) -> None:
        self._operations.append(f"raw:{content}")

    def cursor_up(self, amount: int) -> None:
        self._operations.append(f"up:{amount}")

    def erase_down(self) -> None:
        self._operations.append("erase-down")

    def get_size(self) -> Size:
        return Size(rows=12, columns=self._columns)


class _Renderer:
    def __init__(self, operations: list[str], *, fail_reset: bool = False) -> None:
        self._operations = operations
        self._fail_reset = fail_reset

    async def wait_for_cpr_responses(self) -> None:
        self._operations.append("cpr")

    def erase(self) -> None:
        self._operations.append("erase")

    def reset(self) -> None:
        self._operations.append("reset")
        if self._fail_reset:
            raise OSError("redraw failed")


class _App:
    def __init__(self, operations: list[str], *, fail_reset: bool = False, columns: int = 10) -> None:
        self.is_running = True
        self.output = _Output(operations, columns=columns)
        self.renderer = _Renderer(operations, fail_reset=fail_reset)
        self._running_in_terminal_f: asyncio.Future[None] | None = None
        self._running_in_terminal = False
        self._operations = operations

    def _request_absolute_cursor_position(self) -> None:
        self._operations.append("position")

    def _redraw(self) -> None:
        self._operations.append("redraw")


class _Session:
    def __init__(self, app: Any) -> None:
        self.app = app


async def test_inline_output_orders_the_complete_transaction() -> None:
    operations: list[str] = []
    app = _App(operations)
    adapter = PromptToolkitInlineOutput(_Session(app))

    await adapter.write(lambda: operations.append("write"))

    assert operations == ["hide", "erase", "write", "reset", "position", "redraw", "show", "flush"]
    assert app._running_in_terminal is False
    assert app._running_in_terminal_f is not None
    assert app._running_in_terminal_f.done()


def test_synchronous_redraw_uses_the_compatibility_boundary() -> None:
    operations: list[str] = []
    app = _App(operations)

    PromptToolkitInlineOutput.redraw_now(app)

    assert operations == ["redraw", "flush"]


async def test_inline_output_waits_for_cpr_before_erasing() -> None:
    operations: list[str] = []
    app = _App(operations)
    app.output.responds_to_cpr = True

    await PromptToolkitInlineOutput(_Session(app)).write(lambda: operations.append("write"))

    assert operations[:3] == ["cpr", "hide", "erase"]


async def test_live_output_replaces_wrapped_snapshot_and_finalizes_once() -> None:
    operations: list[str] = []
    app = _App(operations)
    adapter = PromptToolkitInlineOutput(_Session(app))

    await adapter.write_live("abcdefgh", key=("stdout", 1), is_live=True)
    operations.clear()
    await adapter.write_live("abcdefghijkl", key=("stdout", 1), is_live=True)

    assert operations.index("up:1") < operations.index("erase-down")
    assert "raw:abcdefghijkl" in operations
    assert operations.count("raw:\r\n") == 1

    operations.clear()
    await adapter.write_live("abcdefghijkl\n", key=("stdout", 1), is_live=False)

    assert operations.index("up:2") < operations.index("erase-down")
    assert "raw:abcdefghijkl\n" in operations
    assert "raw:\r\n" not in operations


async def test_live_output_counts_tab_stops_when_replacing_and_finalizing_wrapped_line() -> None:
    operations: list[str] = []
    app = _App(operations, columns=32)
    adapter = PromptToolkitInlineOutput(_Session(app))
    first = "> " + ("x" * 22) + "\tTAIL"

    await adapter.write_live(first, key=("stdout", 1), is_live=True)
    operations.clear()
    await adapter.write_live(first + "-MORE", key=("stdout", 1), is_live=True)

    assert operations.index("up:2") < operations.index("erase-down")
    assert f"raw:{first}-MORE" in operations

    operations.clear()
    await adapter.write_live(first + "-MORE\n", key=("stdout", 1), is_live=False)

    assert operations.index("up:2") < operations.index("erase-down")
    assert f"raw:{first}-MORE\n" in operations
    assert "raw:\r\n" not in operations


def test_live_row_measurement_keeps_ansi_wide_characters_and_tabs_consistent() -> None:
    assert PromptToolkitInlineOutput._display_rows("\033[2m> " + ("界" * 15) + "\033[0m", 32) == 1
    assert PromptToolkitInlineOutput._display_rows("\033[2m> " + ("界" * 16) + "\033[0m", 32) == 2
    assert PromptToolkitInlineOutput._display_rows("> " + ("x" * 22) + "\tTAIL", 32) == 2
    assert PromptToolkitInlineOutput._display_rows("\033[2m> " + ("界" * 11) + "\tTAIL\033[0m", 32) == 2


async def test_suppression_marker_discards_live_snapshot_instead_of_redrawing_it() -> None:
    operations: list[str] = []
    app = _App(operations)
    adapter = PromptToolkitInlineOutput(_Session(app))
    await adapter.write_live("partial", key=("stdout", 1), is_live=True)
    operations.clear()

    await adapter.write(lambda: operations.append("marker"), preserve_live=False)

    assert "up:1" in operations
    assert "erase-down" in operations
    assert "marker" in operations
    assert "raw:partial" not in operations


async def test_application_stopping_during_cpr_writes_without_redraw() -> None:
    operations: list[str] = []
    app = _App(operations)
    app.output.responds_to_cpr = True

    async def stop_during_cpr() -> None:
        operations.append("cpr")
        app.is_running = False

    setattr(app.renderer, "wait_for_cpr_responses", stop_during_cpr)

    await PromptToolkitInlineOutput(_Session(app)).write(lambda: operations.append("write"))

    assert operations == ["cpr", "write"]
    assert app._running_in_terminal_f is not None
    assert app._running_in_terminal_f.done()


async def test_cancelled_waiter_does_not_publish_or_release_a_transaction() -> None:
    operations: list[str] = []
    app = _App(operations)
    previous = asyncio.get_running_loop().create_future()
    app._running_in_terminal_f = previous
    adapter = PromptToolkitInlineOutput(_Session(app))

    task = asyncio.create_task(adapter.write(lambda: operations.append("write")))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert app._running_in_terminal_f is previous
    assert not previous.done()
    assert operations == []


async def test_concurrent_transactions_chain_after_the_current_future() -> None:
    operations: list[str] = []
    app = _App(operations)
    previous = asyncio.get_running_loop().create_future()
    app._running_in_terminal_f = previous
    adapter = PromptToolkitInlineOutput(_Session(app))

    first_task = asyncio.create_task(adapter.write(lambda: operations.append("first")))
    second_task = asyncio.create_task(adapter.write(lambda: operations.append("second")))
    await asyncio.sleep(0)
    assert operations == []
    previous.set_result(None)
    await asyncio.gather(first_task, second_task)

    first_group = ["hide", "erase", "first", "reset", "position", "redraw", "show", "flush"]
    second_group = ["hide", "erase", "second", "reset", "position", "redraw", "show", "flush"]
    assert operations in (first_group + second_group, second_group + first_group)


async def test_transaction_retries_when_the_claimed_predecessor_changes() -> None:
    operations: list[str] = []
    app = _App(operations)
    first = asyncio.get_running_loop().create_future()
    replacement = asyncio.get_running_loop().create_future()
    app._running_in_terminal_f = first
    adapter = PromptToolkitInlineOutput(_Session(app))

    task = asyncio.create_task(adapter.write(lambda: operations.append("write")))
    await asyncio.sleep(0)
    app._running_in_terminal_f = replacement
    first.set_result(None)
    await asyncio.sleep(0)
    assert operations == []

    replacement.set_result(None)
    await task
    assert "write" in operations


async def test_erase_failure_still_shows_cursor_and_releases_transaction() -> None:
    operations: list[str] = []
    app = _App(operations)

    def fail_erase() -> None:
        operations.append("erase")
        raise OSError("erase failed")

    setattr(app.renderer, "erase", fail_erase)

    with pytest.raises(OSError, match="erase failed"):
        await PromptToolkitInlineOutput(_Session(app)).write(lambda: operations.append("write"))

    assert operations == ["hide", "erase", "show", "flush"]
    assert app._running_in_terminal_f is not None
    assert app._running_in_terminal_f.done()


async def test_write_may_resolve_its_own_transaction_future() -> None:
    app = _App([])

    def write() -> None:
        assert app._running_in_terminal_f is not None
        app._running_in_terminal_f.set_result(None)

    await PromptToolkitInlineOutput(_Session(app)).write(write)

    assert app._running_in_terminal_f is not None
    assert app._running_in_terminal_f.done()


async def test_redraw_failure_releases_transaction_and_restores_running_flag() -> None:
    app = _App([], fail_reset=True)
    adapter = PromptToolkitInlineOutput(_Session(app))

    with pytest.raises(OSError, match="redraw failed"):
        await adapter.write(lambda: None)

    assert app._running_in_terminal is False
    assert app._running_in_terminal_f is not None
    assert app._running_in_terminal_f.done()


async def test_non_running_application_writes_without_private_api() -> None:
    app = type("StoppedApp", (), {"is_running": False})()
    writes: list[str] = []

    await PromptToolkitInlineOutput(_Session(app)).write(lambda: writes.append("write"))

    assert writes == ["write"]


async def test_incompatible_running_application_fails_before_writing() -> None:
    app = type("BrokenApp", (), {"is_running": True})()

    with pytest.raises(PromptToolkitCompatibilityError, match="_running_in_terminal_f"):
        await PromptToolkitInlineOutput(_Session(app)).write(lambda: pytest.fail("must not write"))
