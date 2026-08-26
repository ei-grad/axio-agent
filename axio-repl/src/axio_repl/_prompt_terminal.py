"""Isolated prompt-toolkit transaction for writing above an inline prompt."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.formatted_text import ANSI, fragment_list_to_text, to_formatted_text
from prompt_toolkit.utils import get_cwidth


@dataclass(frozen=True, slots=True)
class _LiveOutput:
    key: tuple[str, int]
    content: str
    rows: int
    priority: int
    order: int


class PromptToolkitCompatibilityError(RuntimeError):
    """The installed prompt-toolkit lacks private APIs required by the adapter."""


class PromptToolkitInlineOutput:
    """Serialize one erase/write/redraw transaction without detaching input."""

    def __init__(self, prompt_session: Any) -> None:
        self._session = prompt_session
        self._live: OrderedDict[tuple[str, int], _LiveOutput] = OrderedDict()
        self._next_live_order = 1

    @staticmethod
    def redraw_now(app: Any) -> None:
        """Synchronously render changed temporary UI state before prompt exit."""

        if not app.is_running:
            return
        PromptToolkitInlineOutput._validate(app)
        app._redraw()
        app.output.flush()

    async def write(self, write: Callable[[], None], *, preserve_live: bool = True) -> None:
        app = self._session.app
        if not app.is_running:
            self._write_without_prompt(app, write, preserve_live=preserve_live)
            return
        self._validate(app)

        await self._write_running(app, write, preserve_live=preserve_live)

    async def write_live(
        self,
        content: str,
        *,
        key: tuple[str, int],
        is_live: bool,
        priority: int = 0,
    ) -> None:
        """Replace or finalize one live logical line above the active prompt."""

        app = self._session.app
        if not app.is_running:
            self._write_live_without_prompt(app.output, content, key=key, is_live=is_live, priority=priority)
            return
        self._validate(app)
        self._validate_live_output(app.output)

        def write_content() -> None:
            app.output.enable_autowrap()
            app.output.write_raw(content)

        replacement = (key, content, is_live, priority)
        await self._write_running(app, write_content, replacement=replacement)

    async def discard_live(self, key: tuple[str, int]) -> None:
        """Remove one replaceable line without committing it to scrollback."""

        if key not in self._live:
            return
        app = self._session.app
        if not app.is_running:
            self._write_live_without_prompt(app.output, "", key=key, is_live=False, priority=0)
            return
        self._validate(app)
        self._validate_live_output(app.output)
        await self._write_running(app, lambda: None, replacement=(key, "", False, 0))

    async def _write_running(
        self,
        app: Any,
        write: Callable[[], None],
        *,
        replacement: tuple[tuple[str, int], str, bool, int] | None = None,
        preserve_live: bool = True,
    ) -> None:
        completed = asyncio.get_running_loop().create_future()
        await self._claim_transaction(app, completed)
        rendering_suspended = False
        cursor_hidden = False
        try:
            if app.output.responds_to_cpr:
                await app.renderer.wait_for_cpr_responses()
            if not app.is_running:
                if replacement is None:
                    self._write_without_prompt(app, write, preserve_live=preserve_live)
                else:
                    key, content, is_live, priority = replacement
                    self._write_live_without_prompt(
                        app.output,
                        content,
                        key=key,
                        is_live=is_live,
                        priority=priority,
                    )
                return
            app.output.hide_cursor()
            cursor_hidden = True
            app.renderer.erase()
            app._running_in_terminal = True
            rendering_suspended = True
            self._erase_live(app.output, tuple(self._live.values()))
            if replacement is None:
                write()
                if not preserve_live:
                    self._live.clear()
            else:
                key, content, is_live, priority = replacement
                if is_live:
                    previous = self._live.get(key)
                    order = previous.order if previous is not None else self._next_live_order
                    if previous is None:
                        self._next_live_order += 1
                    self._live[key] = _LiveOutput(key, content, 0, priority, order)
                else:
                    self._live.pop(key, None)
                    write()
            self._draw_live_outputs(app.output)
        finally:
            try:
                if rendering_suspended:
                    app._running_in_terminal = False
                    app.renderer.reset()
                    app._request_absolute_cursor_position()
                    app._redraw()
            finally:
                if cursor_hidden:
                    app.output.show_cursor()
                    app.output.flush()
                if not completed.done():
                    completed.set_result(None)

    def _write_without_prompt(
        self,
        app: Any,
        write: Callable[[], None],
        *,
        preserve_live: bool,
    ) -> None:
        if not self._live:
            write()
            return
        self._validate_live_output(app.output)
        self._erase_live(app.output, tuple(self._live.values()))
        write()
        if preserve_live:
            self._draw_live_outputs(app.output)
        else:
            self._live.clear()
        app.output.flush()

    def _write_live_without_prompt(
        self,
        output: Any,
        content: str,
        *,
        key: tuple[str, int],
        is_live: bool,
        priority: int,
    ) -> None:
        previous = self._live.get(key)
        self._erase_live(output, tuple(self._live.values()))
        if is_live:
            order = previous.order if previous is not None else self._next_live_order
            if previous is None:
                self._next_live_order += 1
            self._live[key] = _LiveOutput(key, content, 0, priority, order)
        else:
            self._live.pop(key, None)
            output.enable_autowrap()
            output.write_raw(content)
        self._draw_live_outputs(output)
        output.flush()

    def _draw_live_outputs(self, output: Any) -> None:
        ordered = sorted(self._live.values(), key=lambda item: (item.priority, item.order))
        for live in ordered:
            rows = self._draw_live(output, live.content)
            self._live[live.key] = _LiveOutput(
                live.key,
                live.content,
                rows,
                live.priority,
                live.order,
            )

    @staticmethod
    def _erase_live(output: Any, live_outputs: tuple[_LiveOutput, ...]) -> None:
        rows = sum(live.rows for live in live_outputs)
        if rows == 0:
            return
        output.cursor_up(rows)
        output.write_raw("\r")
        output.erase_down()

    @classmethod
    def _draw_live(cls, output: Any, content: str) -> int:
        output.enable_autowrap()
        output.write_raw(content)
        return cls._draw_live_separator(output, content)

    @classmethod
    def _draw_live_separator(cls, output: Any, content: str) -> int:
        output.write_raw("\r\n")
        columns = max(1, int(output.get_size().columns))
        return cls._display_rows(content, columns)

    @staticmethod
    def _display_rows(content: str, columns: int) -> int:
        plain = fragment_list_to_text(to_formatted_text(ANSI(content)))
        rows = 1
        column = 0
        for character in plain:
            if character == "\t":
                column = min(columns, ((column // 8) + 1) * 8)
                continue
            if character == "\r":
                column = 0
                continue
            if character == "\n":
                rows += 1
                column = 0
                continue
            width = get_cwidth(character)
            if width <= 0:
                continue
            width = min(width, columns)
            if column >= columns or column + width > columns:
                rows += 1
                column = 0
            column += width
        return rows

    @staticmethod
    async def _claim_transaction(app: Any, completed: asyncio.Future[None]) -> None:
        while True:
            previous = app._running_in_terminal_f
            if previous is not None:
                await asyncio.shield(previous)
            if app._running_in_terminal_f is previous:
                app._running_in_terminal_f = completed
                return

    @staticmethod
    def _validate(app: Any) -> None:
        required_app = (
            "_running_in_terminal_f",
            "_running_in_terminal",
            "_request_absolute_cursor_position",
            "_redraw",
            "renderer",
            "output",
        )
        missing = [name for name in required_app if not hasattr(app, name)]
        renderer = getattr(app, "renderer", None)
        output = getattr(app, "output", None)
        if renderer is not None:
            missing.extend(
                f"renderer.{name}"
                for name in ("erase", "reset", "wait_for_cpr_responses")
                if not hasattr(renderer, name)
            )
        if output is not None:
            missing.extend(
                f"output.{name}"
                for name in ("responds_to_cpr", "hide_cursor", "show_cursor", "flush")
                if not hasattr(output, name)
            )
        if missing:
            raise PromptToolkitCompatibilityError(
                "installed prompt-toolkit is incompatible with inline output adapter; missing " + ", ".join(missing)
            )

    @staticmethod
    def _validate_live_output(output: Any) -> None:
        required = ("cursor_up", "enable_autowrap", "erase_down", "get_size", "write_raw")
        missing = [f"output.{name}" for name in required if not hasattr(output, name)]
        if missing:
            raise PromptToolkitCompatibilityError(
                "installed prompt-toolkit is incompatible with live output adapter; missing " + ", ".join(missing)
            )
