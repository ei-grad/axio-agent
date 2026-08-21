"""Isolated prompt-toolkit transaction for writing above an inline prompt."""

from __future__ import annotations

import asyncio
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


class PromptToolkitCompatibilityError(RuntimeError):
    """The installed prompt-toolkit lacks private APIs required by the adapter."""


class PromptToolkitInlineOutput:
    """Serialize one erase/write/redraw transaction without detaching input."""

    def __init__(self, prompt_session: Any) -> None:
        self._session = prompt_session
        self._live: _LiveOutput | None = None

    async def write(self, write: Callable[[], None], *, preserve_live: bool = True) -> None:
        app = self._session.app
        if not app.is_running:
            write()
            if not preserve_live:
                self._live = None
            return
        self._validate(app)

        await self._write_running(app, write, preserve_live=preserve_live)

    async def write_live(self, content: str, *, key: tuple[str, int], is_live: bool) -> None:
        """Replace or finalize one live logical line above the active prompt."""

        app = self._session.app
        if not app.is_running:
            self._write_live_without_prompt(app.output, content, key=key, is_live=is_live)
            return
        self._validate(app)
        self._validate_live_output(app.output)

        def write_content() -> None:
            app.output.enable_autowrap()
            app.output.write_raw(content)

        replacement = (key, content, is_live)
        await self._write_running(app, write_content, replacement=replacement)

    async def _write_running(
        self,
        app: Any,
        write: Callable[[], None],
        *,
        replacement: tuple[tuple[str, int], str, bool] | None = None,
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
                    write()
                    if not preserve_live:
                        self._live = None
                else:
                    key, content, is_live = replacement
                    self._write_live_without_prompt(app.output, content, key=key, is_live=is_live)
                return
            app.output.hide_cursor()
            cursor_hidden = True
            app.renderer.erase()
            app._running_in_terminal = True
            rendering_suspended = True
            previous = self._live
            if previous is not None:
                self._erase_live(app.output, previous)
            if replacement is None:
                write()
                if previous is not None and preserve_live:
                    rows = self._draw_live(app.output, previous.content)
                    self._live = _LiveOutput(previous.key, previous.content, rows)
                elif previous is not None:
                    self._live = None
            else:
                key, content, is_live = replacement
                write()
                if is_live:
                    rows = self._draw_live_separator(app.output, content)
                    self._live = _LiveOutput(key, content, rows)
                elif previous is not None and previous.key != key:
                    rows = self._draw_live(app.output, previous.content)
                    self._live = _LiveOutput(previous.key, previous.content, rows)
                else:
                    self._live = None
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

    def _write_live_without_prompt(
        self,
        output: Any,
        content: str,
        *,
        key: tuple[str, int],
        is_live: bool,
    ) -> None:
        previous = self._live
        visible = content
        if previous is not None and previous.key == key and content.startswith(previous.content):
            visible = content[len(previous.content) :]
        output.enable_autowrap()
        output.write_raw(visible)
        output.flush()
        self._live = _LiveOutput(key, content, 0) if is_live else None

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
    def _erase_live(output: Any, live: _LiveOutput) -> None:
        output.cursor_up(live.rows)
        output.write_raw("\r")
        output.erase_down()

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
