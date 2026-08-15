"""Isolated prompt-toolkit transaction for writing above an inline prompt."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


class PromptToolkitCompatibilityError(RuntimeError):
    """The installed prompt-toolkit lacks private APIs required by the adapter."""


class PromptToolkitInlineOutput:
    """Serialize one erase/write/redraw transaction without detaching input."""

    def __init__(self, prompt_session: Any) -> None:
        self._session = prompt_session

    async def write(self, write: Callable[[], None]) -> None:
        app = self._session.app
        if not app.is_running:
            write()
            return
        self._validate(app)

        completed = asyncio.get_running_loop().create_future()
        await self._claim_transaction(app, completed)
        rendering_suspended = False
        cursor_hidden = False
        try:
            if app.output.responds_to_cpr:
                await app.renderer.wait_for_cpr_responses()
            if not app.is_running:
                write()
                return
            app.output.hide_cursor()
            cursor_hidden = True
            app.renderer.erase()
            app._running_in_terminal = True
            rendering_suspended = True
            write()
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
