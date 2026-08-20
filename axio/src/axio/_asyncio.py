"""Cancellation-safe task helpers shared by runtime layers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any


def cancel_task_once(task: asyncio.Task[Any]) -> bool:
    """Request cancellation only for a live task without one already pending."""

    if task.done() or task.cancelling() > 0:
        return False
    task.cancel()
    return True


async def shield_until_done[T](awaitable: Awaitable[T]) -> tuple[T, asyncio.CancelledError | None]:
    """Finish an operation despite late cancellation and report the first late request."""

    task = asyncio.ensure_future(awaitable)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            return await asyncio.shield(task), cancellation
        except asyncio.CancelledError as exc:
            if task.done():
                return task.result(), cancellation or exc
            cancellation = cancellation or exc
