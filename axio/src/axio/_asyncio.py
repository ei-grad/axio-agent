"""Cancellation-safe task helpers shared by runtime layers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CancellationCause:
    """Host-provided reason for cancellation that crosses task boundaries."""

    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("cancellation reason must not be empty")


def cancel_task_once(task: asyncio.Task[Any], *, message: object | None = None) -> bool:
    """Request cancellation only for a live task without one already pending."""

    if task.done() or task.cancelling() > 0:
        return False
    if message is None:
        task.cancel()
    else:
        task.cancel(message)
    return True


async def cancel_tasks_bounded(
    tasks: tuple[asyncio.Task[Any], ...],
    *,
    grace_seconds: float = 1.0,
    message: object | None = None,
) -> tuple[asyncio.Task[Any], ...]:
    """Cancel tasks, repeat cancellation after a grace period, and return stragglers."""

    if grace_seconds <= 0:
        raise ValueError("grace_seconds must be positive")
    pending = {task for task in tasks if not task.done()}
    for attempt in range(3):
        for task in pending:
            if attempt == 0:
                cancel_task_once(task, message=message)
            elif message is None:
                task.cancel()
            else:
                task.cancel(message)
        if pending:
            _, pending = await asyncio.wait(pending, timeout=grace_seconds)
    for task in tasks:
        if task not in pending:
            with suppress(asyncio.CancelledError):
                task.exception()
    return tuple(pending)


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
