"""Tool calls that keep running after the turn that started them.

A tool result belongs to the tool_use_id of the call that produced it, so it
cannot arrive ten turns later — the protocol expects the pair to close in the
same exchange. A detached call therefore closes immediately with a handle, and
the real output is collected through that handle afterwards.

The registry lives here because the agent loop starts the work, while whatever
collects it — a monitor tool, a REPL command — lives further out.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Literal

from . import notify

BACKGROUND_PARAM = "background"

BACKGROUND_PROPERTY: dict[str, Any] = {
    "type": "boolean",
    "default": False,
    "description": (
        "Run detached and return a handle instead of the result. Use it when the call is slow enough "
        "to be worth doing while you carry on: the result reaches you on its own once it lands, and "
        "monitor(tasks=[handle]) waits for it or returns its full output."
    ),
}

State = Literal["running", "done", "failed"]


@dataclass
class BackgroundCall:
    id: str
    name: str
    started_at: float
    task: asyncio.Task[str]
    owner: str | None = None
    collected: bool = False

    @property
    def state(self) -> State:
        if not self.task.done():
            return "running"
        return "failed" if self.task.exception() is not None else "done"

    def output(self) -> str:
        """Result, or the error if it failed. Empty while still running."""
        if not self.task.done():
            return ""
        exc = self.task.exception()
        if exc is not None:
            return f"{type(exc).__name__}: {exc}"
        return self.task.result()


_calls: dict[str, BackgroundCall] = {}
_waiters: list[asyncio.Future[BackgroundCall]] = []


def _notify(call: BackgroundCall) -> None:
    waiters, _waiters[:] = list(_waiters), []
    for waiter in waiters:
        if not waiter.done():
            waiter.set_result(call)


def _completion_message(call: BackgroundCall) -> str:
    # Handle, tool name, state and the way to get the rest come before the
    # output, so truncating a long result cannot eat them.
    return (
        f"[background task {call.id}] {call.name}: {call.state}\n"
        f'Full output if truncated: monitor(tasks=["{call.id}"])\n'
        f"{call.output()}"
    )


def _on_done(call: BackgroundCall, task: asyncio.Task[str]) -> None:
    _notify(call)
    # A cancelled task means shutdown, and a collected one was already read by
    # its owner; neither is news worth delivering.
    if task.cancelled() or call.collected:
        return
    notify.post(_completion_message(call), owner=call.owner, tag=call.id)


def start(name: str, coro: Coroutine[Any, Any, str]) -> str:
    """Run *coro* detached and return the handle used to collect it later."""
    handle = f"bg-{name}-{uuid.uuid4().hex[:6]}"
    task: asyncio.Task[str] = asyncio.ensure_future(coro)
    call = BackgroundCall(id=handle, name=name, started_at=time.time(), task=task, owner=notify.current_owner())
    _calls[handle] = call
    task.add_done_callback(lambda t: _on_done(call, t))
    return handle


def get(handle: str) -> BackgroundCall | None:
    return _calls.get(handle)


def snapshot() -> list[BackgroundCall]:
    return list(_calls.values())


def describe(handle: str) -> str:
    call = _calls.get(handle)
    if call is None:
        return f"{handle}: unknown handle"
    if call.state == "running":
        return f"{handle} ({call.name}): running for {time.time() - call.started_at:.0f}s"
    call.collected = True
    # Read here, so the automatic delivery of the same result is called off.
    notify.retract(call.owner, call.id)
    return f"{handle} ({call.name}): {call.state}\n{call.output()}"


async def next_completion() -> BackgroundCall:
    """Resolve when any detached call finishes.

    Only completions after the call are seen, so check :func:`snapshot` for work
    that finished while nobody was waiting.
    """
    waiter: asyncio.Future[BackgroundCall] = asyncio.get_running_loop().create_future()
    _waiters.append(waiter)
    return await waiter


async def cancel_all() -> None:
    for call in _calls.values():
        if not call.task.done():
            call.task.cancel()
    await asyncio.gather(*(c.task for c in _calls.values()), return_exceptions=True)
    _calls.clear()


def started_message(name: str, handle: str) -> str:
    return (
        f"{name} runs detached as {handle}. Its result will reach you on its own once it lands — "
        f'monitor(tasks=["{handle}"]) only if you need to wait for it or want the details.'
    )
