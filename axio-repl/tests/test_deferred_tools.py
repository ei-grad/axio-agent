from __future__ import annotations

import asyncio
from typing import Any

import pytest
from axio.agent import Agent, ToolDispatch
from axio.blocks import ToolResultBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_tool_use_response
from axio.tool import Tool
from axio_tools_agents.runtime import ExecutionMode, SessionEventHub, new_turn_identity, observe_agent_turn

from axio_repl._deferred_tools import (
    DeferredToolNotification,
    DeferredToolPhase,
    DeferredToolRegistry,
)


async def test_result_waits_for_protocol_placeholder_before_delivery() -> None:
    delivered: list[str] = []

    async def deliver(notification: object) -> None:
        delivered.append(notification.as_user_text())  # type: ignore[attr-defined]

    registry = DeferredToolRegistry(deliver)
    result_ready: asyncio.Future[list[ToolResultBlock]] = asyncio.get_running_loop().create_future()

    async def result() -> list[ToolResultBlock]:
        return await result_ready

    task = asyncio.create_task(result())
    dispatch = ToolDispatch((ToolUseBlock(id="call-1", name="shell", input={}),), task, "main")
    registry.dispatch_started(dispatch)
    registry.defer(dispatch)

    result_ready.set_result([ToolResultBlock(tool_use_id="call-1", content="done")])
    await asyncio.sleep(0)
    assert delivered == []
    assert registry.snapshots()[0].phase is DeferredToolPhase.DEFERRED

    registry.protocol_closed(dispatch)
    for _ in range(10):
        if delivered:
            break
        await asyncio.sleep(0)

    assert len(delivered) == 1
    assert "name=shell, call_id=call-1" in delivered[0]
    assert delivered[0].endswith("done")
    assert registry.snapshots() == ()


async def test_parallel_results_become_distinct_user_notifications() -> None:
    delivered: list[tuple[str, str]] = []

    async def deliver(notification: object) -> None:
        delivered.append((notification.tool_use_id, notification.text))  # type: ignore[attr-defined]

    registry = DeferredToolRegistry(deliver)

    async def result() -> list[ToolResultBlock]:
        return [
            ToolResultBlock(tool_use_id="one", content="first"),
            ToolResultBlock(tool_use_id="two", content="second"),
        ]

    task = asyncio.create_task(result())
    dispatch = ToolDispatch(
        (
            ToolUseBlock(id="one", name="shell", input={}),
            ToolUseBlock(id="two", name="python", input={}),
        ),
        task,
        "main",
    )
    registry.dispatch_started(dispatch)
    registry.defer(dispatch)
    registry.protocol_closed(dispatch)
    await task
    for _ in range(10):
        if len(delivered) == 2:
            break
        await asyncio.sleep(0)

    assert delivered == [("one", "first"), ("two", "second")]


async def test_close_cancels_session_owned_dispatch_and_reports_snapshot() -> None:
    async def deliver(_notification: object) -> None:
        raise AssertionError("shutdown must not deliver an unfinished result")

    registry = DeferredToolRegistry(deliver)

    async def never() -> list[ToolResultBlock]:
        await asyncio.Future()
        return []

    task = asyncio.create_task(never())
    dispatch = ToolDispatch((ToolUseBlock(id="call", name="shell", input={}),), task, "main")
    registry.dispatch_started(dispatch)
    registry.defer(dispatch)
    registry.protocol_closed(dispatch)

    snapshots = await registry.close()

    assert snapshots[0].tool_use_ids == ("call",)
    assert snapshots[0].phase is DeferredToolPhase.PROTOCOL_CLOSED
    assert task.cancelled()
    assert registry.snapshots() == ()


async def test_background_turn_deferral_retains_owner_and_closes_protocol_once() -> None:
    release = asyncio.Event()
    started = asyncio.Event()
    delivered: list[DeferredToolNotification] = []
    dispatch_owner: list[tuple[str, str | None]] = []

    async def slow_tool() -> str:
        await release.wait()
        return "actual child result"

    async def deliver(notification: DeferredToolNotification) -> None:
        delivered.append(notification)

    def dispatch_started(agent_id: str, turn_id: str | None) -> None:
        dispatch_owner.append((agent_id, turn_id))
        started.set()

    registry = DeferredToolRegistry(deliver, on_dispatch_started=dispatch_started)
    context = MemoryContextStore()
    identity = new_turn_identity(
        agent_id="child-1",
        parent_agent_id="main",
        execution_mode=ExecutionMode.BACKGROUND,
        run_id="child-run",
    )
    agent = Agent(
        system="child",
        transport=StubTransport([make_tool_use_response("slow", "call-1", {})]),
        tools=[Tool[Any](name="slow", handler=slow_tool)],
        deferred_tool_sink=registry,
    )
    turn = asyncio.create_task(
        observe_agent_turn(
            agent=agent,
            context=context,
            prompt="run",
            identity=identity,
            hub=SessionEventHub(),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    assert dispatch_owner == [("child-1", identity.turn_id)]
    snapshot = registry.snapshots()[0]
    assert snapshot.agent_id == "child-1"
    assert snapshot.turn_id == identity.turn_id
    assert registry.request_preemption(identity.turn_id)
    assert not registry.request_preemption(identity.turn_id)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    history = await context.get_history()
    protocol_results = [
        block
        for message in history
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.tool_use_id == "call-1"
    ]
    assert len(protocol_results) == 1
    assert "continues after interruption" in str(protocol_results[0].content)

    release.set()
    for _ in range(10):
        if delivered:
            break
        await asyncio.sleep(0)

    assert len(delivered) == 1
    assert delivered[0].agent_id == "child-1"
    assert delivered[0].run_id == "child-run"
    assert delivered[0].tool_use_id == "call-1"
    assert delivered[0].text == "actual child result"
    assert registry.snapshots() == ()
