from __future__ import annotations

import asyncio
from typing import Any

import pytest
from axio.agent import Agent, ToolDispatch
from axio.blocks import ToolResultBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.events import ToolUseStart
from axio.exceptions import HandlerError
from axio.testing import StubTransport, make_tool_use_response
from axio.tool import Tool
from axio_tools_agents.runtime import (
    AgentEventEnvelope,
    ExecutionMode,
    MessageCommitted,
    ObservedContextStore,
    SessionEventHub,
    new_turn_identity,
    observe_agent_turn,
)

from axio_repl import ReplRenderer, render_runtime_event
from axio_repl._deferred_tools import (
    DeferredToolNotification,
    DeferredToolPhase,
    DeferredToolRegistry,
)
from axio_repl._multiplexer import sanitize_terminal_text
from axio_repl._theme import DEFAULT_THEME, NO_COLOR_THEME


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


async def test_just_started_dispatch_can_be_cancelled_without_running_handler() -> None:
    handler_started = asyncio.Event()

    async def deliver(_notification: object) -> None:
        raise AssertionError("cancelled dispatch must not be delivered as deferred")

    async def result() -> list[ToolResultBlock]:
        handler_started.set()
        await asyncio.Future()
        return []

    registry = DeferredToolRegistry(deliver)
    task = asyncio.create_task(result())
    dispatch = ToolDispatch((ToolUseBlock(id="call", name="shell", input={}),), task, "main")
    registry.dispatch_started(dispatch)

    assert registry.cancel_before_run(dispatch)
    await asyncio.gather(task, return_exceptions=True)
    registry.dispatch_finished(dispatch)

    assert task.cancelled()
    assert not handler_started.is_set()
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

    def dispatch_started(_dispatch: ToolDispatch, agent_id: str, turn_id: str | None) -> None:
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


async def test_repeat_cancel_cannot_tear_deferred_tool_protocol_finalization() -> None:
    tool_started = asyncio.Event()
    tool_release = asyncio.Event()
    assistant_commit_started = asyncio.Event()
    assistant_commit_release = asyncio.Event()
    committed: list[MessageCommitted] = []
    delivered: list[DeferredToolNotification] = []

    async def slow_tool() -> str:
        tool_started.set()
        await tool_release.wait()
        return "actual result"

    async def deliver(notification: DeferredToolNotification) -> None:
        delivered.append(notification)

    async def observe_commit(envelope: AgentEventEnvelope) -> None:
        event = envelope.event
        if not isinstance(event, MessageCommitted):
            return
        committed.append(event)
        if event.message.role == "assistant" and any(
            isinstance(block, ToolUseBlock) for block in event.message.content
        ):
            assistant_commit_started.set()
            await assistant_commit_release.wait()

    hub = SessionEventHub()
    hub.subscribe(observe_commit)
    registry = DeferredToolRegistry(deliver)
    context = ObservedContextStore(MemoryContextStore(), hub)
    identity = new_turn_identity(
        agent_id="main",
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        context_id=context.session_id,
    )
    agent = Agent(
        system="main",
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
            hub=hub,
        )
    )
    await asyncio.wait_for(tool_started.wait(), timeout=1)

    assert registry.request_preemption(identity.turn_id)
    turn.cancel("initial cancellation")
    await asyncio.wait_for(assistant_commit_started.wait(), timeout=1)
    turn.cancel("late cancellation")
    assistant_commit_release.set()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(turn, timeout=1)
    assert cancelled.value.args == ("initial cancellation",)

    tool_result_commits = [
        event
        for event in committed
        if event.message.role == "user"
        and any(
            isinstance(block, ToolResultBlock) and block.tool_use_id == "call-1" for block in event.message.content
        )
    ]
    assert len(tool_result_commits) == 1
    assert registry.snapshots()[0].phase is DeferredToolPhase.PROTOCOL_CLOSED

    tool_release.set()
    for _ in range(10):
        if delivered:
            break
        await asyncio.sleep(0)

    assert len(delivered) == 1
    assert delivered[0].tool_use_id == "call-1"
    assert delivered[0].text == "actual result"
    assert registry.snapshots() == ()


@pytest.mark.parametrize(("is_error", "glyph"), [(False, "✓"), (True, "✗")])
async def test_deferred_result_keeps_original_badge_and_renders_once(
    capsys: pytest.CaptureFixture[str],
    is_error: bool,
    glyph: str,
) -> None:
    tool_started = asyncio.Event()
    release = asyncio.Event()
    dispatch_started = asyncio.Event()
    delivered: list[DeferredToolNotification] = []
    renderer = ReplRenderer(theme=NO_COLOR_THEME)

    async def slow_tool() -> str:
        tool_started.set()
        await release.wait()
        if is_error:
            raise HandlerError("deferred failed")
        return "deferred succeeded"

    async def deliver(notification: DeferredToolNotification) -> None:
        delivered.append(notification)
        assert await renderer.render_deferred_tool_result(notification)
        await renderer.incoming(notification.as_user_text(), suppress_display=True)

    def on_dispatch_started(_dispatch: ToolDispatch, _agent_id: str, _turn_id: str | None) -> None:
        dispatch_started.set()

    registry = DeferredToolRegistry(
        deliver,
        on_dispatch_started=on_dispatch_started,
        on_dispatch_deferred=renderer.defer_tool_calls,
    )
    hub = SessionEventHub()

    async def render(envelope: AgentEventEnvelope) -> None:
        await render_runtime_event(renderer, envelope)

    hub.subscribe(render)
    context = MemoryContextStore()
    identity = new_turn_identity(
        agent_id="main",
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        run_id="main-run",
        context_id=context.session_id,
    )
    agent = Agent(
        system="main",
        transport=StubTransport([make_tool_use_response("slow", "provider-call-1", {})]),
        tools=[Tool[Any](name="slow", handler=slow_tool)],
        deferred_tool_sink=registry,
    )
    turn = asyncio.create_task(
        observe_agent_turn(
            agent=agent,
            context=context,
            prompt="run",
            identity=identity,
            hub=hub,
        )
    )
    await asyncio.wait_for(tool_started.wait(), timeout=1)
    await asyncio.wait_for(dispatch_started.wait(), timeout=1)
    assert "▶ slow #001" in sanitize_terminal_text(capsys.readouterr().out)

    assert registry.request_preemption(identity.turn_id)
    turn.cancel("defer")
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert renderer.active_tool_call_count == 1

    release.set()
    for _ in range(20):
        if delivered:
            break
        await asyncio.sleep(0)

    assert len(delivered) == 1
    output = sanitize_terminal_text(capsys.readouterr().out)
    expected_body = "deferred failed" if is_error else "deferred succeeded"
    assert output.count(f"{glyph} slow #001") == 1
    assert output.count(expected_body) == 1
    assert "provider-call-1" not in output
    assert "provider-call-1" in delivered[0].as_user_text()
    assert renderer.active_tool_call_count == 0

    history = await context.get_history()
    placeholders = [
        block
        for message in history
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.tool_use_id == "provider-call-1"
    ]
    assert len(placeholders) == 1
    assert "continues after interruption" in str(placeholders[0].content)

    assert not await renderer.render_deferred_tool_result(delivered[0])
    assert capsys.readouterr().out == ""


async def test_deferred_patch_result_uses_semantic_diff_styles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(theme=DEFAULT_THEME)
    content = "+1 -1\n@@ -42,2 +42,2 @@ render\n unchanged\n-old\n+new"
    await renderer.render(
        "main",
        ToolUseStart(index=0, tool_use_id="patch-call", name="patch_file"),
        run_id="main-run",
        turn_id="main-turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    renderer.defer_tool_calls(
        "main",
        "main-run",
        "main-turn",
        (ToolUseBlock(id="patch-call", name="patch_file", input={}),),
    )
    capsys.readouterr()
    notification = DeferredToolNotification(
        agent_id="main",
        run_id="main-run",
        turn_id="main-turn",
        tool_use_id="patch-call",
        tool_name="patch_file",
        text=content,
        is_error=False,
    )

    assert await renderer.render_deferred_tool_result(notification)

    output = capsys.readouterr().out
    assert f"{DEFAULT_THEME.stdout.ansi}+1 -1{DEFAULT_THEME.reset}" in output
    assert f"{DEFAULT_THEME.tool.ansi}@@ -42,2 +42,2 @@ render{DEFAULT_THEME.reset}" in output
    assert f"{DEFAULT_THEME.reasoning.ansi} unchanged{DEFAULT_THEME.reset}" in output
    assert f"{DEFAULT_THEME.error.ansi}-old{DEFAULT_THEME.reset}" in output
    assert f"{DEFAULT_THEME.success.ansi}+new{DEFAULT_THEME.reset}" in output
    assert sanitize_terminal_text(output).count("✓ patch_file #001") == 1
    assert notification.text == content


async def test_shutdown_releases_deferred_badge_ownership() -> None:
    renderer = ReplRenderer(theme=NO_COLOR_THEME)
    await renderer.render(
        "main",
        ToolUseStart(index=0, tool_use_id="call", name="shell"),
        run_id="run",
        turn_id="turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    renderer.defer_tool_calls(
        "main",
        "run",
        "turn",
        (ToolUseBlock(id="call", name="shell", input={}),),
    )
    assert renderer.active_tool_call_count == 1

    await renderer.discard_deferred_tool_calls()

    assert renderer.active_tool_call_count == 0
