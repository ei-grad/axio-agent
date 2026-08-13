from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from axio.agent import Agent
from axio.blocks import TextBlock
from axio.context import MemoryContextStore
from axio.events import (
    IterationEnd,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.messages import Message
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.tool import Tool
from axio.types import StopReason, Usage

from axio_tools_agents.runtime import (
    AgentEventEnvelope,
    ContextForked,
    ExecutionMode,
    MessageCommitted,
    ObservedContextStore,
    SessionEventHub,
    TurnFinished,
    TurnStarted,
    TurnStatus,
    current_turn_identity,
    new_turn_identity,
    observe_agent_turn,
)


async def test_event_hub_fans_out_one_total_order_to_every_subscriber() -> None:
    hub = SessionEventHub(session_id="session-1")
    first: list[int] = []
    second: list[int] = []

    async def collect_first(envelope: AgentEventEnvelope) -> None:
        first.append(envelope.seq)

    async def collect_second(envelope: AgentEventEnvelope) -> None:
        second.append(envelope.seq)

    hub.subscribe(collect_first)
    hub.subscribe(collect_second)
    await asyncio.gather(
        *(
            hub.publish(
                TextDelta(index=0, delta=str(index)),
                run_id="run",
                agent_id="agent",
                parent_agent_id=None,
                turn_id=str(index),
                execution_mode=ExecutionMode.BACKGROUND,
            )
            for index in range(20)
        )
    )

    assert first == list(range(1, 21))
    assert second == first


async def test_event_hub_isolates_a_failed_observer_from_other_observers() -> None:
    hub = SessionEventHub()
    observed: list[str] = []

    async def fail(_envelope: AgentEventEnvelope) -> None:
        raise RuntimeError("observer failed")

    async def collect(envelope: AgentEventEnvelope) -> None:
        assert isinstance(envelope.event, TextDelta)
        observed.append(envelope.event.delta)

    hub.subscribe(fail)
    hub.subscribe(collect)

    await hub.publish(
        TextDelta(index=0, delta="still delivered"),
        run_id="run",
        agent_id="agent",
        parent_agent_id=None,
        turn_id="turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )

    assert observed == ["still delivered"]


async def test_observed_turn_forwards_full_stream_and_finishes_with_typed_outcome() -> None:
    hub = SessionEventHub(session_id="session-1")
    events: list[object] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        events.append(envelope.event)

    hub.subscribe(collect)
    agent = Agent(system="child", transport=StubTransport([make_text_response("answer")]))
    identity = new_turn_identity(
        agent_id="child",
        parent_agent_id="parent",
        execution_mode=ExecutionMode.FOREGROUND,
        parent_tool_use_id="call-1",
        run_id="run-1",
    )

    outcome = await observe_agent_turn(
        agent=agent,
        context=MemoryContextStore(),
        prompt="work",
        identity=identity,
        hub=hub,
    )

    assert outcome.status is TurnStatus.SUCCEEDED
    assert outcome.text == "answer"
    assert outcome.identity == identity
    assert isinstance(events[0], TurnStarted)
    assert any(isinstance(event, TextDelta) and event.delta == "answer" for event in events)
    assert any(isinstance(event, SessionEndEvent) for event in events)
    assert events[-1] == TurnFinished(
        status=TurnStatus.SUCCEEDED,
        stop_reason=StopReason.end_turn,
        error=None,
    )


async def test_observed_context_publishes_commits_and_forks_with_context_ids() -> None:
    hub = SessionEventHub(session_id="session-1")
    envelopes: list[AgentEventEnvelope] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        envelopes.append(envelope)

    hub.subscribe(collect)
    context = ObservedContextStore(MemoryContextStore(), hub)
    identity = new_turn_identity(
        agent_id="main",
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        run_id="run-1",
        context_id=context.session_id,
    )

    await observe_agent_turn(
        agent=Agent(system="main", transport=StubTransport([make_text_response("answer")])),
        context=context,
        prompt="question",
        identity=identity,
        hub=hub,
    )
    child = await context.fork()

    committed = [envelope for envelope in envelopes if isinstance(envelope.event, MessageCommitted)]
    committed_events = [envelope.event for envelope in committed]
    assert all(isinstance(event, MessageCommitted) for event in committed_events)
    assert [event.message.role for event in committed_events if isinstance(event, MessageCommitted)] == [
        "user",
        "assistant",
    ]
    assert all(envelope.context_id == context.session_id for envelope in committed)
    forked = next(envelope for envelope in envelopes if isinstance(envelope.event, ContextForked))
    assert isinstance(forked.event, ContextForked)
    assert forked.context_id == context.session_id
    assert forked.event.source_context_id == context.session_id
    assert forked.event.child_context_id == child.session_id


async def test_failed_context_append_is_not_reported_as_committed() -> None:
    class FailingContext(MemoryContextStore):
        async def append(self, message: Message) -> None:
            del message
            raise OSError("storage unavailable")

    hub = SessionEventHub()
    events: list[object] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        events.append(envelope.event)

    hub.subscribe(collect)
    context = ObservedContextStore(FailingContext(), hub)
    context.bind_identity(
        new_turn_identity(
            agent_id="main",
            parent_agent_id=None,
            execution_mode=ExecutionMode.FOREGROUND,
            context_id=context.session_id,
        )
    )

    with pytest.raises(OSError, match="storage unavailable"):
        await context.append(Message(role="user", content=[TextBlock(text="not stored")]))

    assert not any(isinstance(event, MessageCommitted) for event in events)


async def test_observed_turn_forwards_child_tool_arguments_output_and_result_in_order() -> None:
    async def stream_tool(value: str) -> str:
        return f"final:{value}"

    async def stream(value: str) -> AsyncIterator[tuple[str, str]]:
        yield "stdout", "first\n"
        yield "stdout", "second\n"

    stream_tool.stream = stream  # type: ignore[attr-defined]
    transport = StubTransport(
        [
            make_tool_use_response("stream_tool", tool_id="child-call", tool_input={"value": "x"}),
            make_text_response("child done"),
        ]
    )
    agent = Agent(system="child", transport=transport, tools=[Tool(name="stream_tool", handler=stream_tool)])
    hub = SessionEventHub()
    events: list[object] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        events.append(envelope.event)

    hub.subscribe(collect)
    identity = new_turn_identity(
        agent_id="child",
        parent_agent_id="parent",
        execution_mode=ExecutionMode.FOREGROUND,
        parent_tool_use_id="parent-call",
    )

    outcome = await observe_agent_turn(
        agent=agent,
        context=MemoryContextStore(),
        prompt="work",
        identity=identity,
        hub=hub,
    )

    relevant = [
        type(event)
        for event in events
        if isinstance(event, ToolUseStart | ToolInputDelta | ToolOutputDelta | ToolResult | TextDelta)
    ]
    assert relevant == [
        ToolUseStart,
        ToolInputDelta,
        ToolOutputDelta,
        ToolOutputDelta,
        ToolResult,
        TextDelta,
    ]
    assert outcome.text == "child done"


class _LateFailureTransport:
    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta(index=0, delta="not success yet")
        yield IterationEnd(iteration=1, stop_reason=StopReason.end_turn, usage=Usage(0, 0))
        raise KeyboardInterrupt("late failure")


async def test_outcome_is_not_finalized_from_terminal_event_before_iterator_finishes() -> None:
    hub = SessionEventHub()
    identity = new_turn_identity(
        agent_id="child",
        parent_agent_id="parent",
        execution_mode=ExecutionMode.FOREGROUND,
    )

    outcome = await observe_agent_turn(
        agent=Agent(system="child", transport=_LateFailureTransport()),
        context=MemoryContextStore(),
        prompt="work",
        identity=identity,
        hub=hub,
    )

    assert outcome.status is TurnStatus.FAILED
    assert outcome.stop_reason is StopReason.error
    assert outcome.error == "KeyboardInterrupt: late failure"


class _BlockingTransport:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        try:
            self.started.set()
            await asyncio.Event().wait()
            yield TextDelta(index=0, delta="unreachable")
        finally:
            self.closed.set()


async def test_cancel_closes_child_stream_and_records_cancelled_turn() -> None:
    transport = _BlockingTransport()
    hub = SessionEventHub()
    finished: list[TurnFinished] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        event = envelope.event
        if isinstance(event, TurnFinished):
            finished.append(event)

    hub.subscribe(collect)
    identity = new_turn_identity(
        agent_id="child",
        parent_agent_id="parent",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    task = asyncio.create_task(
        observe_agent_turn(
            agent=Agent(system="child", transport=transport),
            context=MemoryContextStore(),
            prompt="work",
            identity=identity,
            hub=hub,
        )
    )
    await transport.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert transport.closed.is_set()
    assert finished[-1].status is TurnStatus.CANCELLED
    assert current_turn_identity() is None
