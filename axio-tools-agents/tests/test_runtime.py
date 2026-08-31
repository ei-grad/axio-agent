from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from axio.agent import Agent
from axio.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.events import (
    IterationEnd,
    Refusal,
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
    observe_agent_turn_messages,
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


async def test_event_hub_holds_later_publication_until_reserved_ingress_arrives() -> None:
    hub = SessionEventHub(session_id="session-1")
    observed: list[tuple[int, str]] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        assert isinstance(envelope.event, TextDelta)
        observed.append((envelope.seq, envelope.event.delta))

    hub.subscribe(collect)
    accepted_enter_seq = hub.reserve_sequence()
    later = asyncio.create_task(
        hub.publish(
            TextDelta(index=0, delta="peer-after-enter"),
            run_id="run",
            agent_id="peer",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.BACKGROUND,
        )
    )
    await asyncio.sleep(0)

    assert not later.done()
    assert observed == []

    enter = await hub.publish(
        TextDelta(index=0, delta="enter"),
        run_id="run",
        agent_id="main",
        parent_agent_id=None,
        turn_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        reserved_seq=accepted_enter_seq,
    )
    peer = await later

    assert enter.seq == accepted_enter_seq == 1
    assert peer.seq == 2
    assert observed == [(1, "enter"), (2, "peer-after-enter")]


async def test_event_hub_boundary_waits_for_an_already_reserved_ingress() -> None:
    hub = SessionEventHub(session_id="session-1")
    accepted_enter_seq = hub.reserve_sequence()
    boundary = asyncio.create_task(hub.wait_through_current_sequence())
    await asyncio.sleep(0)

    assert not boundary.done()

    await hub.publish(
        TextDelta(index=0, delta="enter"),
        run_id="run",
        agent_id="main",
        parent_agent_id=None,
        turn_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        reserved_seq=accepted_enter_seq,
    )

    assert await asyncio.wait_for(boundary, timeout=1) == accepted_enter_seq


async def test_event_hub_discarded_ingress_slot_releases_later_publication() -> None:
    hub = SessionEventHub(session_id="session-1")
    observed: list[tuple[int, str]] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        assert isinstance(envelope.event, TextDelta)
        observed.append((envelope.seq, envelope.event.delta))

    hub.subscribe(collect)
    command_seq = hub.reserve_sequence()
    later = asyncio.create_task(
        hub.publish(
            TextDelta(index=0, delta="peer-after-command"),
            run_id="run",
            agent_id="peer",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.BACKGROUND,
        )
    )
    await asyncio.sleep(0)

    assert not later.done()
    await hub.discard_reserved_sequence(command_seq)
    peer = await asyncio.wait_for(later, timeout=1)

    assert command_seq == 1
    assert peer.seq == 2
    assert observed == [(2, "peer-after-command")]
    with pytest.raises(ValueError, match="is not reserved"):
        await hub.discard_reserved_sequence(command_seq)


async def test_event_hub_cancellation_does_not_strand_a_later_publication() -> None:
    hub = SessionEventHub(session_id="session-1")
    delivering_reserved = asyncio.Event()
    observed: list[str] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        assert isinstance(envelope.event, TextDelta)
        if envelope.event.delta == "reserved":
            delivering_reserved.set()
            await asyncio.Future()
        observed.append(envelope.event.delta)

    hub.subscribe(collect)
    reserved_seq = hub.reserve_sequence()
    later = asyncio.create_task(
        hub.publish(
            TextDelta(index=0, delta="later"),
            run_id="run",
            agent_id="peer",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.BACKGROUND,
        )
    )
    await asyncio.sleep(0)
    reserved = asyncio.create_task(
        hub.publish(
            TextDelta(index=0, delta="reserved"),
            run_id="run",
            agent_id="main",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.FOREGROUND,
            reserved_seq=reserved_seq,
        )
    )
    await asyncio.wait_for(delivering_reserved.wait(), timeout=1)

    reserved.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reserved
    peer = await asyncio.wait_for(later, timeout=1)

    assert peer.seq == 2
    assert observed == ["later"]


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


async def test_observed_turn_preserves_refusal_text_in_failed_outcome() -> None:
    hub = SessionEventHub(session_id="session-1")
    identity = new_turn_identity(
        agent_id="child",
        parent_agent_id="parent",
        execution_mode=ExecutionMode.BACKGROUND,
    )
    transport = StubTransport(
        [[Refusal(index=0, text="policy refusal"), IterationEnd(1, StopReason.refusal, Usage(0, 0))]]
    )

    outcome = await observe_agent_turn(
        agent=Agent(system="child", transport=transport),
        context=MemoryContextStore(),
        prompt="work",
        identity=identity,
        hub=hub,
    )

    assert outcome.status is TurnStatus.FAILED
    assert outcome.stop_reason is StopReason.refusal
    assert outcome.text == "policy refusal"


async def test_observed_turn_preserves_a_distinct_ordered_input_batch() -> None:
    hub = SessionEventHub(session_id="session-1")
    events: list[object] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        events.append(envelope.event)

    hub.subscribe(collect)
    context = MemoryContextStore()
    messages = (
        Message(role="user", content=[TextBlock(text="first")]),
        Message(role="user", content=[TextBlock(text="peer")]),
        Message(role="user", content=[TextBlock(text="second")]),
    )
    identity = new_turn_identity(
        agent_id="main",
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        run_id="run-1",
    )

    outcome = await observe_agent_turn_messages(
        agent=Agent(system="main", transport=StubTransport([make_text_response("answer")])),
        context=context,
        messages=messages,
        identity=identity,
        hub=hub,
    )

    assert outcome.succeeded
    assert (await context.get_history())[:3] == list(messages)
    assert events[0] == TurnStarted(prompt="first\n\npeer\n\nsecond")


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


async def test_observed_input_batch_correlates_only_its_source_messages() -> None:
    hub = SessionEventHub(session_id="session-1")
    events: list[object] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        events.append(envelope.event)

    hub.subscribe(collect)
    context = ObservedContextStore(MemoryContextStore(), hub)
    messages = (
        Message(role="user", content=[TextBlock(text="interactive")]),
        Message(role="user", content=[TextBlock(text="peer")]),
    )
    identity = new_turn_identity(
        agent_id="main",
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        run_id="run-1",
        context_id=context.session_id,
    )

    await observe_agent_turn_messages(
        agent=Agent(system="main", transport=StubTransport([make_text_response("answer")])),
        context=context,
        messages=messages,
        identity=identity,
        hub=hub,
        source_input_ids=("input-1", None),
    )

    commits = [event for event in events if isinstance(event, MessageCommitted)]
    assert [event.source_input_id for event in commits] == ["input-1", None, None]


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


async def test_cancellation_during_commit_observation_cannot_split_tool_exchange() -> None:
    async def answer() -> str:
        return "tool answer"

    hub = SessionEventHub()
    assistant_commit_started = asyncio.Event()

    async def block_assistant_commit(envelope: AgentEventEnvelope) -> None:
        event = envelope.event
        if isinstance(event, MessageCommitted) and any(
            isinstance(block, ToolUseBlock) for block in event.message.content
        ):
            assistant_commit_started.set()
            await asyncio.Event().wait()

    hub.subscribe(block_assistant_commit)
    inner = MemoryContextStore()
    context = ObservedContextStore(inner, hub)
    identity = new_turn_identity(
        agent_id="main",
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        context_id=context.session_id,
    )
    agent = Agent(
        system="main",
        transport=StubTransport(
            [
                make_tool_use_response("answer", tool_id="call-1", tool_input={}),
                make_text_response("done"),
            ]
        ),
        tools=[Tool(name="answer", handler=answer)],
    )
    task = asyncio.create_task(
        observe_agent_turn(agent=agent, context=context, prompt="work", identity=identity, hub=hub)
    )
    await asyncio.wait_for(assistant_commit_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    history = await inner.get_history()
    tool_uses = [block for message in history for block in message.content if isinstance(block, ToolUseBlock)]
    tool_results = [block for message in history for block in message.content if isinstance(block, ToolResultBlock)]
    assert [block.id for block in tool_uses] == ["call-1"]
    assert [block.tool_use_id for block in tool_results] == ["call-1"]


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


class _PartialBlockingTransport:
    def __init__(self) -> None:
        self.partial_sent = asyncio.Event()

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools, system
        yield TextDelta(index=0, delta="available partial")
        self.partial_sent.set()
        await asyncio.Future()


async def test_cancelled_observed_turn_commits_available_partial_text_once() -> None:
    transport = _PartialBlockingTransport()
    context = MemoryContextStore()
    identity = new_turn_identity(
        agent_id="child",
        parent_agent_id="main",
        execution_mode=ExecutionMode.BACKGROUND,
    )
    task = asyncio.create_task(
        observe_agent_turn(
            agent=Agent(system="child", transport=transport),
            context=context,
            prompt="work",
            identity=identity,
            hub=SessionEventHub(),
        )
    )
    await transport.partial_sent.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    partial_messages = [
        message
        for message in await context.get_history()
        if message.role == "assistant" and message.content == [TextBlock(text="available partial")]
    ]
    assert len(partial_messages) == 1


async def test_repeat_cancel_cannot_tear_partial_commit_or_cancelled_lifecycle() -> None:
    transport = _PartialBlockingTransport()
    hub = SessionEventHub()
    inner = MemoryContextStore()
    context = ObservedContextStore(inner, hub)
    partial_commit_started = asyncio.Event()
    partial_commit_release = asyncio.Event()
    finished: list[TurnFinished] = []

    async def observe(envelope: AgentEventEnvelope) -> None:
        event = envelope.event
        if isinstance(event, MessageCommitted) and event.message.role == "assistant":
            partial_commit_started.set()
            await partial_commit_release.wait()
        elif isinstance(event, TurnFinished):
            finished.append(event)

    hub.subscribe(observe)
    identity = new_turn_identity(
        agent_id="child",
        parent_agent_id="main",
        execution_mode=ExecutionMode.BACKGROUND,
        context_id=context.session_id,
    )
    task = asyncio.create_task(
        observe_agent_turn(
            agent=Agent(system="child", transport=transport),
            context=context,
            prompt="work",
            identity=identity,
            hub=hub,
        )
    )
    await transport.partial_sent.wait()

    task.cancel("initial cancellation")
    await asyncio.wait_for(partial_commit_started.wait(), timeout=1)
    task.cancel("late cancellation")
    partial_commit_release.set()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(task, timeout=1)
    assert cancelled.value.args == ("initial cancellation",)

    partial_messages = [
        message
        for message in await inner.get_history()
        if message.role == "assistant" and message.content == [TextBlock(text="available partial")]
    ]
    assert len(partial_messages) == 1
    assert finished == [TurnFinished(status=TurnStatus.CANCELLED, stop_reason=None, error="turn cancelled")]
