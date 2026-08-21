"""Tests for Agent.run_stream() and run(): core loop, stop reasons, usage."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from axio.agent import Agent, ToolDispatch
from axio.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.events import (
    Error,
    IterationEnd,
    ReasoningDelta,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    ToolResult,
)
from axio.messages import InputProvenance, Message
from axio.testing import StubTransport, make_echo_tool, make_text_response, make_tool_use_response
from axio.tool import Tool
from axio.types import CostSource, StopReason, Usage


class CapturingTransport:
    """Records messages passed to each stream() call."""

    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        self._responses = responses
        self._call_count = 0
        self.calls: list[list[Message]] = []

    async def _generate(self, events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
        for event in events:
            yield event

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        self.calls.append(list(messages))
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._generate(self._responses[idx])


async def _ok(msg: str) -> str:
    return "ok"


class TestRunStream:
    async def test_end_turn_yields_text_and_session_end(self) -> None:
        transport = StubTransport(
            [
                [
                    TextDelta(0, "Hello"),
                    TextDelta(0, " world"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("hi", MemoryContextStore()):
            events.append(e)

        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_events) == 2
        last = events[-1]
        assert isinstance(last, SessionEndEvent)
        assert last.stop_reason == StopReason.end_turn

    async def test_session_end_total_usage(self) -> None:
        transport = StubTransport(
            [
                [
                    TextDelta(0, "hi"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        end = await agent.run_stream("hi", MemoryContextStore()).get_session_end()
        assert end.total_usage == Usage(10, 5)


class TestRun:
    async def test_returns_concatenated_text(self) -> None:
        transport = StubTransport(
            [
                [
                    TextDelta(0, "Hello"),
                    TextDelta(0, " world"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        result = await agent.run("hi", MemoryContextStore())
        assert result == "Hello world"

    async def test_direct_prompt_is_explicitly_human_authored(self) -> None:
        agent = Agent(system="test", tools=[], transport=StubTransport([make_text_response("done")]))
        context = MemoryContextStore()

        await agent.run("hi", context)

        [prompt, _response] = await context.get_history()
        assert prompt.provenance == InputProvenance(human_authored=True, source="direct", author="human")


class TestMessageBatchRun:
    async def test_preserves_distinct_ordered_messages_for_one_model_operation(self) -> None:
        transport = CapturingTransport([make_text_response("Done", 1)])
        agent = Agent(system="test", tools=[], transport=transport)
        context = MemoryContextStore()
        first = Message(role="user", content=[TextBlock(text="first")])
        peer = Message(role="user", content=[TextBlock(text="peer")])
        second = Message(role="user", content=[TextBlock(text="second")])

        result = await agent.run_messages((first, peer, second), context)

        assert result == "Done"
        history = await context.get_history()
        assert history[:3] == [first, peer, second]
        assert transport.calls[0][:3] == [first, peer, second]
        assert history[0] is not history[1]

    async def test_rejects_an_empty_message_batch_before_starting(self) -> None:
        agent = Agent(system="test", tools=[], transport=StubTransport([make_text_response("unused")]))

        with pytest.raises(ValueError, match="must not be empty"):
            agent.run_stream_messages((), MemoryContextStore())

    async def test_input_commit_hook_runs_after_context_append_and_before_transport(self) -> None:
        transport = CapturingTransport([make_text_response("Done", 1)])
        agent = Agent(system="test", tools=[], transport=transport)
        context = MemoryContextStore()
        message = Message(role="user", content=[TextBlock(text="queued")])
        observations: list[tuple[list[Message], int]] = []

        async def input_committed() -> None:
            observations.append((await context.get_history(), len(transport.calls)))

        await agent.run_messages((message,), context, on_input_committed=input_committed)

        assert observations == [([message], 0)]


class _RecordingDeferredSink:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.events: list[str] = []
        self.dispatch: ToolDispatch | None = None

    def dispatch_started(self, dispatch: ToolDispatch) -> None:
        self.dispatch = dispatch
        self.events.append("started")
        self.started.set()

    def dispatch_finished(self, dispatch: ToolDispatch) -> None:
        assert dispatch is self.dispatch
        self.events.append("finished")

    def defer(self, dispatch: ToolDispatch) -> None:
        assert dispatch is self.dispatch
        self.events.append("deferred")

    def should_defer(self, dispatch: ToolDispatch) -> bool:
        assert dispatch is self.dispatch
        return True

    def protocol_closed(self, dispatch: ToolDispatch) -> None:
        assert dispatch is self.dispatch
        self.events.append("protocol-closed")


async def test_cancelled_tool_dispatch_can_be_deferred_after_protocol_placeholder() -> None:
    release = asyncio.Event()

    async def slow_tool() -> str:
        await release.wait()
        return "actual result"

    sink = _RecordingDeferredSink()
    context = MemoryContextStore()
    agent = Agent(
        system="test",
        tools=[Tool[Any](name="slow", handler=slow_tool)],
        transport=StubTransport([make_tool_use_response("slow", "call-1", {})]),
        deferred_tool_sink=sink,
    )
    events: list[StreamEvent] = []

    async def consume() -> None:
        async for event in agent.run_stream("run", context):
            events.append(event)

    turn = asyncio.create_task(consume())
    await asyncio.wait_for(sink.started.wait(), timeout=1)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert sink.events == ["started", "deferred", "protocol-closed"]
    assert sink.dispatch is not None
    assert not sink.dispatch.task.done()
    history = await context.get_history()
    assistant = history[-2]
    placeholder_message = history[-1]
    assert any(isinstance(block, ToolUseBlock) and block.id == "call-1" for block in assistant.content)
    placeholders = [block for block in placeholder_message.content if isinstance(block, ToolResultBlock)]
    assert len(placeholders) == 1
    assert placeholders[0].tool_use_id == "call-1"
    assert "continues after interruption" in str(placeholders[0].content)
    assert not any(isinstance(event, ToolResult) for event in events)

    release.set()
    results = await asyncio.wait_for(sink.dispatch.task, timeout=1)
    assert results[0].content == "actual result"


async def test_cancelled_tool_dispatch_stops_when_sink_does_not_authorize_deferral() -> None:
    tool_cancelled = asyncio.Event()

    async def slow_tool() -> str:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            tool_cancelled.set()
            raise
        return "unreachable"

    class NonDeferringSink(_RecordingDeferredSink):
        def should_defer(self, dispatch: ToolDispatch) -> bool:
            assert dispatch is self.dispatch
            return False

    sink = NonDeferringSink()
    context = MemoryContextStore()
    agent = Agent(
        system="test",
        tools=[Tool[Any](name="slow", handler=slow_tool)],
        transport=StubTransport([make_tool_use_response("slow", "call-1", {})]),
        deferred_tool_sink=sink,
    )

    events: list[StreamEvent] = []

    async def consume() -> None:
        async for event in agent.run_stream("run", context):
            events.append(event)

    turn = asyncio.create_task(consume())
    await asyncio.wait_for(sink.started.wait(), timeout=1)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert sink.events == ["started", "finished"]
    await asyncio.wait_for(tool_cancelled.wait(), timeout=1)
    history = await context.get_history()
    results = [block for block in history[-1].content if isinstance(block, ToolResultBlock)]
    assert len(results) == 1
    assert results[0].is_error
    assert results[0].content == "[interrupted by user]"
    assert [
        (event.tool_use_id, event.name, event.is_error, event.content)
        for event in events
        if isinstance(event, ToolResult)
    ] == [("call-1", "slow", True, "[interrupted by user]")]


class TestMultiIteration:
    async def test_tool_use_then_end_turn(self) -> None:
        tool = make_echo_tool()
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_text_response("Done", 2),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("do it", MemoryContextStore()):
            events.append(e)

        iteration_ends = [e for e in events if isinstance(e, IterationEnd)]
        assert len(iteration_ends) == 2
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        last = events[-1]
        assert isinstance(last, SessionEndEvent)
        assert last.stop_reason == StopReason.end_turn

    async def test_total_usage_across_iterations(self) -> None:
        tool = make_echo_tool()
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1, Usage(10, 5)),
                make_text_response("Done", 2, Usage(3, 7)),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        end = await agent.run_stream("go", MemoryContextStore()).get_session_end()
        assert end.total_usage == Usage(13, 12)

    async def test_total_provider_cost_across_iterations(self) -> None:
        tool = make_echo_tool()
        first = Usage(10, 5, cost_usd=0.1, cost_source=CostSource.provider)
        second = Usage(3, 7, cost_usd=0.2, cost_source=CostSource.provider)
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1, first),
                make_text_response("Done", 2, second),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)

        end = await agent.run_stream("go", MemoryContextStore()).get_session_end()

        assert (end.total_usage.input_tokens, end.total_usage.output_tokens) == (13, 12)
        assert end.total_usage.cost_usd == pytest.approx(0.3)
        assert end.total_usage.cost_source is CostSource.provider


class TestContextTokenTracking:
    async def test_agent_updates_context_tokens(self) -> None:
        transport = StubTransport(
            [
                [
                    TextDelta(0, "hi"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        context = MemoryContextStore()
        await agent.run("go", context)
        assert await context.get_context_tokens() == (10, 5)

    async def test_agent_accumulates_context_tokens_across_iterations(self) -> None:
        tool = make_echo_tool()
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1, Usage(10, 5)),
                make_text_response("Done", 2, Usage(3, 7)),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        context = MemoryContextStore()
        await agent.run("go", context)
        assert await context.get_context_tokens() == (13, 12)


class TestReasoningPassthrough:
    async def test_reasoning_delta_yielded_but_not_stored(self) -> None:
        """ReasoningDelta events pass through the stream but are NOT stored in context."""
        transport = StubTransport(
            [
                [
                    ReasoningDelta(0, "thinking..."),
                    TextDelta(0, "answer"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        context = MemoryContextStore()
        events: list[StreamEvent] = []
        async for e in agent.run_stream("hi", context):
            events.append(e)

        # ReasoningDelta is yielded
        reasoning = [e for e in events if isinstance(e, ReasoningDelta)]
        assert len(reasoning) == 1
        assert reasoning[0].delta == "thinking..."

        # TextDelta is yielded
        text = [e for e in events if isinstance(e, TextDelta)]
        assert len(text) == 1
        assert text[0].delta == "answer"

        # Only text is stored in assistant message, not reasoning
        history = await context.get_history()
        assistant_msgs = [m for m in history if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        from axio.blocks import TextBlock

        text_blocks = [b for b in assistant_msgs[0].content if isinstance(b, TextBlock)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "answer"

    async def test_repetitive_reasoning_is_truncated_before_provider_completion(self) -> None:
        repeated_chunk = "— " * 100
        provider_events: list[StreamEvent] = [ReasoningDelta(0, repeated_chunk) for _ in range(10)]
        provider_events.append(IterationEnd(1, StopReason.end_turn, Usage(10, 5)))
        agent = Agent(system="test", tools=[], transport=StubTransport([provider_events]))
        context = MemoryContextStore()

        events = [event async for event in agent.run_stream("hi", context)]

        reasoning = [event for event in events if isinstance(event, ReasoningDelta)]
        notices = [event for event in events if isinstance(event, TextDelta)]
        assert len(reasoning) < 10
        assert [event.delta for event in notices] == ["\n\n[Output truncated: repetitive content detected]"]
        assert isinstance(events[-1], SessionEndEvent)
        history = await context.get_history()
        assistant = next(message for message in history if message.role == "assistant")
        assert assistant.content == [TextBlock(text="\n\n[Output truncated: repetitive content detected]")]

    async def test_short_repeated_reasoning_does_not_trigger_loop_detection(self) -> None:
        transport = StubTransport(
            [
                [
                    ReasoningDelta(0, "checking " * 30),
                    TextDelta(0, "answer"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)

        events = [event async for event in agent.run_stream("hi", MemoryContextStore())]

        assert not any(isinstance(event, TextDelta) and "Output truncated" in event.delta for event in events)
        assert any(isinstance(event, TextDelta) and event.delta == "answer" for event in events)


class TestMaxIterations:
    async def test_max_iterations_reached(self) -> None:
        """C7: max_iterations emits SessionEndEvent(stop_reason=error)."""
        tool: Tool[Any] = Tool(name="echo", description="echo", handler=_ok)
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_tool_use_response("echo", "c2", {"msg": "hi"}, 2),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport, max_iterations=1)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        last = events[-1]
        assert isinstance(last, SessionEndEvent)
        assert last.stop_reason == StopReason.error

    async def test_max_iterations_is_announced_not_only_logged(self) -> None:
        """An agent watched from elsewhere must not look like it succeeded."""
        tool: Tool[Any] = Tool(name="echo", description="echo", handler=_ok)
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_tool_use_response("echo", "c2", {"msg": "hi"}, 2),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport, max_iterations=1)
        events = [e async for e in agent.run_stream("go", MemoryContextStore())]

        errors = [e for e in events if isinstance(e, Error)]
        assert len(errors) == 1
        assert "1 iterations" in str(errors[0].exception)


class TestLastIterationMessage:
    async def test_injected_only_on_last_iteration(self) -> None:
        """last_iteration_message is appended to history only on the final iteration."""
        tool: Tool[Any] = Tool(name="echo", description="echo", handler=_ok)
        hint = Message(role="system", content=[TextBlock(text="wrap up now")])
        transport = CapturingTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_text_response("Done", 2),
            ]
        )
        agent = Agent(
            system="test",
            tools=[tool],
            transport=transport,
            max_iterations=2,
            last_iteration_message=hint,
        )
        await agent.run("go", MemoryContextStore())

        # iteration 1: hint NOT in history
        assert hint not in transport.calls[0]
        # iteration 2 (last): hint IS the final message
        assert transport.calls[1][-1] is hint

    async def test_not_injected_when_none(self) -> None:
        """No injection when last_iteration_message is None (default)."""
        transport = CapturingTransport([make_text_response("hi", 1)])
        agent = Agent(system="test", tools=[], transport=transport)
        await agent.run("go", MemoryContextStore())

        history = transport.calls[0]
        assert all(m.role != "system" for m in history)

    async def test_not_stored_in_context(self) -> None:
        """last_iteration_message is injected into the stream but not persisted."""
        tool: Tool[Any] = Tool(name="echo", description="echo", handler=_ok)
        hint = Message(role="system", content=[TextBlock(text="wrap up")])
        transport = CapturingTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_text_response("Done", 2),
            ]
        )
        agent = Agent(
            system="test",
            tools=[tool],
            transport=transport,
            max_iterations=2,
            last_iteration_message=hint,
        )
        context = MemoryContextStore()
        await agent.run("go", context)

        history = await context.get_history()
        assert hint not in history
