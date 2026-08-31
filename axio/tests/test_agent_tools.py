"""Tests for Agent tool dispatch: invocation, errors, parallel execution."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

import pytest

from axio.agent import Agent, ToolExecutionTiming
from axio.blocks import ImageBlock, ToolResultBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.events import (
    IterationEnd,
    SessionEndEvent,
    StreamEvent,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.exceptions import GuardError, HandlerError
from axio.messages import InputProvenance, Message
from axio.permission import PermissionGuard
from axio.testing import StubTransport, make_echo_tool, make_text_response, make_tool_use_response
from axio.tool import CURRENT_TOOL_CALL, Tool, ToolCallContext
from axio.types import StopReason, Usage

calls_log: list[dict[str, Any]] = []


async def _tracking(msg: str) -> str:
    data = {"msg": msg}
    calls_log.append(data)
    return json.dumps(data)


async def _handler_x(x: int) -> str:
    return "a"


async def _handler_y(y: int) -> str:
    return "b"


async def _bad(**kwargs: object) -> str:
    raise ValueError("boom")


class TestToolInvocation:
    async def test_handler_called(self) -> None:
        calls_log.clear()
        tool: Tool[Any] = Tool(name="echo", description="echo", handler=_tracking)
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        await agent.run("go", MemoryContextStore())
        assert len(calls_log) == 1
        assert calls_log[0] == {"msg": "hi"}

    async def test_result_in_context(self) -> None:
        tool = make_echo_tool()
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        ctx = MemoryContextStore()
        agent = Agent(system="test", tools=[tool], transport=transport)
        await agent.run("go", ctx)
        history = await ctx.get_history()
        user_msgs = [m for m in history if m.role == "user"]
        tool_results = [b for m in user_msgs for b in m.content if isinstance(b, ToolResultBlock)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_use_id == "c1"
        assert not tool_results[0].is_error


class TestTwoToolsOneResponse:
    async def test_both_called(self) -> None:
        """C2: every ToolUseBlock has a corresponding ToolResultBlock."""
        calls: list[str] = []

        async def _a(x: int) -> str:
            calls.append("a")
            return "a"

        async def _b(y: int) -> str:
            calls.append("b")
            return "b"

        tool_a: Tool[Any] = Tool(name="a", description="a", handler=_a)
        tool_b: Tool[Any] = Tool(name="b", description="b", handler=_b)
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "a"),
                    ToolInputDelta(0, "c1", json.dumps({"x": 1})),
                    ToolUseStart(1, "c2", "b"),
                    ToolInputDelta(1, "c2", json.dumps({"y": 2})),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[tool_a, tool_b], transport=transport)
        await agent.run("go", MemoryContextStore())
        assert set(calls) == {"a", "b"}


class TestCurrentToolCall:
    async def test_non_streaming_handler_sees_correlation_and_caller_is_restored(self) -> None:
        seen: list[ToolCallContext] = []

        async def handler() -> str:
            seen.append(CURRENT_TOOL_CALL.get())
            return "ok"

        tool: Tool[Any] = Tool(name="inspect", handler=handler)
        agent = Agent(system="test", tools=[tool], transport=StubTransport())
        outer = ToolCallContext(tool_use_id="outer", tool_name="outer", iteration=0)
        token = CURRENT_TOOL_CALL.set(outer)
        try:
            [result] = await agent.dispatch_tools([ToolUseBlock(id="call-7", name="inspect", input={})], 7)
            assert CURRENT_TOOL_CALL.get() is outer
        finally:
            CURRENT_TOOL_CALL.reset(token)

        assert not result.is_error
        assert seen == [ToolCallContext(tool_use_id="call-7", tool_name="inspect", iteration=7)]

    async def test_concurrent_handlers_have_isolated_correlation(self) -> None:
        seen: dict[str, tuple[ToolCallContext, ToolCallContext]] = {}

        async def handler() -> str:
            before = CURRENT_TOOL_CALL.get()
            await asyncio.sleep(0)
            after = CURRENT_TOOL_CALL.get()
            seen[before.tool_use_id] = (before, after)
            return before.tool_use_id

        tools: list[Tool[Any]] = [Tool(name="a", handler=handler), Tool(name="b", handler=handler)]
        agent = Agent(system="test", tools=tools, transport=StubTransport())
        blocks = [ToolUseBlock(id="call-a", name="a", input={}), ToolUseBlock(id="call-b", name="b", input={})]
        results = await agent.dispatch_tools(blocks, 3)

        assert [result.content for result in results] == ["call-a", "call-b"]
        assert seen == {
            "call-a": (
                ToolCallContext(tool_use_id="call-a", tool_name="a", iteration=3),
                ToolCallContext(tool_use_id="call-a", tool_name="a", iteration=3),
            ),
            "call-b": (
                ToolCallContext(tool_use_id="call-b", tool_name="b", iteration=3),
                ToolCallContext(tool_use_id="call-b", tool_name="b", iteration=3),
            ),
        }

    async def test_failed_handler_restores_caller_context(self) -> None:
        async def handler() -> str:
            assert CURRENT_TOOL_CALL.get().tool_use_id == "failed-call"
            raise RuntimeError("failed")

        tool: Tool[Any] = Tool(name="fail", handler=handler)
        agent = Agent(system="test", tools=[tool], transport=StubTransport())
        outer = ToolCallContext(tool_use_id="outer", tool_name="outer", iteration=0)
        token = CURRENT_TOOL_CALL.set(outer)
        try:
            [result] = await agent.dispatch_tools([ToolUseBlock(id="failed-call", name="fail", input={})], 2)
            assert CURRENT_TOOL_CALL.get() is outer
        finally:
            CURRENT_TOOL_CALL.reset(token)

        assert result.is_error

    async def test_streaming_handler_sees_correlation_until_stream_finishes(self) -> None:
        seen: list[ToolCallContext] = []

        async def handler() -> str:
            return "unused"

        async def stream() -> AsyncGenerator[tuple[str, str], None]:
            seen.append(CURRENT_TOOL_CALL.get())
            yield "stdout", "one"
            await asyncio.sleep(0)
            seen.append(CURRENT_TOOL_CALL.get())
            yield "stdout", "two"

        handler.stream = stream  # type: ignore[attr-defined]
        tool: Tool[Any] = Tool(name="stream", handler=handler)
        agent = Agent(system="test", tools=[tool], transport=StubTransport())
        queue: asyncio.Queue[ToolOutputDelta | None] = asyncio.Queue()
        outer = ToolCallContext(tool_use_id="outer", tool_name="outer", iteration=0)
        token = CURRENT_TOOL_CALL.set(outer)
        try:
            [result] = await agent._dispatch_tools_streaming(
                [ToolUseBlock(id="stream-call", name="stream", input={})], 9, queue
            )
            assert CURRENT_TOOL_CALL.get() is outer
        finally:
            CURRENT_TOOL_CALL.reset(token)

        expected = ToolCallContext(tool_use_id="stream-call", tool_name="stream", iteration=9)
        assert seen == [expected, expected]
        assert result.content == "onetwo"


class TestUnknownTool:
    async def test_produces_error_result(self) -> None:
        """C9: unknown tool produces is_error=True, loop continues."""
        transport = StubTransport([make_tool_use_response("nonexistent", "c1", {}), make_text_response("Done")])
        agent = Agent(system="test", tools=[], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error


class TestHandlerException:
    async def test_exception_wrapped_as_error_result(self) -> None:
        tool: Tool[Any] = Tool(name="bad", description="bad", handler=_bad)
        transport = StubTransport([make_tool_use_response("bad", "c1", {}), make_text_response("Done")])
        ctx = MemoryContextStore()
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", ctx):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error
        assert isinstance(events[-1], SessionEndEvent)


class _RaisingGuard(PermissionGuard):
    def __init__(self, factory: Callable[[], Exception]) -> None:
        self._factory = factory

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        raise self._factory()


def _build_failing_tool(
    *,
    handler_error: Callable[[], Exception] | None = None,
    guard_error: Callable[[], Exception] | None = None,
    streaming: bool = False,
) -> Tool[Any]:
    async def handler() -> str:
        if handler_error is not None:
            raise handler_error()
        return "ok"

    if streaming:

        async def stream() -> AsyncGenerator[tuple[str, str], None]:
            if handler_error is not None:
                raise handler_error()
            yield ("output", "ok")

        handler.stream = stream  # type: ignore[attr-defined]

    guards = (_RaisingGuard(guard_error),) if guard_error is not None else ()
    return Tool(name="failing", handler=handler, guards=guards)


async def _dispatch_one(tool: Tool[Any], mode: str) -> ToolResultBlock:
    agent = Agent(system="test", tools=[tool], transport=StubTransport())
    block = ToolUseBlock(id="c1", name="failing", input={})
    if mode == "dispatch_tools":
        [result] = await agent.dispatch_tools([block], 1)
        return result
    queue: asyncio.Queue[ToolOutputDelta | None] = asyncio.Queue()
    [result] = await agent._dispatch_tools_streaming([block], 1, queue)
    return result


def _failure_logs(caplog: pytest.LogCaptureFixture) -> list[tuple[int, bool, str]]:
    return [
        (r.levelno, r.exc_info is not None, r.getMessage())
        for r in caplog.records
        if r.name == "axio.agent" and r.getMessage().startswith("Tool failing")
    ]


_DISPATCH_MODES = [
    pytest.param("dispatch_tools", False, id="dispatch_tools"),
    pytest.param("streaming_dispatch", False, id="streaming_dispatch-plain_tool"),
    pytest.param("streaming_dispatch", True, id="streaming_dispatch-streaming_tool"),
]


class TestToolFailureLogging:
    """Expected tool failures are logged as such; only crashes get a traceback."""

    @pytest.mark.parametrize("mode, streaming", _DISPATCH_MODES)
    async def test_handler_error_logs_at_info(
        self, mode: str, streaming: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        tool = _build_failing_tool(handler_error=lambda: HandlerError("file not found"), streaming=streaming)
        with caplog.at_level(logging.INFO, logger="axio.agent"):
            result = await _dispatch_one(tool, mode)

        assert result.is_error
        assert result.content == "file not found"
        assert _failure_logs(caplog) == [(logging.INFO, False, "Tool failing failed: file not found")]
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    @pytest.mark.parametrize("mode, streaming", _DISPATCH_MODES)
    async def test_handler_crash_logs_at_error_with_traceback(
        self, mode: str, streaming: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        tool = _build_failing_tool(handler_error=lambda: ValueError("boom"), streaming=streaming)
        with caplog.at_level(logging.INFO, logger="axio.agent"):
            result = await _dispatch_one(tool, mode)

        assert result.is_error
        assert result.content == "ValueError: boom"
        assert _failure_logs(caplog) == [
            (logging.ERROR, True, "Tool failing raised HandlerCrash: ValueError: boom"),
        ]

    @pytest.mark.parametrize("mode, streaming", _DISPATCH_MODES)
    async def test_guard_denial_logs_at_info(
        self, mode: str, streaming: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        tool = _build_failing_tool(guard_error=lambda: GuardError("denied by policy"), streaming=streaming)
        with caplog.at_level(logging.INFO, logger="axio.agent"):
            result = await _dispatch_one(tool, mode)

        assert result.is_error
        assert result.content == "denied by policy"
        assert _failure_logs(caplog) == [(logging.INFO, False, "Tool failing failed: denied by policy")]
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    @pytest.mark.parametrize("mode, streaming", _DISPATCH_MODES)
    async def test_guard_crash_logs_at_error_with_traceback(
        self, mode: str, streaming: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        tool = _build_failing_tool(guard_error=lambda: RuntimeError("guard bug"), streaming=streaming)
        with caplog.at_level(logging.INFO, logger="axio.agent"):
            result = await _dispatch_one(tool, mode)

        assert result.is_error
        assert result.content == "RuntimeError: guard bug"
        assert _failure_logs(caplog) == [
            (logging.ERROR, True, "Tool failing raised GuardCrash: RuntimeError: guard bug"),
        ]

    async def test_cancelled_handler_propagates_out_of_dispatch(self) -> None:
        """Cancellation is not a tool failure: it must not be reported to the model."""

        async def handler() -> str:
            raise asyncio.CancelledError

        agent = Agent(system="test", tools=[Tool(name="cancelled", handler=handler)], transport=StubTransport())
        with pytest.raises(asyncio.CancelledError):
            await agent.dispatch_tools([ToolUseBlock(id="c1", name="cancelled", input={})], 1)


class TestMalformedJson:
    async def test_malformed_json_returns_error_result(self) -> None:
        """Truncated JSON → ToolResult(is_error=True), loop continues."""
        tool = make_echo_tool()
        # Truncated JSON: '{"directory": ".'  (missing closing quote and brace)
        truncated = '{"msg": ".'
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "list_files"),
                    ToolInputDelta(0, "c1", truncated),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error
        assert tool_results[0].tool_use_id == "c1"

        # Loop should continue - we get a SessionEndEvent with end_turn
        session_ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert len(session_ends) == 1
        assert session_ends[0].stop_reason == StopReason.end_turn

    async def test_mixed_valid_and_malformed_tools(self) -> None:
        """Two parallel tool calls: one valid, one malformed. Valid runs, malformed errors."""
        tool = make_echo_tool()
        valid_args = json.dumps({"msg": "hello"})
        malformed_args = '{"msg": "trunc'

        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "echo"),
                    ToolInputDelta(0, "c1", valid_args),
                    ToolUseStart(1, "c2", "echo"),
                    ToolInputDelta(1, "c2", malformed_args),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 2

        valid_result = next(r for r in tool_results if r.tool_use_id == "c1")
        malformed_result = next(r for r in tool_results if r.tool_use_id == "c2")

        assert not valid_result.is_error
        assert malformed_result.is_error


class TestToolResultCarriesData:
    async def test_content_and_input_populated(self) -> None:
        """ToolResult events carry the tool input dict and result content string."""
        tool = make_echo_tool()
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        r = tool_results[0]
        assert r.input == {"msg": "hi"}
        assert r.content != ""
        assert not r.is_error

    async def test_streamed_string_whitespace_reaches_context_handler_and_result_exactly(self) -> None:
        expected = "        first line\n\tsecond line"
        received: list[str] = []

        async def capture(content: str) -> str:
            received.append(content)
            return content

        raw = json.dumps({"content": expected})
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "patch_file"),
                    ToolInputDelta(0, "c1", raw[:17]),
                    ToolInputDelta(0, "c1", raw[17:25]),
                    ToolInputDelta(0, "c1", raw[25:]),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        context = MemoryContextStore()
        agent = Agent(system="test", tools=[Tool(name="patch_file", handler=capture)], transport=transport)
        events = [event async for event in agent.run_stream("go", context)]

        result = next(event for event in events if isinstance(event, ToolResult))
        assert result.input == {"content": expected}
        assert result.content == expected
        assert received == [expected]

        history = await context.get_history()
        tool_use = next(
            block
            for message in history
            if message.role == "assistant"
            for block in message.content
            if isinstance(block, ToolUseBlock)
        )
        tool_result = next(
            block
            for message in history
            if message.role == "user"
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        assert tool_use.input == {"content": expected}
        assert tool_result.content == expected

    async def test_error_result_has_content(self) -> None:
        """Error ToolResult events carry the error message as content."""
        tool: Tool[Any] = Tool(name="bad", description="bad", handler=_bad)
        transport = StubTransport([make_tool_use_response("bad", "c1", {}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        r = tool_results[0]
        assert r.is_error
        assert "boom" in r.content


class TestStopReasonOverride:
    async def test_stop_reason_override_when_tool_blocks_present(self) -> None:
        """Transport returns end_turn with tool calls → agent overrides to tool_use and dispatches."""
        tool = make_echo_tool()
        # Transport returns end_turn but includes tool call events
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "echo"),
                    ToolInputDelta(0, "c1", json.dumps({"msg": "hi"})),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        # Tool should have been dispatched despite end_turn
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error

        # Session should end with end_turn (from the second iteration's text response)
        session_ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert len(session_ends) == 1
        assert session_ends[0].stop_reason == StopReason.end_turn


class TestRefusedDispatch:
    async def test_a_refused_call_is_kept_in_history_with_a_result_saying_why(self) -> None:
        """A turn whose reason does not vouch for its calls still has to say so to the model.

        Stripped from the history instead, the next request cannot tell the attempt from a turn
        that called nothing, and the model makes the same call again.
        """
        ran: list[str] = []

        async def handler(msg: str) -> str:
            ran.append(msg)
            return msg

        tool: Tool[Any] = Tool(name="echo", description="echo", handler=handler)
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "echo"),
                    ToolInputDelta(0, "c1", json.dumps({"msg": "hi"})),
                    IterationEnd(1, StopReason.max_tokens, Usage(10, 5)),
                ]
            ]
        )
        context = MemoryContextStore()
        agent = Agent(system="test", tools=[tool], transport=transport)
        events = [e async for e in agent.run_stream("go", context)]

        assert ran == [], "a turn that does not vouch for its calls must not run them"

        history = await context.get_history()
        calls = [b for m in history for b in m.content if isinstance(b, ToolUseBlock)]
        results = [b for m in history for b in m.content if isinstance(b, ToolResultBlock)]
        assert [b.id for b in calls] == ["c1"], "the attempted call is what the next turn reads"
        assert [b.tool_use_id for b in results] == ["c1"], "a stored call with no result is refused next"
        assert results[0].is_error
        assert "max_tokens" in str(results[0].content)
        result_message = next(m for m in history if results[0] in m.content)
        assert result_message.provenance == InputProvenance(
            human_authored=False,
            source="tool-result",
            author="axio",
        )

        # The call was reported as started, so a caller with no result for it waits forever.
        reported = [e for e in events if isinstance(e, ToolResult)]
        assert [(e.tool_use_id, e.is_error) for e in reported] == [("c1", True)]
        assert [e.stop_reason for e in events if isinstance(e, SessionEndEvent)] == [StopReason.max_tokens]

    async def test_refused_call_and_result_share_one_atomic_store_boundary(self) -> None:
        class FailRefusalBatchStore(MemoryContextStore):
            def __init__(self) -> None:
                super().__init__()
                self.batches = 0

            async def append_many(self, messages: list[Message]) -> None:
                self.batches += 1
                if self.batches == 2:
                    raise OSError("refusal batch failed")
                await super().append_many(messages)

        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "echo"),
                    ToolInputDelta(0, "c1", json.dumps({"msg": "hi"})),
                    IterationEnd(1, StopReason.refusal, Usage(10, 5)),
                ]
            ]
        )
        context = FailRefusalBatchStore()
        agent = Agent(system="test", tools=[make_echo_tool()], transport=transport)

        with pytest.raises(OSError, match="refusal batch failed"):
            async for _ in agent.run_stream("go", context):
                pass

        history = await context.get_history()
        assert [message.role for message in history] == ["user"]
        assert not any(
            isinstance(block, ToolUseBlock | ToolResultBlock) for message in history for block in message.content
        )


class TestStreamingToolDispatch:
    async def test_streaming_handler_emits_keyed_output_deltas(self) -> None:
        """A tool with .stream attribute emits ToolOutputDelta events with key per field."""

        async def _stream(msg: str) -> AsyncGenerator[tuple[str, str], None]:
            yield ("stdout", "line1\n")
            yield ("stderr", "warn\n")
            yield ("stdout", "line2\n")

        async def streaming_handler(msg: str) -> str:
            parts = []
            async for _, t in _stream(msg):
                parts.append(t)
            return "".join(parts)

        streaming_handler.stream = _stream  # type: ignore[attr-defined]

        tool: Tool[object] = Tool(name="streamer", description="streams", handler=streaming_handler)
        transport = StubTransport(
            [make_tool_use_response("streamer", "c1", {"msg": "hi"}), make_text_response("Done")]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        deltas = [e for e in events if isinstance(e, ToolOutputDelta)]
        assert len(deltas) == 3
        assert deltas[0].key == "stdout"
        assert deltas[0].delta == "line1\n"
        assert deltas[1].key == "stderr"
        assert deltas[1].delta == "warn\n"
        assert deltas[2].key == "stdout"
        assert deltas[2].delta == "line2\n"
        assert deltas[0].name == "streamer"

        results = [e for e in events if isinstance(e, ToolResult)]
        assert len(results) == 1
        assert results[0].content == "line1\nwarn\nline2\n"
        assert not results[0].is_error

    async def test_non_streaming_tool_no_output_deltas(self) -> None:
        """A normal tool (no .stream) produces no ToolOutputDelta events."""
        tool: Tool[object] = Tool(name="echo", description="echo", handler=_tracking)
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        deltas = [e for e in events if isinstance(e, ToolOutputDelta)]
        assert len(deltas) == 0

    async def test_mixed_streaming_and_non_streaming(self) -> None:
        """Parallel dispatch: one streaming, one normal."""

        async def _stream(msg: str) -> AsyncGenerator[tuple[str, str], None]:
            yield ("output", "s1")
            yield ("output", "s2")

        async def streaming_handler(msg: str) -> str:
            parts = []
            async for _, t in _stream(msg):
                parts.append(t)
            return "".join(parts)

        streaming_handler.stream = _stream  # type: ignore[attr-defined]

        stream_tool: Tool[object] = Tool(name="streamer", description="streams", handler=streaming_handler)
        normal_tool: Tool[object] = Tool(name="echo", description="echo", handler=_tracking)

        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "streamer"),
                    ToolInputDelta(0, "c1", json.dumps({"msg": "hi"})),
                    ToolUseStart(1, "c2", "echo"),
                    ToolInputDelta(1, "c2", json.dumps({"msg": "world"})),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[stream_tool, normal_tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        deltas = [e for e in events if isinstance(e, ToolOutputDelta)]
        assert len(deltas) == 2
        assert all(d.name == "streamer" for d in deltas)

        results = [e for e in events if isinstance(e, ToolResult)]
        assert len(results) == 2
        assert {event.tool_use_id for event in results} == {"c1", "c2"}
        assert all(event.started_at is not None and event.started_at.tzinfo is not None for event in results)
        assert all(
            event.started_at is not None
            and event.finished_at is not None
            and event.finished_at >= event.started_at
            and event.duration_seconds is not None
            and event.duration_seconds >= 0
            for event in results
        )

    async def test_tool_timing_excludes_late_result_consumption(self) -> None:
        completed = asyncio.Event()

        async def immediate() -> str:
            completed.set()
            return "done"

        tool: Tool[object] = Tool(name="immediate", handler=immediate)
        agent = Agent(system="test", tools=[tool], transport=StubTransport())
        queue: asyncio.Queue[ToolOutputDelta | None] = asyncio.Queue()
        timings: dict[str, ToolExecutionTiming] = {}
        task = asyncio.create_task(
            agent._dispatch_tools_streaming(
                [ToolUseBlock(id="c1", name="immediate", input={})],
                1,
                queue,
                timings,
            )
        )
        await completed.wait()
        await asyncio.sleep(0)
        measured = timings["c1"].duration_seconds

        await asyncio.sleep(0.05)
        [result] = await task

        assert result.duration_seconds == measured

    @pytest.mark.parametrize("emit_chunk", [False, True])
    async def test_self_cancelled_streaming_tool_closes_dispatch(self, emit_chunk: bool) -> None:
        closed = asyncio.Event()

        async def _stream() -> AsyncGenerator[tuple[str, str], None]:
            try:
                if emit_chunk:
                    yield "stdout", "partial\n"
                raise asyncio.CancelledError
            finally:
                closed.set()

        async def streaming_handler() -> str:
            return "unreachable"

        streaming_handler.stream = _stream  # type: ignore[attr-defined]
        context = MemoryContextStore()
        agent = Agent(
            system="test",
            tools=[Tool(name="streamer", handler=streaming_handler)],
            transport=StubTransport(
                [make_tool_use_response("streamer", "cancelled-call", {}), make_text_response("unreachable")]
            ),
        )

        async def consume() -> None:
            async for _event in agent.run_stream("go", context):
                pass

        task = asyncio.create_task(consume())
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        assert closed.is_set()
        history = await context.get_history()
        tool_uses = [block for message in history for block in message.content if isinstance(block, ToolUseBlock)]
        tool_results = [
            block for message in history for block in message.content if isinstance(block, ToolResultBlock)
        ]
        assert [block.id for block in tool_uses] == ["cancelled-call"]
        assert [block.tool_use_id for block in tool_results] == ["cancelled-call"]
        assert tool_results[0].is_error
        assert "[interrupted by user]" in str(tool_results[0].content)
        assert ("partial" in str(tool_results[0].content)) is emit_chunk
        assert tool_results[0].started_at is not None
        assert tool_results[0].finished_at is not None
        assert tool_results[0].duration_seconds is not None

    async def test_streaming_tool_base_exception_closes_dispatch_without_committing_tool_use(self) -> None:
        class FatalToolError(BaseException):
            pass

        closed = asyncio.Event()

        async def _stream() -> AsyncGenerator[tuple[str, str], None]:
            try:
                raise FatalToolError("fatal stream")
                yield "stdout", "unreachable"
            finally:
                closed.set()

        async def streaming_handler() -> str:
            return "unreachable"

        streaming_handler.stream = _stream  # type: ignore[attr-defined]
        context = MemoryContextStore()
        agent = Agent(
            system="test",
            tools=[Tool(name="streamer", handler=streaming_handler)],
            transport=StubTransport([make_tool_use_response("streamer", "fatal-call", {})]),
        )

        async def consume() -> None:
            async for _event in agent.run_stream("go", context):
                pass

        with pytest.raises(FatalToolError, match="fatal stream"):
            await asyncio.wait_for(consume(), timeout=1)

        assert closed.is_set()
        history = await context.get_history()
        assert not any(
            isinstance(block, ToolUseBlock | ToolResultBlock) for message in history for block in message.content
        )


class TestDispatchThatBreaksInsteadOfFinishing:
    """A streaming dispatch talks to the reader through a queue, ended by one sentinel."""

    async def test_a_dispatch_that_raises_does_not_hang_the_turn(self) -> None:
        # Put only on the way out of a normal return, the sentinel never arrived when the dispatch
        # raised, and the reader waited on a queue nothing would fill again.
        async def _stream(msg: str) -> AsyncGenerator[tuple[str, str], None]:
            yield ("stdout", msg)

        async def streaming_handler(msg: str) -> str:
            return msg

        streaming_handler.stream = _stream  # type: ignore[attr-defined]
        tool: Tool[object] = Tool(name="streamer", description="streams", handler=streaming_handler)

        class _Broken(Agent):
            async def _dispatch_tools_streaming(
                self,
                blocks: list[ToolUseBlock],
                iteration: int,
                output_queue: asyncio.Queue[Any],
                execution_timings: dict[str, ToolExecutionTiming] | None = None,
            ) -> list[ToolResultBlock]:
                del execution_timings
                raise RuntimeError("the dispatch itself broke")

        agent = _Broken(
            system="",
            tools=[tool],
            transport=StubTransport([make_tool_use_response("streamer", "c1", {"msg": "hi"})]),
        )

        events: list[StreamEvent] = []
        # The timeout is the assertion: the reader waited on a queue nothing would fill again, and
        # the turn hung with no error and no timeout of its own.
        async with asyncio.timeout(5):
            with pytest.raises(RuntimeError, match="the dispatch itself broke"):
                async for event in agent.run_stream("go", MemoryContextStore()):
                    events.append(event)

        assert [e.stop_reason for e in events if isinstance(e, SessionEndEvent)] == [StopReason.error]


class TestTheMediaNudge:
    """Gemini ends a turn holding media in about twenty tokens, so the agent asks it to go on."""

    @staticmethod
    async def _stored() -> list[Message]:
        async def picture() -> list[ImageBlock]:
            return [ImageBlock(media_type="image/png", data=b"\x89PNG")]

        tool: Tool[Any] = Tool(name="picture", description="a picture", handler=picture)

        class _Nudging(StubTransport):
            nudge_on_media_tool_result = True

        transport = _Nudging(
            [
                make_tool_use_response("picture", "c1", {}),
                make_text_response("done"),
            ]
        )
        context = MemoryContextStore()
        async for _ in Agent(system="", tools=[tool], transport=transport).run_stream("go", context):
            pass
        return await context.get_history()

    async def test_it_travels_in_the_message_the_results_are_in(self) -> None:
        # Appended as a message of its own it made two user turns in a row, which Anthropic
        # refuses outright and Gemini survives only because its converter merges them.
        history = await self._stored()

        user_turns = [m for m in history if m.role == "user"]
        results = [m for m in user_turns if any(isinstance(b, ToolResultBlock) for b in m.content)]
        assert len(results) == 1
        assert [type(b).__name__ for b in results[0].content] == ["ToolResultBlock", "TextBlock"]

    async def test_no_two_user_turns_ever_land_in_a_row(self) -> None:
        history = await self._stored()

        roles = [m.role for m in history]
        assert all(a != b or a != "user" for a, b in zip(roles, roles[1:], strict=False))
