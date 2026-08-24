"""Provider-response circuit breaker tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from axio.agent import Agent
from axio.blocks import TextBlock
from axio.context import MemoryContextStore
from axio.events import (
    Error,
    IterationEnd,
    ReasoningDelta,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolOutputDelta,
    ToolUseStart,
)
from axio.exceptions import ProviderOutputLimitError
from axio.messages import Message
from axio.models import Capability, ModelSpec
from axio.provider_output import ProviderOutputGuard, ProviderOutputPolicy
from axio.tool import Tool
from axio.types import StopReason, Usage


def _nonrepetitive_code(start: int, minimum_chars: int) -> str:
    parts: list[str] = []
    size = 0
    line = start
    while size < minimum_chars:
        value = f"const value_{line} = {line * 7919}; // строка {line}\n"
        parts.append(value)
        size += len(value)
        line += 1
    return "".join(parts)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_guard_counts_model_wire_text_but_not_tool_execution_output() -> None:
    policy = ProviderOutputPolicy(max_response_bytes=10, sustained_rate_bytes_per_second=None)
    guard = ProviderOutputGuard(policy, effective_output_tokens=None)

    assert guard.inspect(TextDelta(index=0, delta="аб")) is None  # four UTF-8 bytes
    assert guard.inspect(ReasoningDelta(index=0, delta="x")) is None
    assert guard.inspect(ToolInputDelta(index=0, tool_use_id="call", partial_json='{"a":')) is None
    assert guard.accepted_bytes == 10
    assert guard.inspect(ToolOutputDelta(tool_use_id="call", name="shell", key="stdout", delta="x" * 10_000)) is None
    assert guard.accepted_bytes == 10
    assert isinstance(guard.inspect(TextDelta(index=0, delta="!")), ProviderOutputLimitError)


def test_guard_detects_growing_snapshots_only_within_one_semantic_stream() -> None:
    policy = ProviderOutputPolicy(
        max_response_bytes=None,
        sustained_rate_bytes_per_second=None,
        cumulative_snapshot_min_prefix_chars=1024,
    )
    guard = ProviderOutputGuard(policy, effective_output_tokens=None)
    base = _nonrepetitive_code(0, 3_000)
    second = base + _nonrepetitive_code(10_000, 500)

    assert guard.inspect(TextDelta(index=0, delta=base)) is None
    assert guard.inspect(TextDelta(index=0, delta=second)) is None
    assert guard.inspect(TextDelta(index=1, delta=second + "independent index")) is None
    assert guard.inspect(ReasoningDelta(index=0, delta=second + "independent reasoning")) is None
    assert guard.inspect(ToolInputDelta(index=0, tool_use_id="one", partial_json=second)) is None
    assert guard.inspect(ToolInputDelta(index=0, tool_use_id="two", partial_json=second + "other call")) is None
    assert guard.inspect(TextDelta(index=2, delta=base)) is None
    assert guard.inspect(TextDelta(index=2, delta=base)) is None  # repeated prose is not a growing snapshot

    violation = guard.inspect(TextDelta(index=0, delta=second + _nonrepetitive_code(20_000, 500)))

    assert isinstance(violation, ProviderOutputLimitError)
    assert "cumulative snapshot" in str(violation)


def test_guard_rate_limit_has_a_buffered_first_frame_allowance() -> None:
    clock = _Clock()
    policy = ProviderOutputPolicy(
        max_response_bytes=None,
        sustained_rate_bytes_per_second=100,
        rate_burst_bytes=100,
        cumulative_snapshot_min_prefix_chars=10_000,
    )
    guard = ProviderOutputGuard(policy, effective_output_tokens=None, clock=clock)

    assert guard.inspect(TextDelta(index=0, delta="a" * 150)) is None
    assert isinstance(guard.inspect(TextDelta(index=1, delta="b")), ProviderOutputLimitError)
    clock.now = 1.0
    assert guard.inspect(TextDelta(index=1, delta="b" * 100)) is None


def test_token_envelope_is_conservative_for_code_cyrillic_and_whitespace() -> None:
    policy = ProviderOutputPolicy(max_response_bytes=None, sustained_rate_bytes_per_second=None)
    guard = ProviderOutputGuard(policy, effective_output_tokens=512)
    generated_file = _nonrepetitive_code(0, 3_000) + "Привет, мир\n" + " " * 12_000

    assert len(generated_file.encode()) > 15_000
    assert guard.inspect(ToolInputDelta(index=0, tool_use_id="write", partial_json=generated_file)) is None
    violation = guard.inspect(ToolInputDelta(index=1, tool_use_id="other", partial_json="z" * 20_000))
    assert isinstance(violation, ProviderOutputLimitError)
    assert "512 output tokens" in str(violation)


class _UnterminatedTransport:
    def __init__(self, events: list[StreamEvent], *, requested_limit: int | None) -> None:
        self.events = events
        self.requested_limit = requested_limit
        self.model = ModelSpec(id="test", max_output_tokens=8_192)
        self.limit_snapshots = 0
        self.closed = False

    def output_token_limit(
        self,
        messages: list[Message],
        tools: list[Tool[Any]],
        system: str,
    ) -> int | None:
        del messages, tools, system
        self.limit_snapshots += 1
        return self.requested_limit

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[Any]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools, system
        try:
            for index, event in enumerate(self.events):
                if index == 1:
                    self.model = ModelSpec(id="mutated", max_output_tokens=1_000_000)
                yield event
        finally:
            self.closed = True


async def test_agent_snapshots_exact_request_limit_and_rejects_before_yield_or_persistence() -> None:
    first = _nonrepetitive_code(0, 9_000)
    rejected = _nonrepetitive_code(50_000, 10_000)
    transport = _UnterminatedTransport(
        [TextDelta(index=0, delta=first), TextDelta(index=0, delta=rejected)],
        requested_limit=64,
    )
    policy = ProviderOutputPolicy(max_response_bytes=None, sustained_rate_bytes_per_second=None)
    context = MemoryContextStore()

    agent = Agent("test", transport, provider_output_policy=policy)
    events: list[StreamEvent] = []
    async for event in agent.run_stream("go", context):
        if isinstance(event, TextDelta) and "Output truncated" in event.delta:
            assert transport.closed
        events.append(event)

    assert transport.limit_snapshots == 1
    assert transport.closed
    assert [event.delta for event in events if isinstance(event, TextDelta)][0] == first
    assert rejected not in "".join(event.delta for event in events if isinstance(event, TextDelta))
    error = next(event.exception for event in events if isinstance(event, Error))
    assert isinstance(error, ProviderOutputLimitError)
    assert "64 output tokens" in str(error)
    assert isinstance(events[-1], SessionEndEvent)
    assert events[-1].total_usage == Usage(0, 0)
    assistant = (await context.get_history())[-1]
    assert isinstance(assistant.content[0], TextBlock)
    assert rejected not in assistant.content[0].text


@pytest.mark.parametrize(
    "provider_event",
    [
        TextDelta(index=0, delta="G" * 40_000),
        ReasoningDelta(index=0, delta="R" * 40_000),
        ToolInputDelta(index=0, tool_use_id="call", partial_json="J" * 40_000),
    ],
)
async def test_agent_rejects_giant_buffered_wire_frame_and_closes_stream(provider_event: StreamEvent) -> None:
    prefix: list[StreamEvent] = []
    if isinstance(provider_event, ToolInputDelta):
        prefix.append(ToolUseStart(index=0, tool_use_id="call", name="write_file"))
    transport = _UnterminatedTransport(
        [*prefix, provider_event, IterationEnd(1, StopReason.end_turn, Usage(1, 1))],
        requested_limit=None,
    )
    policy = ProviderOutputPolicy(max_response_bytes=32 * 1024, sustained_rate_bytes_per_second=None)
    context = MemoryContextStore()

    agent = Agent("test", transport, provider_output_policy=policy)
    events = [event async for event in agent.run_stream("go", context)]

    assert transport.closed
    assert provider_event not in events
    assert not any(isinstance(event, IterationEnd) for event in events)
    assert any(isinstance(event, Error) for event in events)
    assert isinstance(events[-1], SessionEndEvent)
    assert events[-1].total_usage == Usage(0, 0)
    assistant = (await context.get_history())[-1]
    assert sum(len(block.text) for block in assistant.content if isinstance(block, TextBlock)) < 200


async def test_agent_stops_cumulative_snapshots_well_before_one_megabyte_without_rate_limit() -> None:
    first = _nonrepetitive_code(0, 8_000)
    second = first + _nonrepetitive_code(10_000, 2_000)
    third = second + _nonrepetitive_code(20_000, 2_000)
    transport = _UnterminatedTransport(
        [TextDelta(0, first), TextDelta(0, second), TextDelta(0, third)],
        requested_limit=None,
    )
    policy = ProviderOutputPolicy(sustained_rate_bytes_per_second=None)
    context = MemoryContextStore()

    agent = Agent("test", transport, provider_output_policy=policy)
    events = [event async for event in agent.run_stream("go", context)]

    rendered = "".join(event.delta for event in events if isinstance(event, TextDelta))
    assert transport.closed
    assert third not in rendered
    assert len(rendered.encode()) < 64 * 1024
    assert "cumulative response snapshots" in rendered


async def test_agent_stops_impossibly_fast_nonrepetitive_output() -> None:
    first = _nonrepetitive_code(0, 150)
    rejected = _nonrepetitive_code(10_000, 150)
    transport = _UnterminatedTransport(
        [TextDelta(0, first), TextDelta(1, rejected)],
        requested_limit=None,
    )
    policy = ProviderOutputPolicy(
        max_response_bytes=None,
        sustained_rate_bytes_per_second=1,
        rate_burst_bytes=256,
        cumulative_snapshot_min_prefix_chars=10_000,
    )
    agent = Agent("test", transport, provider_output_policy=policy)

    events = [event async for event in agent.run_stream("go", MemoryContextStore())]

    assert transport.closed
    assert rejected not in "".join(event.delta for event in events if isinstance(event, TextDelta))
    error = next(event.exception for event in events if isinstance(event, Error))
    assert isinstance(error, ProviderOutputLimitError)
    assert "sustained decoded-byte rate" in str(error)


class _GeneratedFileTransport:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.model = ModelSpec(
            id="test",
            max_output_tokens=8_192,
            capabilities=frozenset({Capability.text, Capability.tool_use}),
        )

    def output_token_limit(
        self,
        messages: list[Message],
        tools: list[Tool[Any]],
        system: str,
    ) -> int:
        del messages, tools, system
        return 512

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[Any]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools, system
        self.calls += 1
        if self.calls == 1:
            yield ToolUseStart(index=0, tool_use_id="write", name="write_file")
            yield ToolInputDelta(
                index=0,
                tool_use_id="write",
                partial_json=json.dumps({"content": self.content}, ensure_ascii=False),
            )
            yield IterationEnd(1, StopReason.tool_use, Usage(10, 400))
            return
        yield TextDelta(index=0, delta="done")
        yield IterationEnd(2, StopReason.end_turn, Usage(20, 1))


async def test_normal_fifteen_kilobyte_generated_file_passes() -> None:
    generated_file = _nonrepetitive_code(0, 3_000) + "Привет\n" + " " * 12_000
    captured: list[str] = []

    async def write_file(content: str) -> str:
        captured.append(content)
        return "written"

    transport = _GeneratedFileTransport(generated_file)
    agent = Agent("test", transport, tools=[Tool(name="write_file", handler=write_file)])

    result = await agent.run("generate", MemoryContextStore())

    assert result == "done"
    assert captured == [generated_file]
