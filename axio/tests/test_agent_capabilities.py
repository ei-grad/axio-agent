"""Tests for Agent capability-aware behavior: tool filtering based on model capabilities."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest

from axio.agent import PATCH_LINE_FRAMING_INSTRUCTION, Agent
from axio.blocks import ToolResultBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.events import StreamEvent, ToolResult
from axio.messages import Message
from axio.models import Capability, ModelSpec
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.tool import Tool


async def msg_handler(msg: str) -> str:
    return json.dumps({"msg": msg})


class _ModelTransport(StubTransport):
    model: ModelSpec
    tool_argument_codec: str | None = None

    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        super().__init__(responses)
        self.tools_received: list[list[Tool[Any]]] = []
        self.systems_received: list[str] = []
        self.histories_received: list[list[Message]] = []

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        self.tools_received.append(tools)
        self.systems_received.append(system)
        self.histories_received.append(messages)
        return super().stream(messages, tools, system)


def _make_transport_with_model(
    responses: list[list[StreamEvent]],
    capabilities: Iterable[Capability],
) -> _ModelTransport:
    """Create a StubTransport with a model attribute that has given capabilities."""
    transport = _ModelTransport(responses)
    transport.model = ModelSpec(id="test-model", capabilities=frozenset(capabilities))
    return transport


class TestToolFiltering:
    async def test_tools_passed_when_model_has_tool_use(self) -> None:
        """When model has tool_use capability, tools are dispatched normally."""
        tool: Tool[object] = Tool(name="echo", description="echo", handler=msg_handler)
        transport = _make_transport_with_model(
            [make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")],
            capabilities=[Capability.text, Capability.tool_use],
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error

    async def test_tools_empty_when_model_lacks_tool_use(self) -> None:
        """When model lacks tool_use capability, no tools are passed to transport."""
        tool: Tool[object] = Tool(name="echo", description="echo", handler=msg_handler)
        transport = _make_transport_with_model(
            [make_text_response("I cannot use tools")],
            capabilities=[Capability.text, Capability.vision],
        )

        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        assert len(transport.tools_received) == 1
        assert transport.tools_received[0] == []

    async def test_tools_passed_when_transport_has_no_model(self) -> None:
        """When transport has no model attribute, tools are passed as-is (backward compat)."""
        tool: Tool[object] = Tool(name="echo", description="echo", handler=msg_handler)
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        # StubTransport has no .model attribute by default
        assert not hasattr(transport, "model")

        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error

    async def test_empty_capabilities_filters_tools(self) -> None:
        """When model declares empty capabilities, tools are filtered out."""
        tool: Tool[object] = Tool(name="echo", description="echo", handler=msg_handler)
        transport = _make_transport_with_model(
            [make_text_response("No tools available")],
            capabilities=[],
        )

        agent = Agent(system="test", tools=[tool], transport=transport)
        async for _ in agent.run_stream("go", MemoryContextStore()):
            pass

        assert len(transport.tools_received) >= 1
        assert all(t == [] for t in transport.tools_received)


class TestPatchLineFraming:
    async def test_auto_only_advertises_legacy_framing_without_transport_codec(self) -> None:
        tool: Tool[object] = Tool(name="patch_file", description="Patch a file.", handler=msg_handler)
        legacy = _make_transport_with_model([make_text_response("legacy")], [Capability.text, Capability.tool_use])
        protected = _make_transport_with_model(
            [make_text_response("protected")],
            [Capability.text, Capability.tool_use],
        )
        protected.tool_argument_codec = "axio.verbatim.v1"

        await Agent(system="base", tools=[tool], transport=legacy).run("go", MemoryContextStore())
        await Agent(system="base", tools=[tool], transport=protected).run("go", MemoryContextStore())

        assert PATCH_LINE_FRAMING_INSTRUCTION in legacy.systems_received[0]
        assert PATCH_LINE_FRAMING_INSTRUCTION not in protected.systems_received[0]

    async def test_on_and_off_are_isolated_across_concurrent_agents(self) -> None:
        shared_tool: Tool[object] = Tool(name="patch_file", description="Patch a file.", handler=msg_handler)
        on_transport = _make_transport_with_model([make_text_response("on")], [Capability.text, Capability.tool_use])
        off_transport = _make_transport_with_model([make_text_response("off")], [Capability.text, Capability.tool_use])
        on_agent = Agent(
            system="base",
            tools=[shared_tool],
            transport=on_transport,
            patch_line_framing="on",
        )
        off_agent = Agent(
            system="base",
            tools=[shared_tool],
            transport=off_transport,
            patch_line_framing="off",
        )

        await asyncio.gather(
            on_agent.run("go", MemoryContextStore()),
            off_agent.run("go", MemoryContextStore()),
        )

        assert PATCH_LINE_FRAMING_INSTRUCTION in on_transport.systems_received[0]
        assert PATCH_LINE_FRAMING_INSTRUCTION not in off_transport.systems_received[0]
        assert on_agent.system == off_agent.system == "base"
        assert shared_tool.description == "Patch a file."

    @pytest.mark.parametrize(
        ("initial_codec", "next_codec", "expected_visibility"),
        [
            ("axio.verbatim.v1", None, [False, True]),
            (None, "axio.verbatim.v1", [True, False]),
        ],
    )
    async def test_auto_recomputes_at_provider_boundaries_with_one_persisted_context(
        self,
        initial_codec: str | None,
        next_codec: str | None,
        expected_visibility: list[bool],
    ) -> None:
        transport = _make_transport_with_model(
            [make_tool_use_response("patch_file", "c1", {"msg": "hi"}), make_text_response("Done")],
            [Capability.text, Capability.tool_use],
        )
        transport.tool_argument_codec = initial_codec
        context = MemoryContextStore()
        tool: Tool[object] = Tool(name="patch_file", description="Patch a file.", handler=msg_handler)

        async def switch_at_boundary() -> None:
            transport.tool_argument_codec = next_codec

        result = await Agent(
            system="base",
            tools=[tool],
            transport=transport,
            before_next_provider_request=switch_at_boundary,
        ).run("go", context)

        assert result == "Done"
        assert [PATCH_LINE_FRAMING_INSTRUCTION in system for system in transport.systems_received] == (
            expected_visibility
        )
        assert len(transport.histories_received) == 2
        second_history = transport.histories_received[1]
        persisted_call = next(
            block for message in second_history for block in message.content if isinstance(block, ToolUseBlock)
        )
        assert persisted_call.input == {"msg": "hi"}
        call_message_index = next(
            index
            for index, message in enumerate(second_history)
            if any(isinstance(block, ToolUseBlock) for block in message.content)
        )
        result_message = second_history[call_message_index + 1]
        assert result_message.role == "user"
        assert any(
            isinstance(block, ToolResultBlock) and block.tool_use_id == persisted_call.id
            for block in result_message.content
        )

    async def test_instruction_is_omitted_without_an_active_patch_tool(self) -> None:
        transport = _make_transport_with_model([make_text_response("done")], [Capability.text, Capability.tool_use])

        await Agent(system="base", tools=[], transport=transport, patch_line_framing="on").run(
            "go", MemoryContextStore()
        )

        assert transport.systems_received == ["base"]

    def test_invalid_mode_is_rejected(self) -> None:
        transport = _make_transport_with_model([make_text_response("done")], [Capability.text])

        with pytest.raises(ValueError, match="patch_line_framing"):
            Agent(system="base", tools=[], transport=transport, patch_line_framing="sometimes")  # type: ignore[arg-type]
