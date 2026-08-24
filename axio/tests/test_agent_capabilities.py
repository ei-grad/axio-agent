"""Tests for Agent capability-aware behavior: tool filtering based on model capabilities."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest

from axio.agent import Agent
from axio.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.diff import PATCH_LINE_FRAMING_INSTRUCTION, patch_protocol_transition, prepare_patch_input
from axio.events import Error, IterationEnd, StreamEvent, ToolInputDelta, ToolResult, ToolUseStart
from axio.exceptions import ToolProtocolError
from axio.messages import INPUT_PROVENANCE_SYSTEM_INSTRUCTION, InputProvenance, Message
from axio.models import Capability, ModelSpec
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.tool import Tool
from axio.tool_codec import encode_tool_arguments
from axio.types import StopReason, Usage


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
    @staticmethod
    def _tool() -> Tool[object]:
        return Tool(
            name="patch_file",
            description="Patch a file.",
            handler=msg_handler,
            input_preparer=prepare_patch_input,
            protocol_transition=patch_protocol_transition,
        )

    @staticmethod
    def _protocol_messages(history: list[Message]) -> list[Message]:
        return [
            message
            for message in history
            if message.provenance is not None and message.provenance.authority == "tool-protocol"
        ]

    @classmethod
    def _protocol_states(cls, history: list[Message]) -> list[str]:
        states: list[str] = []
        for message in cls._protocol_messages(history):
            provenance = message.provenance
            assert provenance is not None
            assert provenance.protocol_state is not None
            states.append(provenance.protocol_state)
        return states

    async def test_auto_only_adds_protocol_message_without_transport_codec(self) -> None:
        tool = self._tool()
        legacy = _make_transport_with_model([make_text_response("legacy")], [Capability.text, Capability.tool_use])
        protected = _make_transport_with_model(
            [make_text_response("protected")],
            [Capability.text, Capability.tool_use],
        )
        protected.tool_argument_codec = "axio.verbatim.v1"

        await Agent(system="base", tools=[tool], transport=legacy).run("go", MemoryContextStore())
        await Agent(system="base", tools=[tool], transport=protected).run("go", MemoryContextStore())

        legacy_messages = self._protocol_messages(legacy.histories_received[0])
        assert len(legacy_messages) == 1
        block = legacy_messages[0].content[0]
        assert isinstance(block, TextBlock)
        assert PATCH_LINE_FRAMING_INSTRUCTION in block.text
        assert self._protocol_messages(protected.histories_received[0]) == []
        assert protected.systems_received == ["base"]
        assert legacy.systems_received == [f"base\n\nInput provenance:\n- {INPUT_PROVENANCE_SYSTEM_INSTRUCTION}"]

    async def test_switch_after_notice_without_a_patch_call_appends_transition(self) -> None:
        transport = _make_transport_with_model(
            [make_text_response("first"), make_text_response("second")],
            [Capability.text, Capability.tool_use],
        )
        context = MemoryContextStore()
        agent = Agent(system="base", tools=[self._tool()], transport=transport)

        assert await agent.run("first request", context) == "first"
        transport.tool_argument_codec = "axio.verbatim.v1"
        assert await agent.run("second request", context) == "second"

        assert self._protocol_states(transport.histories_received[0]) == ["patch-file:line-framed:prior="]
        assert self._protocol_states(transport.histories_received[1]) == [
            "patch-file:line-framed:prior=",
            "patch-file:literal:prior=line-framed",
        ]
        assert not any(
            isinstance(block, ToolUseBlock) for message in transport.histories_received[1] for block in message.content
        )

    async def test_codec_change_during_protocol_append_is_corrected_before_provider(self) -> None:
        transport = _make_transport_with_model(
            [make_text_response("done")],
            [Capability.text, Capability.tool_use],
        )

        class SwitchingContext(MemoryContextStore):
            switched = False

            async def append_many(self, messages: list[Message]) -> None:
                await super().append_many(messages)
                if not self.switched and self._contains_protocol_message(messages):
                    self.switched = True
                    transport.tool_argument_codec = "axio.verbatim.v1"

            @staticmethod
            def _contains_protocol_message(messages: list[Message]) -> bool:
                return any(
                    message.provenance is not None and message.provenance.authority == "tool-protocol"
                    for message in messages
                )

        context = SwitchingContext()
        agent = Agent(system="base", tools=[self._tool()], transport=transport)

        assert await agent.run("go", context) == "done"

        assert context.switched
        assert len(transport.histories_received) == 1
        assert self._protocol_states(transport.histories_received[0]) == [
            "patch-file:line-framed:prior=",
            "patch-file:literal:prior=line-framed",
        ]
        final_notice = transport.histories_received[0][-1].content[0]
        assert isinstance(final_notice, TextBlock)
        assert "│ has no framing meaning" in final_notice.text

    async def test_unstable_protocol_configuration_fails_before_provider(self) -> None:
        transport = _make_transport_with_model(
            [make_text_response("unused")],
            [Capability.text, Capability.tool_use],
        )

        class OscillatingContext(MemoryContextStore):
            async def append_many(self, messages: list[Message]) -> None:
                await super().append_many(messages)
                if any(
                    message.provenance is not None and message.provenance.authority == "tool-protocol"
                    for message in messages
                ):
                    transport.tool_argument_codec = (
                        None if transport.tool_argument_codec is not None else "axio.verbatim.v1"
                    )

        events = [
            event
            async for event in Agent(system="base", tools=[self._tool()], transport=transport).run_stream(
                "go", OscillatingContext()
            )
        ]

        assert transport.histories_received == []
        errors = [event.exception for event in events if isinstance(event, Error)]
        assert len(errors) == 1
        assert isinstance(errors[0], ToolProtocolError)
        assert "did not stabilize" in str(errors[0])

    async def test_on_and_off_are_isolated_across_concurrent_agents(self) -> None:
        shared_tool = self._tool()
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

        assert len(self._protocol_messages(on_transport.histories_received[0])) == 1
        assert self._protocol_messages(off_transport.histories_received[0]) == []
        assert on_agent.system == off_agent.system == "base"
        assert INPUT_PROVENANCE_SYSTEM_INSTRUCTION in on_transport.systems_received[0]
        assert off_transport.systems_received == ["base"]
        assert shared_tool.description == "Patch a file."

    @pytest.mark.parametrize(
        ("initial_codec", "next_codec", "expected_states"),
        [
            ("axio.verbatim.v1", None, [[], ["patch-file:line-framed:prior=literal"]]),
            (
                None,
                "axio.verbatim.v1",
                [
                    ["patch-file:line-framed:prior="],
                    ["patch-file:line-framed:prior=", "patch-file:literal:prior=line-framed"],
                ],
            ),
        ],
    )
    async def test_auto_recomputes_at_provider_boundaries_with_one_persisted_context(
        self,
        initial_codec: str | None,
        next_codec: str | None,
        expected_states: list[list[str]],
    ) -> None:
        tool = self._tool()
        canonical_input = {"msg": "hi"}
        wire_input = (
            encode_tool_arguments(canonical_input, tool.input_schema, initial_codec)
            if initial_codec is not None
            else canonical_input
        )
        first_response: list[StreamEvent] = [
            ToolUseStart(
                index=0,
                tool_use_id="c1",
                name="patch_file",
                argument_codec=initial_codec,
            ),
            ToolInputDelta(index=0, tool_use_id="c1", partial_json=json.dumps(wire_input)),
            IterationEnd(iteration=1, stop_reason=StopReason.tool_use, usage=Usage(0, 0)),
        ]
        transport = _make_transport_with_model(
            [first_response, make_text_response("Done")],
            [Capability.text, Capability.tool_use],
        )
        transport.tool_argument_codec = initial_codec
        context = MemoryContextStore()

        async def switch_at_boundary() -> None:
            transport.tool_argument_codec = next_codec

        result = await Agent(
            system="base",
            tools=[tool],
            transport=transport,
            before_next_provider_request=switch_at_boundary,
        ).run("go", context)

        assert result == "Done"
        assert [self._protocol_states(history) for history in transport.histories_received] == expected_states
        assert len(transport.histories_received) == 2
        second_history = transport.histories_received[1]
        persisted_call = next(
            block for message in second_history for block in message.content if isinstance(block, ToolUseBlock)
        )
        assert persisted_call.input == {"msg": "hi"}
        assert persisted_call.input_preparation == ("literal" if initial_codec is not None else "line-framed")
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

    @pytest.mark.parametrize(
        ("initial_policy", "next_policy", "expected_states", "expected_preparation"),
        [
            (
                "on",
                "off",
                [
                    ["patch-file:line-framed:prior="],
                    ["patch-file:line-framed:prior=", "patch-file:literal:prior=line-framed"],
                ],
                "line-framed",
            ),
            ("off", "on", [[], ["patch-file:line-framed:prior=literal"]], "literal"),
        ],
    )
    async def test_policy_switches_at_provider_boundaries_with_one_persisted_context(
        self,
        initial_policy: str,
        next_policy: str,
        expected_states: list[list[str]],
        expected_preparation: str,
    ) -> None:
        transport = _make_transport_with_model(
            [make_tool_use_response("patch_file", "c1", {"msg": "hi"}), make_text_response("Done")],
            [Capability.text, Capability.tool_use],
        )
        context = MemoryContextStore()
        agent = Agent(
            system="base",
            tools=[self._tool()],
            transport=transport,
            patch_line_framing=initial_policy,  # type: ignore[arg-type]
        )

        async def switch_at_boundary() -> None:
            agent.patch_line_framing = next_policy  # type: ignore[assignment]

        agent.before_next_provider_request = switch_at_boundary
        assert await agent.run("go", context) == "Done"

        assert [self._protocol_states(history) for history in transport.histories_received] == expected_states
        second_history = transport.histories_received[1]
        persisted_call = next(
            block for message in second_history for block in message.content if isinstance(block, ToolUseBlock)
        )
        assert persisted_call.input_preparation == expected_preparation

    async def test_opaque_legacy_history_is_preserved_and_explained(self) -> None:
        transport = _make_transport_with_model([make_text_response("Done")], [Capability.text, Capability.tool_use])
        transport.tool_argument_codec = "axio.verbatim.v1"
        context = MemoryContextStore()
        legacy_input = {"content": "│literal\n│      still ambiguous"}
        legacy_call = ToolUseBlock(
            id="old",
            name="patch_file",
            input=legacy_input,
            input_preparation=None,
        )
        await context.append_many(
            [
                Message(role="assistant", content=[legacy_call]),
                Message(role="user", content=[ToolResultBlock(tool_use_id="old", content="old result")]),
            ]
        )

        await Agent(system="base", tools=[self._tool()], transport=transport).run("go", context)

        provider_call = transport.histories_received[0]
        old = next(
            block
            for message in provider_call
            for block in message.content
            if isinstance(block, ToolUseBlock) and block.id == "old"
        )
        assert old.input == legacy_input
        assert old.input_preparation is None
        [notice] = self._protocol_messages(provider_call)
        assert notice.provenance == InputProvenance(
            human_authored=False,
            source="tool-hook",
            author="patch_file",
            authority="tool-protocol",
            protocol_state="patch-file:literal:prior=opaque",
        )
        assert isinstance(notice.content[0], TextBlock)
        assert "historical records" in notice.content[0].text
        persisted = await context.get_history()
        persisted_old = next(
            block
            for message in persisted
            for block in message.content
            if isinstance(block, ToolUseBlock) and block.id == "old"
        )
        assert persisted_old == legacy_call

    async def test_protocol_hook_failure_stops_before_provider_request(self) -> None:
        def broken(context: object) -> object:
            raise RuntimeError("broken transition")

        tool: Tool[object] = Tool(
            name="patch_file",
            handler=msg_handler,
            protocol_transition=broken,  # type: ignore[arg-type]
        )
        transport = _make_transport_with_model([make_text_response("unused")], [Capability.text, Capability.tool_use])

        events = [
            event
            async for event in Agent(system="base", tools=[tool], transport=transport).run_stream(
                "go", MemoryContextStore()
            )
        ]

        assert transport.histories_received == []
        assert any("broken transition" in str(event.exception) for event in events if hasattr(event, "exception"))

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
