import json
from typing import Any

import pytest
from axio.agent import Agent
from axio.blocks import AudioBlock, ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.events import IterationEnd
from axio.exceptions import StreamError
from axio.messages import (
    INPUT_PROVENANCE_FOOTER,
    UNATTRIBUTED_INPUT_PROVENANCE,
    InputProvenance,
    Message,
    input_provenance_header,
)
from axio.models import Capability, ModelSpec
from axio.tool import Tool
from axio.types import StopReason

from axio_transport_openai import OpenAITransport
from axio_transport_openai.responses import _convert_messages

_REASONING = ModelSpec(
    id="gpt-5.6",
    capabilities=frozenset({Capability.text, Capability.reasoning, Capability.tool_use}),
    max_output_tokens=128_000,
    context_window=1_050_000,
)


async def _echo(text: str) -> str:
    return text


def _transport(**kwargs: Any) -> OpenAITransport:
    return OpenAITransport(model=_REASONING, **kwargs)


def _unattributed_input_text(text: str) -> list[dict[str, str]]:
    return [
        {"type": "input_text", "text": input_provenance_header(UNATTRIBUTED_INPUT_PROVENANCE)},
        {"type": "input_text", "text": text},
        {"type": "input_text", "text": INPUT_PROVENANCE_FOOTER},
    ]


def test_streams_from_the_responses_endpoint() -> None:
    assert OpenAITransport.stream_path == "responses"


def test_system_prompt_moves_to_instructions() -> None:
    payload = _transport().build_payload([Message(role="user", content=[TextBlock(text="hi")])], [], "be terse")
    assert payload["instructions"] == "be terse"
    assert payload["store"] is False
    assert payload["input"] == [{"role": "user", "content": _unattributed_input_text("hi")}]


def test_system_messages_in_history_are_preserved() -> None:
    items = _convert_messages([Message(role="system", content=[TextBlock(text="historical policy")])])

    assert items == [{"role": "system", "content": [{"type": "input_text", "text": "historical policy"}]}]


def test_provenance_precedes_responses_text_and_image_parts() -> None:
    provenance = InputProvenance(human_authored=False, source="peer", author="agent-1")
    message = Message(
        role="user",
        content=[TextBlock(text="report"), ImageBlock(media_type="image/png", data=b"image")],
        provenance=provenance,
    )

    [item] = _convert_messages([message])

    assert item["content"][0] == {"type": "input_text", "text": input_provenance_header(provenance)}
    assert item["content"][1] == {"type": "input_text", "text": "report"}
    assert item["content"][2]["type"] == "input_image"
    assert item["content"][3] == {"type": "input_text", "text": INPUT_PROVENANCE_FOOTER}


def test_tools_are_flat_not_nested() -> None:
    payload = _transport().build_payload([], [Tool[Any](name="echo", handler=_echo)], "")
    [tool] = payload["tools"]
    # chat/completions nests these under "function"; responses does not.
    assert tool["type"] == "function"
    assert tool["name"] == "echo"
    assert "parameters" in tool


def test_reasoning_effort_is_sent_only_when_supported() -> None:
    assert _transport().build_payload([], [], "")["reasoning"] == {"summary": "auto"}
    transport = _transport()
    state = transport.configure_effort("high")
    payload = transport.build_payload([], [], "")
    assert payload["reasoning"] == {"summary": "auto", "effort": "high"}
    assert state.mechanism.value == "native-effort"

    plain = ModelSpec(id="gpt-4o", capabilities=frozenset({Capability.text, Capability.tool_use}))
    payload = OpenAITransport(model=plain, reasoning_effort="high").build_payload([], [], "")
    assert "reasoning" not in payload


def test_native_reasoning_effort_requires_an_exact_supported_level() -> None:
    model = ModelSpec(
        id="gpt-5.5-pro",
        capabilities=frozenset({Capability.text, Capability.reasoning}),
    )
    transport = OpenAITransport(model=model)

    state = transport.configure_effort("default")

    assert state.allowed == ("medium", "high", "xhigh")
    with pytest.raises(ValueError, match="low.*not supported"):
        transport.configure_effort("low")
    assert transport.reasoning_effort is None


def test_tool_calls_and_results_become_top_level_items() -> None:
    messages = [
        Message(role="assistant", content=[ToolUseBlock(id="call-1", name="echo", input={"text": "x"})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call-1", content="x")]),
    ]
    items = _convert_messages(messages)
    assert items[0]["type"] == "function_call"
    assert items[0]["call_id"] == "call-1"
    assert json.loads(items[0]["arguments"]) == {"text": "x"}
    assert items[1] == {"type": "function_call_output", "call_id": "call-1", "output": "x"}


def test_tool_result_mixed_with_text_emits_both_in_order() -> None:
    messages = [
        Message(role="assistant", content=[ToolUseBlock(id="call-1", name="echo", input={"text": "x"})]),
        Message(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="call-1", content="x"),
                TextBlock(text="and then?"),
            ],
        ),
    ]
    items = _convert_messages(messages)
    assert items[0]["type"] == "function_call"
    assert items[1] == {"type": "function_call_output", "call_id": "call-1", "output": "x"}
    assert items[2]["role"] == "user"
    assert items[2]["content"] == _unattributed_input_text("and then?")


def test_tool_result_images_are_forwarded_as_user_input() -> None:
    provenance = InputProvenance(human_authored=False, source="tool-result", author="image")
    messages = [
        Message(role="assistant", content=[ToolUseBlock(id="call-1", name="image", input={})]),
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id="call-1",
                    content=[TextBlock(text="diagram"), ImageBlock(media_type="image/png", data=b"png")],
                )
            ],
            provenance=provenance,
        ),
    ]

    items = _convert_messages(messages)

    assert items[1] == {"type": "function_call_output", "call_id": "call-1", "output": "diagram"}
    assert items[2]["role"] == "user"
    assert items[2]["content"][0] == {
        "type": "input_text",
        "text": input_provenance_header(provenance),
    }
    assert items[2]["content"][2]["image_url"].startswith("data:image/png;base64,")
    assert items[2]["content"][3] == {"type": "input_text", "text": INPUT_PROVENANCE_FOOTER}


def test_unsupported_media_is_rejected_explicitly() -> None:
    message = Message(role="user", content=[AudioBlock(media_type="audio/wav", data=b"audio")])

    with pytest.raises(ValueError, match="does not support input media: audio/wav"):
        _convert_messages([message])


def test_unsupported_assistant_media_is_not_dropped() -> None:
    message = Message(role="assistant", content=[AudioBlock(media_type="audio/wav", data=b"audio")])

    with pytest.raises(ValueError, match="does not support media in assistant history: audio/wav"):
        _convert_messages([message])


def test_tool_result_message_then_separate_user_text_message() -> None:
    """A follow-up user message (e.g. an injected notification) arriving as its own
    Message right after the tool-results message must still emit tool output first,
    then the user text as its own item, in order."""
    messages = [
        Message(role="assistant", content=[ToolUseBlock(id="call-1", name="echo", input={"text": "x"})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call-1", content="x")]),
        Message(role="user", content=[TextBlock(text="Notification: background task finished")]),
    ]
    items = _convert_messages(messages)
    assert items[0]["type"] == "function_call"
    assert items[1] == {"type": "function_call_output", "call_id": "call-1", "output": "x"}
    assert items[2]["role"] == "user"
    assert items[2]["content"] == _unattributed_input_text("Notification: background task finished")


def test_unanswered_call_gets_a_placeholder_output() -> None:
    # The API rejects the whole request when a call has no output, which is what
    # an interrupted turn leaves behind.
    items = _convert_messages([Message(role="assistant", content=[ToolUseBlock(id="orphan", name="echo", input={})])])
    assert [i["type"] for i in items] == ["function_call", "function_call_output"]
    assert "not executed" in items[1]["output"]


class _FakeContent:
    def __init__(self, events: list[dict[str, Any]], *, trailing_newline: bool = True) -> None:
        separator = "\n\n"
        body = separator.join(f"data: {json.dumps(e)}" for e in events)
        if trailing_newline:
            body += separator
        self._body = body.encode()

    async def iter_any(self):  # type: ignore[no-untyped-def]
        yield self._body


class _FakeResponse:
    def __init__(self, events: list[dict[str, Any]], *, trailing_newline: bool = True) -> None:
        self.status = 200
        self.content = _FakeContent(events, trailing_newline=trailing_newline)


class _ResponseContext:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    async def __aenter__(self) -> _FakeResponse:
        return self.response

    async def __aexit__(self, *_args: Any) -> None:
        return None


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []
        self.paths: list[str] = []

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> _ResponseContext:
        del headers
        self.paths.append(url)
        self.payloads.append(json)
        return _ResponseContext(self.responses.pop(0))


async def _collect(events: list[dict[str, Any]]) -> list[Any]:
    return [e async for e in _transport()._parse_sse(_FakeResponse(events))]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_text_and_reasoning_deltas_are_separated() -> None:
    events = await _collect(
        [
            {"type": "response.reasoning_text.delta", "delta": "thinking"},
            {"type": "response.output_text.delta", "delta": "hello"},
            {
                "type": "response.completed",
                "response": {"status": "completed", "usage": {"input_tokens": 3, "output_tokens": 4}},
            },
        ]
    )
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ReasoningDelta", "TextDelta", "IterationEnd"]
    assert events[-1].usage.input_tokens == 3
    assert events[-1].stop_reason is StopReason.end_turn


@pytest.mark.asyncio
async def test_argument_deltas_are_joined_to_the_call_id() -> None:
    events = await _collect(
        [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "id": "item-1", "call_id": "call-1", "name": "echo"},
            },
            {"type": "response.function_call_arguments.delta", "item_id": "item-1", "delta": '{"text"'},
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": [{"type": "function_call"}]},
            },
        ]
    )
    start, delta, end = events
    assert start.tool_use_id == "call-1"
    # Deltas arrive keyed by item id and must be reported under the call id.
    assert delta.tool_use_id == "call-1"
    assert end.stop_reason is StopReason.tool_use


@pytest.mark.asyncio
async def test_terminal_event_without_a_complete_sse_frame_is_rejected() -> None:
    response = _FakeResponse(
        [
            {
                "type": "response.completed",
                "response": {"status": "completed", "usage": {"input_tokens": 1, "output_tokens": 2}},
            }
        ],
        trailing_newline=False,
    )

    with pytest.raises(StreamError, match="without response.completed"):
        [event async for event in _transport()._parse_sse(response)]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_incomplete_response_with_null_usage_is_max_tokens() -> None:
    response = _FakeResponse(
        [
            {
                "type": "response.incomplete",
                "response": {
                    "usage": None,
                    "output": [{"type": "function_call"}],
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            }
        ]
    )

    events = [event async for event in _transport()._parse_sse(response)]  # type: ignore[arg-type]

    assert len(events) == 1
    assert isinstance(events[0], IterationEnd)
    assert events[0].stop_reason is StopReason.max_tokens
    assert events[0].usage.input_tokens == 0
    assert events[0].usage.output_tokens == 0


@pytest.mark.asyncio
async def test_stream_requires_a_terminal_event() -> None:
    response = _FakeResponse([{"type": "response.output_text.delta", "delta": "partial"}])

    with pytest.raises(StreamError, match="without response.completed"):
        [event async for event in _transport()._parse_sse(response)]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_malformed_sse_is_an_error() -> None:
    response = _FakeResponse([])
    response.content._body = b"data: {not json}\n\n"

    with pytest.raises(StreamError, match="Invalid OpenAI Responses SSE payload"):
        [event async for event in _transport()._parse_sse(response)]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_failed_response_event_is_an_error() -> None:
    response = _FakeResponse([{"type": "response.failed", "response": {"error": {"message": "request failed"}}}])

    with pytest.raises(StreamError, match="request failed"):
        [event async for event in _transport()._parse_sse(response)]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_top_level_error_preserves_code_and_message() -> None:
    response = _FakeResponse([{"type": "error", "code": "rate_limit_exceeded", "message": "Please retry later"}])

    with pytest.raises(StreamError, match="rate_limit_exceeded: Please retry later"):
        [event async for event in _transport()._parse_sse(response)]  # type: ignore[arg-type]


def test_output_limit_leaves_room_for_responses_input() -> None:
    model = ModelSpec(id="m", context_window=100_000, max_output_tokens=100_000)
    transport = OpenAITransport(model=model)
    messages = [Message(role="user", content=[TextBlock(text="x" * 60_000)])]

    assert transport.build_payload(messages, [], "")["max_output_tokens"] <= 80_000


def test_output_token_limit_reports_prompt_fitted_request_value() -> None:
    model = ModelSpec(id="m", context_window=100_000, max_output_tokens=100_000)
    transport = OpenAITransport(model=model)
    messages = [Message(role="user", content=[TextBlock(text="x" * 60_000)])]

    assert (
        transport.output_token_limit(messages, [], "")
        == transport.build_payload(messages, [], "")["max_output_tokens"]
    )


def test_output_limit_leaves_room_for_responses_instructions() -> None:
    model = ModelSpec(id="m", context_window=100_000, max_output_tokens=100_000)
    transport = OpenAITransport(model=model)

    assert transport.build_payload([], [], "x" * 60_000)["max_output_tokens"] <= 80_000


def test_extra_params_can_explicitly_replace_responses_fields() -> None:
    transport = _transport(extra_params={"input": []})

    assert transport.build_payload([], [], "")["input"] == []


def test_active_model_round_trips() -> None:
    transport = _transport()

    restored = OpenAITransport.from_dict(transport.to_dict())

    assert restored.model == _REASONING


@pytest.mark.asyncio
async def test_agent_tool_round_trip_uses_responses_items() -> None:
    first = _FakeResponse(
        [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "id": "item-1", "call_id": "call-1", "name": "echo"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "item-1",
                "delta": '{"text":"hello"}',
            },
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": [{"type": "function_call"}]},
            },
        ]
    )
    second = _FakeResponse(
        [
            {"type": "response.output_text.delta", "delta": "done"},
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": [{"type": "message"}]},
            },
        ]
    )
    session = _FakeSession([first, second])
    transport = _transport(session=session)
    agent = Agent(system="system", transport=transport, tools=[Tool[Any](name="echo", handler=_echo)])

    result = await agent.run("go", MemoryContextStore())

    assert result == "done"
    assert session.paths == [
        "https://api.openai.com/v1/responses",
        "https://api.openai.com/v1/responses",
    ]
    second_input = session.payloads[1]["input"]
    assert any(item.get("type") == "function_call" for item in second_input)
    assert any(item.get("type") == "function_call_output" and item["output"] == "hello" for item in second_input)


@pytest.mark.asyncio
async def test_agent_returns_refusal_text() -> None:
    response = _FakeResponse(
        [
            {"type": "response.refusal.delta", "delta": "I cannot help with that."},
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": [{"type": "message"}]},
            },
        ]
    )
    session = _FakeSession([response])
    transport = _transport(session=session)
    agent = Agent(system="system", transport=transport, tools=[])

    result = await agent.run("go", MemoryContextStore())

    assert result == "I cannot help with that."
