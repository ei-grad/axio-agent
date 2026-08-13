"""Tests for Anthropic transport — Vertex AI, endpoint/header building, config."""

from __future__ import annotations

from typing import Any

from axio.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from axio.messages import Message
from axio.models import ModelRegistry

from axio_transport_anthropic import (
    ANTHROPIC_MODELS,
    AnthropicTransport,
    _convert_messages,
)

# ---------------------------------------------------------------------------
# _convert_messages
# ---------------------------------------------------------------------------


def test_convert_messages_basic() -> None:
    messages = [
        Message(role="user", content=[TextBlock(text="Hi")]),
        Message(role="assistant", content=[TextBlock(text="Hello")]),
    ]
    result = _convert_messages(messages)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"


def test_convert_messages_adjacent_user_messages_stay_separate() -> None:
    """The agent loop can inject a follow-up user message (e.g. a notification) as its
    own Message right after a tool-results message. The Messages API accepts
    consecutive same-role messages (master already ships this shape on the interrupt
    path), so both must come through as their own dict, in order — no merging."""
    messages = [
        Message(role="assistant", content=[ToolUseBlock(id="call_1", name="get_weather", input={})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_1", content="22C")]),
        Message(role="user", content=[TextBlock(text="Notification: background task finished")]),
    ]
    result = _convert_messages(messages)
    assert len(result) == 3
    assert result[0]["role"] == "assistant"
    assert result[1]["role"] == "user"
    assert result[1]["content"][0]["type"] == "tool_result"
    assert result[1]["content"][0]["tool_use_id"] == "call_1"
    assert result[2]["role"] == "user"
    assert result[2]["content"] == [{"type": "text", "text": "Notification: background task finished"}]


# ---------------------------------------------------------------------------
# Transport config
# ---------------------------------------------------------------------------


def test_transport_defaults() -> None:
    t = AnthropicTransport()
    assert t.model.id == "claude-sonnet-4-6"
    assert t.name == "Anthropic"


def test_transport_to_from_dict() -> None:
    t = AnthropicTransport(api_key="sk-test")
    d = t.to_dict()
    assert d["api_key"] == "sk-test"
    t2 = AnthropicTransport.from_dict(d)
    assert t2.api_key == "sk-test"


def test_transport_vertexai_to_from_dict() -> None:
    t = AnthropicTransport(vertexai=True, project="my-project", location="us-east5")
    d = t.to_dict()
    assert d["vertexai"] is True
    assert d["project"] == "my-project"
    assert d["location"] == "us-east5"
    t2 = AnthropicTransport.from_dict(d)
    assert t2.vertexai is True
    assert t2.project == "my-project"


def test_string_settings_coercion() -> None:
    t = AnthropicTransport(
        vertexai="true",  # type: ignore[arg-type]
    )
    assert t.vertexai is True


def test_models_registry() -> None:
    assert isinstance(ANTHROPIC_MODELS, ModelRegistry)
    assert "claude-sonnet-4-6" in ANTHROPIC_MODELS
    assert "claude-opus-4-6" in ANTHROPIC_MODELS
    assert "claude-haiku-4-5" in ANTHROPIC_MODELS
    assert "claude-haiku-4-5-20251001" in ANTHROPIC_MODELS


# ---------------------------------------------------------------------------
# Endpoint building
# ---------------------------------------------------------------------------


def test_direct_api_endpoint(monkeypatch: Any) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    t = AnthropicTransport(api_key="sk-test")
    assert t._build_url() == "https://api.anthropic.com/v1/messages"


def test_vertex_endpoint_regional() -> None:
    t = AnthropicTransport(vertexai=True, project="proj", location="us-east5")
    endpoint = t._build_url()
    assert "us-east5-aiplatform.googleapis.com" in endpoint
    assert "publishers/anthropic/models/claude-sonnet-4-6" in endpoint
    assert ":streamRawPredict" in endpoint


def test_vertex_endpoint_global() -> None:
    t = AnthropicTransport(vertexai=True, project="proj", location="global")
    endpoint = t._build_url()
    assert "aiplatform.googleapis.com/v1/" in endpoint
    assert "locations/global/" in endpoint
    assert "global-aiplatform" not in endpoint


# ---------------------------------------------------------------------------
# Header building
# ---------------------------------------------------------------------------


def test_direct_api_headers(monkeypatch: Any) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    t = AnthropicTransport(api_key="sk-test")
    headers = t._build_headers()
    assert headers["x-api-key"] == "sk-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def test_direct_api_body_includes_model(monkeypatch: Any) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    t = AnthropicTransport(api_key="sk-test")
    body = t.build_payload(
        [Message(role="user", content=[TextBlock(text="Hi")])],
        [],
        "system prompt",
    )
    assert body["model"] == "claude-sonnet-4-6"
    assert "anthropic_version" not in body


def test_vertex_body_includes_version() -> None:
    t = AnthropicTransport(vertexai=True, project="proj", location="us-east5")
    body = t.build_payload(
        [Message(role="user", content=[TextBlock(text="Hi")])],
        [],
        "",
    )
    assert body["anthropic_version"] == "vertex-2023-10-16"
    assert "model" not in body
