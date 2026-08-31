"""Tests for Anthropic transport — Vertex AI, endpoint/header building, config."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.messages import (
    INPUT_PROVENANCE_FOOTER,
    UNATTRIBUTED_INPUT_PROVENANCE,
    InputProvenance,
    Message,
    input_provenance_header,
)
from axio.models import ModelRegistry

import axio_transport_anthropic
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


def test_convert_messages_frames_non_human_text_and_image() -> None:
    provenance = InputProvenance(human_authored=False, source="background-outcome", author="child-1")
    message = Message(
        role="user",
        content=[TextBlock(text="done"), ImageBlock(media_type="image/png", data=b"image")],
        provenance=provenance,
    )

    [converted] = _convert_messages([message])

    assert converted["content"][0] == {"type": "text", "text": input_provenance_header(provenance)}
    assert converted["content"][1] == {"type": "text", "text": "done"}
    assert converted["content"][2]["type"] == "image"
    assert converted["content"][3] == {"type": "text", "text": INPUT_PROVENANCE_FOOTER}


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
    assert result[2]["content"] == [
        {"type": "text", "text": input_provenance_header(UNATTRIBUTED_INPUT_PROVENANCE)},
        {"type": "text", "text": "Notification: background task finished"},
        {"type": "text", "text": INPUT_PROVENANCE_FOOTER},
    ]


def test_adjacent_human_and_internal_inputs_have_closed_independent_envelopes() -> None:
    human = InputProvenance(human_authored=True, source="interactive", author="human")
    peer = InputProvenance(human_authored=False, source="peer", author="child-1")

    converted = _convert_messages(
        [
            Message(role="user", content=[TextBlock(text="question")], provenance=human),
            Message(role="user", content=[TextBlock(text="report")], provenance=peer),
        ]
    )

    assert converted[0]["content"] == [
        {"type": "text", "text": input_provenance_header(human)},
        {"type": "text", "text": "question"},
        {"type": "text", "text": INPUT_PROVENANCE_FOOTER},
    ]
    assert converted[1]["content"] == [
        {"type": "text", "text": input_provenance_header(peer)},
        {"type": "text", "text": "report"},
        {"type": "text", "text": INPUT_PROVENANCE_FOOTER},
    ]


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


def test_transport_vertexai_to_from_dict(monkeypatch: Any) -> None:
    # Pinned for the same reason as test_string_settings_coercion: an explicit
    # vertexai raises without google-auth[requests], and round-tripping the
    # config is what this test is about.
    monkeypatch.setattr(axio_transport_anthropic, "_google_auth_available", lambda: True)
    t = AnthropicTransport(vertexai=True, project="my-project", location="us-east5")
    d = t.to_dict()
    assert d["vertexai"] is True
    assert d["project"] == "my-project"
    assert d["location"] == "us-east5"
    t2 = AnthropicTransport.from_dict(d)
    assert t2.vertexai is True
    assert t2.project == "my-project"


def test_string_settings_coercion(monkeypatch: Any) -> None:
    # Availability is pinned so the coercion is what this test measures: an
    # explicit vertexai now raises when google-auth[requests] is absent, which
    # would otherwise make the result depend on the environment.
    monkeypatch.setattr(axio_transport_anthropic, "_google_auth_available", lambda: True)
    t = AnthropicTransport(
        vertexai="true",  # type: ignore[arg-type]
    )
    assert t.vertexai is True


# ---------------------------------------------------------------------------
# Vertex AI requires google-auth
# ---------------------------------------------------------------------------


def test_vertexai_defaults_to_false() -> None:
    assert AnthropicTransport(api_key="sk-test").vertexai is False


def test_vertexai_raises_when_google_auth_missing(monkeypatch: Any) -> None:
    monkeypatch.setattr(axio_transport_anthropic, "_google_auth_available", lambda: False)
    with pytest.raises(ImportError, match="google-auth"):
        AnthropicTransport(vertexai=True, project="proj")


def test_explicit_vertexai_as_a_string_also_raises(monkeypatch: Any) -> None:
    monkeypatch.setattr(axio_transport_anthropic, "_google_auth_available", lambda: False)
    for value in ("true", "1"):
        with pytest.raises(ImportError, match="google-auth"):
            AnthropicTransport(vertexai=value, project="proj")  # type: ignore[arg-type]


def test_vertexai_false_does_not_require_google_auth(monkeypatch: Any) -> None:
    monkeypatch.setattr(axio_transport_anthropic, "_google_auth_available", lambda: False)
    assert AnthropicTransport(vertexai=False).vertexai is False


def test_google_auth_available_requires_the_requests_extra(monkeypatch: Any) -> None:
    """``_get_vertex_access_token`` imports ``google.auth.transport.requests``.

    That module raises ``ImportError`` when ``requests`` is absent, so
    ``google-auth`` installed without its ``requests`` extra would satisfy a bare
    ``google.auth`` check and still fail at request time — the exact failure this
    guard exists to prevent.
    """
    real_find_spec = importlib.util.find_spec

    def without_requests(name: str, package: str | None = None) -> Any:
        return None if name == "requests" else real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", without_requests)
    assert axio_transport_anthropic._google_auth_available() is False


def test_models_registry() -> None:
    assert isinstance(ANTHROPIC_MODELS, ModelRegistry)
    assert "claude-sonnet-4-6" in ANTHROPIC_MODELS
    assert "claude-opus-4-6" in ANTHROPIC_MODELS
    assert "claude-haiku-4-5" in ANTHROPIC_MODELS
    assert "claude-haiku-4-5-20251001" in ANTHROPIC_MODELS


# ---------------------------------------------------------------------------
# Endpoint building
# ---------------------------------------------------------------------------


def test_direct_api_endpoint() -> None:
    t = AnthropicTransport(api_key="sk-test")
    assert t._build_url() == "https://api.anthropic.com/v1/messages"


def test_vertex_endpoint_regional(monkeypatch: Any) -> None:
    # Availability is pinned: an explicit vertexai raises without
    # google-auth[requests], and the endpoint shape is what this measures.
    monkeypatch.setattr(axio_transport_anthropic, "_google_auth_available", lambda: True)
    t = AnthropicTransport(vertexai=True, project="proj", location="us-east5")
    endpoint = t._build_url()
    assert "us-east5-aiplatform.googleapis.com" in endpoint
    assert "publishers/anthropic/models/claude-sonnet-4-6" in endpoint
    assert ":streamRawPredict" in endpoint


def test_vertex_endpoint_global(monkeypatch: Any) -> None:
    # Availability is pinned: an explicit vertexai raises without
    # google-auth[requests], and the endpoint shape is what this measures.
    monkeypatch.setattr(axio_transport_anthropic, "_google_auth_available", lambda: True)
    t = AnthropicTransport(vertexai=True, project="proj", location="global")
    endpoint = t._build_url()
    assert "aiplatform.googleapis.com/v1/" in endpoint
    assert "locations/global/" in endpoint
    assert "global-aiplatform" not in endpoint


# ---------------------------------------------------------------------------
# Header building
# ---------------------------------------------------------------------------


def test_direct_api_headers() -> None:
    t = AnthropicTransport(api_key="sk-test")
    headers = t._build_headers()
    assert headers["x-api-key"] == "sk-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def test_direct_api_body_includes_model() -> None:
    t = AnthropicTransport(api_key="sk-test")
    body = t.build_payload(
        [Message(role="user", content=[TextBlock(text="Hi")])],
        [],
        "system prompt",
    )
    assert body["model"] == "claude-sonnet-4-6"
    assert "anthropic_version" not in body


def test_vertex_body_includes_version(monkeypatch: Any) -> None:
    # Availability is pinned: an explicit vertexai raises without
    # google-auth[requests], and the endpoint shape is what this measures.
    monkeypatch.setattr(axio_transport_anthropic, "_google_auth_available", lambda: True)
    t = AnthropicTransport(vertexai=True, project="proj", location="us-east5")
    body = t.build_payload(
        [Message(role="user", content=[TextBlock(text="Hi")])],
        [],
        "",
    )
    assert body["anthropic_version"] == "vertex-2023-10-16"
    assert "model" not in body
