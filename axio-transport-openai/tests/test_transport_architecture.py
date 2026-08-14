"""Public transport boundaries and plugin registration contracts."""

from __future__ import annotations

import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import axio_transport_openai
import axio_transport_openai.responses as responses
from axio_transport_openai import ChatCompletionsTransport, OpenAITransport
from axio_transport_openai.custom import CustomChatCompletionsTransport
from axio_transport_openai.llamacpp import LlamaCppTransport
from axio_transport_openai.nebius import NebiusTransport
from axio_transport_openai.openrouter import OpenRouterTransport


def test_public_transports_have_one_endpoint_each() -> None:
    assert OpenAITransport.stream_path == "responses"
    assert ChatCompletionsTransport.stream_path == "chat/completions"
    assert not issubclass(OpenAITransport, ChatCompletionsTransport)
    assert not issubclass(ChatCompletionsTransport, OpenAITransport)
    assert "supports_responses" not in OpenAITransport.__dataclass_fields__
    assert "supports_responses" not in ChatCompletionsTransport.__dataclass_fields__


def test_chat_compatible_providers_use_chat_completions() -> None:
    for transport_type in (
        CustomChatCompletionsTransport,
        NebiusTransport,
        OpenRouterTransport,
        LlamaCppTransport,
    ):
        assert issubclass(transport_type, ChatCompletionsTransport)
        assert transport_type.stream_path == "chat/completions"


def test_removed_transport_names_are_not_exported() -> None:
    assert not hasattr(axio_transport_openai, "OpenAIResponsesTransport")
    assert not hasattr(responses, "OpenAIResponsesTransport")
    assert not hasattr(axio_transport_openai, "OpenAICompatibleTransport")


def test_transport_entry_points_are_unambiguous() -> None:
    project_file = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(project_file.read_text())
    entry_points = project["project"]["entry-points"]["axio.transport"]

    assert entry_points["openai"] == "axio_transport_openai:OpenAITransport"
    assert "openai-responses" not in entry_points
    assert entry_points["openai-custom"].endswith(":CustomChatCompletionsTransport")


def test_installed_openai_entry_point_loads_responses_transport() -> None:
    transport_entries = {entry.name: entry for entry in entry_points(group="axio.transport")}

    assert transport_entries["openai"].load() is OpenAITransport
    assert "openai-responses" not in transport_entries


def test_old_supports_responses_flag_is_ignored_on_decode() -> None:
    restored = OpenAITransport.from_dict({"supports_responses": False})

    assert type(restored) is OpenAITransport
    assert "supports_responses" not in restored.to_dict()
