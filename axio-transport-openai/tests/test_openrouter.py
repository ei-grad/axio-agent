"""Tests for OpenRouter CompletionTransport."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from axio.blocks import ToolUseBlock
from axio.events import IterationEnd, ReasoningDelta, StreamEvent, TextDelta
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.types import CostSource, StopReason, Usage

from axio_transport_openai.openrouter import (
    OPENROUTER_BROKEN_TOOL_ARGUMENT_PROVIDERS,
    OPENROUTER_DEEPSEEK_V4_FLASH_PROVIDERS,
    OpenRouterTransport,
)

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_chunk(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _text_chunks(text: str) -> str:
    lines = ""
    for ch in text:
        lines += _sse_chunk({"choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}]})
    lines += _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    lines += _sse_chunk({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    lines += _sse_done()
    return lines


async def _write_content(content: str) -> str:
    return content


# ---------------------------------------------------------------------------
# Fake server with /api/v1/models and /api/v1/chat/completions
# ---------------------------------------------------------------------------


class FakeOpenRouterServer:
    def __init__(self) -> None:
        self.sse_responses: list[str] = []
        self.received_payloads: list[dict[str, Any]] = []
        self.models_response: dict[str, Any] = {"data": []}
        self.models_status: int = 200
        self.completions_status: int = 200
        self.error_body: str = ""

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/api/v1/models", self._handle_models)
        app.router.add_post("/api/v1/chat/completions", self._handle_completions)
        return app

    async def _handle_models(self, request: web.Request) -> web.Response:
        if self.models_status != 200:
            return web.Response(status=self.models_status, text=self.error_body)
        return web.json_response(self.models_response)

    async def _handle_completions(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        self.received_payloads.append(payload)

        if self.completions_status != 200:
            return web.Response(status=self.completions_status, text=self.error_body)

        sse_body = self.sse_responses.pop(0) if self.sse_responses else _sse_done()
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        await resp.write(sse_body.encode("utf-8"))
        await resp.write_eof()
        return resp


@pytest.fixture
async def fake_server() -> AsyncIterator[tuple[FakeOpenRouterServer, str]]:
    server = FakeOpenRouterServer()
    app = server.make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    sock = site._server.sockets[0]  # type: ignore[union-attr]
    host, port = sock.getsockname()[:2]
    base_url = f"http://{host}:{port}/api/v1"

    yield server, base_url

    await runner.cleanup()


@pytest.fixture
async def transport(fake_server: tuple[FakeOpenRouterServer, str]) -> AsyncIterator[OpenRouterTransport]:
    _, base_url = fake_server
    async with aiohttp.ClientSession() as session:
        yield OpenRouterTransport(base_url=base_url, api_key="test-key", session=session)


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_base_url() -> None:
    t = OpenRouterTransport()
    assert t.base_url == "https://openrouter.ai/api/v1"


def test_default_model() -> None:
    t = OpenRouterTransport()
    assert t.model.id == "google/gemini-2.5-pro-preview"


def test_default_models_inherited() -> None:
    t = OpenRouterTransport()
    assert len(t.models) > 0
    assert "gpt-4.1-mini" in t.models


def test_resolve_model_exact() -> None:
    t = OpenRouterTransport()

    spec = t.resolve_model("gpt-4.1-mini")

    assert spec.id == "gpt-4.1-mini"


def test_resolve_model_variant_uses_base_metadata() -> None:
    base = ModelSpec(
        id="z-ai/glm-4.7",
        capabilities=frozenset({Capability.text, Capability.tool_use}),
        context_window=256_000,
        max_output_tokens=16_384,
        input_cost=0.2,
        output_cost=1.0,
    )
    t = OpenRouterTransport(models=ModelRegistry([base]))

    spec = t.resolve_model("z-ai/glm-4.7:nitro")

    assert spec == ModelSpec(
        id="z-ai/glm-4.7:nitro",
        capabilities=base.capabilities,
        context_window=base.context_window,
        max_output_tokens=base.max_output_tokens,
        input_cost=base.input_cost,
        output_cost=base.output_cost,
    )


def test_resolve_model_pins_a_provider() -> None:
    base = ModelSpec(id="z-ai/glm-4.7", context_window=200_000)
    t = OpenRouterTransport(models=ModelRegistry([base]))

    spec = t.resolve_model("z-ai/glm-4.7@Cerebras")

    assert spec.id == "z-ai/glm-4.7@Cerebras"
    assert spec.context_window == 200_000


def test_resolve_model_pins_a_provider_on_a_variant() -> None:
    base = ModelSpec(id="z-ai/glm-4.7", context_window=200_000)
    t = OpenRouterTransport(models=ModelRegistry([base]))

    spec = t.resolve_model("z-ai/glm-4.7:nitro@DeepInfra")

    assert spec.id == "z-ai/glm-4.7:nitro@DeepInfra"
    assert spec.context_window == 200_000


def test_resolve_model_unknown_with_provider_raises_requested_id() -> None:
    t = OpenRouterTransport(models=ModelRegistry())

    with pytest.raises(KeyError) as exc_info:
        t.resolve_model("nope/nope@Cerebras")

    assert exc_info.value.args == ("nope/nope@Cerebras",)


def test_resolve_model_variant_missing_base_raises_requested_id() -> None:
    t = OpenRouterTransport(models=ModelRegistry())

    with pytest.raises(KeyError) as exc_info:
        t.resolve_model("z-ai/glm-4.7:nitro")

    assert exc_info.value.args == ("z-ai/glm-4.7:nitro",)


# ---------------------------------------------------------------------------
# fetch_models
# ---------------------------------------------------------------------------


async def test_fetch_models_populates_registry(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, _ = fake_server
    server.models_response = {
        "data": [
            {"id": "openai/gpt-4", "context_length": 8192, "top_provider": {"max_completion_tokens": 4096}},
            {
                "id": "anthropic/claude-3-opus",
                "context_length": 200000,
                "top_provider": {"max_completion_tokens": 4096},
            },  # noqa: E501
        ]
    }

    with caplog.at_level(logging.INFO, logger="axio_transport_openai.openrouter"):
        await transport.fetch_models()

    assert isinstance(transport.models, ModelRegistry)
    assert "openai/gpt-4" in transport.models
    assert "anthropic/claude-3-opus" in transport.models
    assert any("Loaded 2 models" in r.message for r in caplog.records)


async def test_fetch_models_populates_specs(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {
        "data": [
            {
                "id": "openai/gpt-4o",
                "context_length": 128000,
                "top_provider": {"max_completion_tokens": 16384, "context_length": 128000, "is_moderated": False},
                "architecture": {
                    "modality": "text+image->text",
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "tokenizer": "GPT",
                    "instruct_type": "chatml",
                },
                "supported_parameters": ["temperature", "tools", "max_tokens"],
                "pricing": {"prompt": "0.0000025", "completion": "0.00001", "request": "0", "image": "0"},
            },
            {
                "id": "openai/text-embedding-3-small",
                "context_length": 8192,
                "top_provider": {"max_completion_tokens": 0},
                "architecture": {
                    "modality": "text->embedding",
                    "input_modalities": ["text"],
                    "output_modalities": ["embedding"],
                },
                "supported_parameters": [],
                "pricing": {"prompt": "0.00000002", "completion": "0"},
            },
        ]
    }

    await transport.fetch_models()

    gpt4o = transport.models["openai/gpt-4o"]
    assert gpt4o.context_window == 128000
    assert gpt4o.max_output_tokens == 16384
    assert Capability.tool_use in gpt4o.capabilities
    assert Capability.vision in gpt4o.capabilities
    assert gpt4o.input_cost == pytest.approx(2.5)
    assert gpt4o.output_cost == pytest.approx(10.0)

    emb = transport.models["openai/text-embedding-3-small"]
    assert emb.context_window == 8192
    assert Capability.embedding in emb.capabilities
    assert Capability.vision not in emb.capabilities
    assert Capability.tool_use not in emb.capabilities


async def test_fetch_models_clears_and_repopulates(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    transport.models["custom/model"] = ModelSpec(id="custom/model", context_window=4096, max_output_tokens=1024)

    server.models_response = {
        "data": [
            {"id": "openai/gpt-4", "context_length": 8192, "top_provider": {"max_completion_tokens": 4096}},
        ]
    }

    await transport.fetch_models()

    assert "custom/model" not in transport.models
    assert "openai/gpt-4" in transport.models


async def test_fetch_models_empty(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {"data": []}

    await transport.fetch_models()

    assert len(transport.models) == 0


async def test_fetch_models_error(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.models_status = 401
    server.error_body = "Unauthorized"

    with pytest.raises(StreamError, match="401"):
        await transport.fetch_models()


async def test_fetch_models_context_from_top_provider_fallback(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    """context_length from top_provider used when top-level field is missing/null."""
    server, _ = fake_server
    server.models_response = {
        "data": [
            {
                "id": "some/model",
                "context_length": None,
                "top_provider": {"context_length": 32768, "max_completion_tokens": 2048},
            },
        ]
    }

    await transport.fetch_models()

    assert transport.models["some/model"].context_window == 32768


async def test_fetch_models_defaults_when_missing(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    """Minimal model entry with no context/pricing fields gets safe defaults."""
    server, _ = fake_server
    server.models_response = {
        "data": [
            {"id": "some/minimal"},
        ]
    }

    await transport.fetch_models()

    m = transport.models["some/minimal"]
    assert m.context_window == 128_000
    assert m.max_output_tokens == 8_000
    assert m.input_cost == 0.0
    assert m.output_cost == 0.0


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


async def test_no_tools_no_capability(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {
        "data": [
            {
                "id": "text-only/model",
                "context_length": 4096,
                "top_provider": {"max_completion_tokens": 512},
                "supported_parameters": ["temperature", "max_tokens"],
            }
        ]
    }

    await transport.fetch_models()

    assert Capability.tool_use not in transport.models["text-only/model"].capabilities


async def test_reasoning_capability_from_supported_parameters(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {
        "data": [
            {"id": "thinks/a-lot", "supported_parameters": ["reasoning", "include_reasoning", "tools"]},
            {"id": "thinks/not", "supported_parameters": ["tools"]},
        ]
    }

    await transport.fetch_models()

    assert Capability.reasoning in transport.models["thinks/a-lot"].capabilities
    assert Capability.reasoning not in transport.models["thinks/not"].capabilities


async def test_reasoning_metadata_constrains_native_effort(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {
        "data": [
            {
                "id": "thinks/mandatory",
                "supported_parameters": ["reasoning"],
                "reasoning": {"supported_efforts": ["high", "low"], "mandatory": True},
            }
        ]
    }

    await transport.fetch_models()
    transport.model = transport.models["thinks/mandatory"]
    state = transport.configure_effort("default")

    assert state.allowed == ("low", "high")
    with pytest.raises(ValueError, match="none.*not supported"):
        transport.configure_effort("none")
    assert "reasoning" not in transport.build_payload([], [], "")


async def test_null_reasoning_efforts_means_all_gateway_levels(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.models_response = {
        "data": [
            {
                "id": "thinks/all",
                "reasoning": {"supported_efforts": None, "mandatory": False},
            },
            {
                "id": "thinks/all-mandatory",
                "reasoning": {"supported_efforts": None, "mandatory": True},
            },
        ]
    }

    await transport.fetch_models()

    transport.model = transport.models["thinks/all"]
    state = transport.configure_effort("high")
    assert state.allowed == ("none", "low", "medium", "high", "xhigh", "max")
    assert transport.build_payload([], [], "")["reasoning"] == {"effort": "high"}

    transport.model = transport.models["thinks/all-mandatory"]
    state = transport.configure_effort("default")
    assert state.allowed == ("low", "medium", "high", "xhigh", "max")
    with pytest.raises(ValueError, match="none.*not supported"):
        transport.configure_effort("none")


def test_non_axio_native_efforts_do_not_enable_prompt_fallback() -> None:
    transport = OpenRouterTransport(
        model=ModelSpec(id="thinks/minimal", capabilities=frozenset({Capability.reasoning})),
        _reasoning_efforts={"thinks/minimal": ("minimal",)},
    )

    state = transport.configure_effort("default")

    assert state.mechanism.value == "native-effort"
    assert state.allowed == ()
    with pytest.raises(ValueError, match="high.*not supported"):
        transport.configure_effort("high")


def test_rejected_native_effort_does_not_mutate_legacy_state() -> None:
    transport = OpenRouterTransport(
        model=ModelSpec(id="thinks/low", capabilities=frozenset({Capability.reasoning})),
        thinking=True,
        _reasoning_efforts={"thinks/low": ("low",)},
    )
    before = transport.build_payload([], [], "")

    with pytest.raises(ValueError, match="high.*not supported"):
        transport.configure_effort("high")

    assert transport.thinking is True
    assert transport.reasoning_effort is None
    assert transport.build_payload([], [], "") == before


# ---------------------------------------------------------------------------
# Request payload
# ---------------------------------------------------------------------------


async def test_an_output_limit_equal_to_the_window_is_not_reserved(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    # Reserving all 262144 for the answer leaves no room for the prompt, and the
    # request is rejected before a token is generated.
    server, _ = fake_server
    server.models_response = {
        "data": [
            {
                "id": "google/gemma-4-31b-it",
                "context_length": 262_144,
                "top_provider": {"max_completion_tokens": 262_144},
            }
        ]
    }
    await transport.fetch_models()
    transport.model = transport.models["google/gemma-4-31b-it"]
    server.sse_responses.append(_text_chunks("Hi"))

    await _collect(transport.stream([], [], "You are terse."))

    assert server.received_payloads[0]["max_completion_tokens"] < 262_144


async def test_provider_suffix_becomes_provider_only(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    transport.model = ModelSpec(id="z-ai/glm-4.7@Cerebras")
    server.sse_responses.append(_text_chunks("Hi"))

    await _collect(transport.stream([], [], ""))

    payload = server.received_payloads[0]
    assert payload["model"] == "z-ai/glm-4.7"
    assert payload["provider"] == {
        "only": ["Cerebras"],
        "ignore": list(OPENROUTER_BROKEN_TOOL_ARGUMENT_PROVIDERS),
    }


async def test_plain_model_excludes_known_whitespace_broken_providers(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    transport.model = ModelSpec(id="z-ai/glm-4.7")
    server.sse_responses.append(_text_chunks("Hi"))

    await _collect(transport.stream([], [], ""))

    payload = server.received_payloads[0]
    assert payload["model"] == "z-ai/glm-4.7"
    assert payload["provider"] == {"ignore": list(OPENROUTER_BROKEN_TOOL_ARGUMENT_PROVIDERS)}


def test_existing_provider_ignore_is_merged_and_deduplicated() -> None:
    transport = OpenRouterTransport(
        extra_params={
            "provider": {
                "ignore": ["deepinfra/fp8", "streamlake/fp8", "DeepInfra/FP8"],
                "allow_fallbacks": False,
            }
        }
    )

    payload = transport.build_payload([], [], "")

    assert payload["provider"] == {
        "ignore": ["deepinfra/fp8", "streamlake/fp8", "relace/fp4", "cloudflare"],
        "allow_fallbacks": False,
    }


def test_safe_provider_only_and_order_are_preserved() -> None:
    transport = OpenRouterTransport(extra_params={"provider": {"only": ["deepinfra/fp8"], "order": ["deepinfra/fp8"]}})

    payload = transport.build_payload([], [], "")

    assert payload["provider"] == {
        "only": ["deepinfra/fp8"],
        "order": ["deepinfra/fp8"],
        "ignore": list(OPENROUTER_BROKEN_TOOL_ARGUMENT_PROVIDERS),
    }


@pytest.mark.parametrize("field", ["only", "order"])
@pytest.mark.parametrize("provider", ["streamlake/fp8", "Relace", "cloudflare"])
def test_explicit_broken_provider_selection_is_rejected(field: str, provider: str) -> None:
    transport = OpenRouterTransport(extra_params={"provider": {field: [provider]}})

    with pytest.raises(ValueError, match=rf"provider\.{field} selects excluded"):
        transport.build_payload([], [], "")


def test_provider_only_and_ignore_conflict_is_rejected() -> None:
    transport = OpenRouterTransport(
        extra_params={"provider": {"only": ["deepinfra/fp8"], "ignore": ["deepinfra/fp8"]}}
    )

    with pytest.raises(ValueError, match="provider.only conflicts with provider.ignore"):
        transport.build_payload([], [], "")


def test_provider_order_outside_only_is_rejected() -> None:
    transport = OpenRouterTransport(extra_params={"provider": {"only": ["deepinfra/fp8"], "order": ["cerebras"]}})

    with pytest.raises(ValueError, match="provider.order must be a subset"):
        transport.build_payload([], [], "")


def test_model_provider_pin_conflicts_are_explicit_and_equivalent_duplicates_are_accepted() -> None:
    broken = OpenRouterTransport(model=ModelSpec(id="deepseek/deepseek-v4-flash@StreamLake"))
    conflicting = OpenRouterTransport(
        model=ModelSpec(id="z-ai/glm-4.7@Cerebras"),
        extra_params={"provider": {"only": ["deepinfra/fp8"]}},
    )
    equivalent = OpenRouterTransport(
        model=ModelSpec(id="z-ai/glm-4.7@Cerebras"),
        extra_params={"provider": {"only": ["Cerebras", "cerebras"]}},
    )

    with pytest.raises(ValueError, match="provider pin selects excluded"):
        broken.build_payload([], [], "")
    with pytest.raises(ValueError, match="provider pin conflicts with provider.only"):
        conflicting.build_payload([], [], "")
    assert equivalent.build_payload([], [], "")["provider"]["only"] == ["Cerebras"]


def test_provider_routing_must_be_an_object() -> None:
    transport = OpenRouterTransport(extra_params={"provider": "deepinfra/fp8"})

    with pytest.raises(ValueError, match="provider routing must be an object"):
        transport.build_payload([], [], "")


def test_provider_policy_round_trips_with_serialized_transport() -> None:
    transport = OpenRouterTransport(
        model=ModelSpec(id="z-ai/glm-4.7"),
        extra_params={"provider": {"order": ["deepinfra/fp8"], "allow_fallbacks": False}},
    )

    restored = OpenRouterTransport.from_dict(transport.to_dict())

    assert restored.extra_params == transport.extra_params
    assert restored.build_payload([], [], "")["provider"] == {
        "order": ["deepinfra/fp8"],
        "allow_fallbacks": False,
        "ignore": list(OPENROUTER_BROKEN_TOOL_ARGUMENT_PROVIDERS),
    }


def test_provider_exclusion_follows_runtime_model_switch() -> None:
    transport = OpenRouterTransport(model=ModelSpec(id="deepseek/deepseek-v4-flash"))

    assert transport.build_payload([], [], "")["provider"] == {
        "only": list(OPENROUTER_DEEPSEEK_V4_FLASH_PROVIDERS),
        "allow_fallbacks": False,
        "ignore": list(OPENROUTER_BROKEN_TOOL_ARGUMENT_PROVIDERS),
    }
    transport.model = ModelSpec(id="deepseek/deepseek-v4-flash@streamlake/fp8")
    with pytest.raises(ValueError, match="provider pin selects excluded"):
        transport.build_payload([], [], "")
    transport.model = ModelSpec(id="deepseek/deepseek-v4-flash@deepinfra/fp8")
    assert transport.build_payload([], [], "")["provider"] == {
        "only": ["deepinfra/fp8"],
        "allow_fallbacks": False,
        "ignore": list(OPENROUTER_BROKEN_TOOL_ARGUMENT_PROVIDERS),
    }


@pytest.mark.parametrize("field", ["only", "order"])
def test_deepseek_v4_flash_rejects_unverified_provider_selection(field: str) -> None:
    transport = OpenRouterTransport(
        model=ModelSpec(id="deepseek/deepseek-v4-flash"),
        extra_params={"provider": {field: ["siliconflow/fp8"]}},
    )

    with pytest.raises(ValueError, match=rf"provider\.{field} selects unverified"):
        transport.build_payload([], [], "")


def test_deepseek_v4_flash_rejects_unverified_provider_pin() -> None:
    transport = OpenRouterTransport(model=ModelSpec(id="deepseek/deepseek-v4-flash@azure/us"))

    with pytest.raises(ValueError, match="provider pin selects unverified"):
        transport.build_payload([], [], "")


def test_deepseek_v4_flash_preserves_verified_provider_subset() -> None:
    transport = OpenRouterTransport(
        model=ModelSpec(id="deepseek/deepseek-v4-flash"),
        extra_params={
            "provider": {
                "only": ["venice", "deepinfra/fp8"],
                "order": ["venice"],
                "allow_fallbacks": True,
            }
        },
    )

    assert transport.build_payload([], [], "")["provider"] == {
        "only": ["venice", "deepinfra/fp8"],
        "order": ["venice"],
        "allow_fallbacks": False,
        "ignore": list(OPENROUTER_BROKEN_TOOL_ARGUMENT_PROVIDERS),
    }


def test_deepseek_v4_flash_user_ignore_narrows_default_allowlist() -> None:
    transport = OpenRouterTransport(
        model=ModelSpec(id="deepseek/deepseek-v4-flash"),
        extra_params={"provider": {"ignore": ["venice"]}},
    )

    assert transport.build_payload([], [], "")["provider"] == {
        "only": ["digitalocean", "deepinfra/fp8", "novita/fp8"],
        "allow_fallbacks": False,
        "ignore": ["venice", *OPENROUTER_BROKEN_TOOL_ARGUMENT_PROVIDERS],
    }


def test_deepseek_v4_flash_rejects_ignoring_every_verified_provider() -> None:
    transport = OpenRouterTransport(
        model=ModelSpec(id="deepseek/deepseek-v4-flash"),
        extra_params={"provider": {"ignore": list(OPENROUTER_DEEPSEEK_V4_FLASH_PROVIDERS)}},
    )

    with pytest.raises(ValueError, match="provider.ignore excludes every verified provider"):
        transport.build_payload([], [], "")


def test_tool_schema_and_history_keep_plain_canonical_strings() -> None:
    transport = OpenRouterTransport(model=ModelSpec(id="deepseek/deepseek-v4-flash"))
    tool: Tool[Any] = Tool(name="write_content", handler=_write_content)
    content = "          first\n\tsecond  "
    history = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call-1", name="write_content", input={"content": content})],
        )
    ]

    payload = transport.build_payload(history, [tool], "")

    schema = payload["tools"][0]["function"]["parameters"]["properties"]["content"]
    call = payload["messages"][0]["tool_calls"][0]
    assert schema == {"type": "string"}
    assert json.loads(call["function"]["arguments"]) == {"content": content}


async def test_thinking_is_requested_the_openrouter_way(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    transport.model = ModelSpec(id="thinks/a-lot", capabilities=frozenset({Capability.reasoning}))
    transport.thinking = True
    server.sse_responses.append(_text_chunks("Hi"))

    await _collect(transport.stream([], [], ""))

    payload = server.received_payloads[0]
    assert payload["reasoning"] == {"enabled": True}
    assert "enable_thinking" not in payload


def test_effort_is_requested_with_openrouter_native_reasoning() -> None:
    transport = OpenRouterTransport(
        model=ModelSpec(id="thinks/a-lot", capabilities=frozenset({Capability.reasoning})),
        _reasoning_efforts={"thinks/a-lot": ("low", "xhigh")},
    )

    state = transport.configure_effort("xhigh")
    payload = transport.build_payload([], [], "")

    assert state.mechanism.value == "native-effort"
    assert state.allowed == ("low", "xhigh")
    assert payload["reasoning"] == {"effort": "xhigh"}


def test_effort_default_clears_legacy_thinking() -> None:
    transport = OpenRouterTransport(
        model=ModelSpec(id="thinks/a-lot", capabilities=frozenset({Capability.reasoning})),
        thinking=True,
    )

    state = transport.configure_effort("default")
    payload = transport.build_payload([], [], "")

    assert state.requested is None
    assert "reasoning" not in payload


# ---------------------------------------------------------------------------
# Streaming (inherited from ChatCompletionsTransport)
# ---------------------------------------------------------------------------


async def test_text_streaming(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    server.sse_responses.append(_text_chunks("Hi"))

    events = await _collect(transport.stream([], [], ""))

    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(e.delta for e in text_deltas) == "Hi"

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert ends[0].stop_reason == StopReason.end_turn
    assert ends[0].usage == Usage(10, 5)


async def test_reported_cost_is_preserved(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    sse = _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    sse += _sse_chunk(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost": 0.000123,
                "cost_details": {"upstream_inference_cost": 0.0001},
            },
        }
    )
    sse += _sse_done()
    server.sse_responses.append(sse)

    events = await _collect(transport.stream([], [], ""))

    usage = [event.usage for event in events if isinstance(event, IterationEnd)][0]
    assert usage == Usage(10, 5, cost_usd=0.000123, cost_source=CostSource.provider)


async def test_multiple_usage_chunks_use_the_latest_reported_total_once(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    sse = _sse_chunk({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 1, "cost": 0.01}})
    sse += _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    sse += _sse_chunk({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.02}})
    sse += _sse_done()
    server.sse_responses.append(sse)

    events = await _collect(transport.stream([], [], ""))

    usage = [event.usage for event in events if isinstance(event, IterationEnd)][0]
    assert usage == Usage(10, 5, cost_usd=0.02, cost_source=CostSource.provider)


async def test_earlier_reported_cost_survives_a_later_token_only_usage_chunk(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    server, _ = fake_server
    sse = _sse_chunk({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 1, "cost": 0.01}})
    sse += _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    sse += _sse_chunk({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    sse += _sse_done()
    server.sse_responses.append(sse)

    events = await _collect(transport.stream([], [], ""))

    usage = [event.usage for event in events if isinstance(event, IterationEnd)][0]
    assert usage == Usage(10, 5, cost_usd=0.01, cost_source=CostSource.provider)


@pytest.mark.parametrize("invalid_cost", [-0.1, True, "0.1", float("nan")])
async def test_malformed_reported_cost_is_ignored(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
    invalid_cost: object,
) -> None:
    server, _ = fake_server
    sse = _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    sse += _sse_chunk({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": invalid_cost}})
    sse += _sse_done()
    server.sse_responses.append(sse)

    events = await _collect(transport.stream([], [], ""))

    usage = [event.usage for event in events if isinstance(event, IterationEnd)][0]
    assert usage == Usage(10, 5)


async def test_reasoning_deltas_are_read(
    fake_server: tuple[FakeOpenRouterServer, str],
    transport: OpenRouterTransport,
) -> None:
    # llama.cpp and DeepSeek send reasoning_content; OpenRouter sends plain
    # reasoning. Reading only one spelling loses the thinking blocks entirely.
    server, _ = fake_server
    server.sse_responses.append(
        _sse_chunk({"choices": [{"index": 0, "delta": {"reasoning": "pondering"}}]}) + _text_chunks("Hi")
    )

    events = await _collect(transport.stream([], [], ""))

    assert [e.delta for e in events if isinstance(e, ReasoningDelta)] == ["pondering"]
