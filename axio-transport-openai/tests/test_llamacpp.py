"""Tests for llama.cpp model discovery and inherited chat streaming."""

from __future__ import annotations

import json
import tomllib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from axio.events import IterationEnd, ReasoningDelta, StreamEvent, TextDelta
from axio.exceptions import StreamError
from axio.models import Capability, ModelRegistry, ModelSpec
from textual.app import App
from textual.widgets import Input

from axio_transport_openai.llamacpp import LLAMA_CPP_BASE_URL, LlamaCppTransport
from axio_transport_openai.tui import LlamaCppSettingsScreen


class FakeLlamaCppServer:
    def __init__(self) -> None:
        self.router_response: dict[str, Any] | None = None
        self.router_props: dict[str, Any] = {"role": "router"}
        self.single_response: dict[str, Any] = {"object": "list", "data": []}
        self.single_props: dict[str, Any] = {}
        self.model_props: dict[str, dict[str, Any]] = {}
        self.props_status: dict[str, int] = {}
        self.requests: list[tuple[str, str, dict[str, str], str | None]] = []
        self.completion_payloads: list[dict[str, Any]] = []

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/models", self._models)
        app.router.add_get("/v1/models", self._v1_models)
        app.router.add_get("/props", self._props)
        app.router.add_post("/v1/chat/completions", self._completions)
        return app

    def _record(self, request: web.Request) -> None:
        self.requests.append(
            (
                request.method,
                request.path,
                dict(request.query),
                request.headers.get("Authorization"),
            )
        )

    async def _models(self, request: web.Request) -> web.Response:
        self._record(request)
        return web.json_response(self.router_response if self.router_response is not None else self.single_response)

    async def _v1_models(self, request: web.Request) -> web.Response:
        self._record(request)
        return web.json_response(self.single_response)

    async def _props(self, request: web.Request) -> web.Response:
        self._record(request)
        model_id = request.query.get("model")
        key = model_id or ""
        if status := self.props_status.get(key):
            return web.Response(status=status, text="props failed")
        if model_id is not None:
            return web.json_response(self.model_props[model_id])
        return web.json_response(self.router_props if self.router_response is not None else self.single_props)

    async def _completions(self, request: web.Request) -> web.StreamResponse:
        self._record(request)
        self.completion_payloads.append(await request.json())
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "done"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return web.Response(text=body, content_type="text/event-stream")


@pytest.fixture
async def fake_server() -> AsyncIterator[tuple[FakeLlamaCppServer, str]]:
    server = FakeLlamaCppServer()
    runner = web.AppRunner(server.make_app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    socket = site._server.sockets[0]  # type: ignore[union-attr]
    host, port = socket.getsockname()[:2]
    yield server, f"http://{host}:{port}/v1"
    await runner.cleanup()


@pytest.fixture
async def transport(fake_server: tuple[FakeLlamaCppServer, str]) -> AsyncIterator[LlamaCppTransport]:
    _, base_url = fake_server
    async with aiohttp.ClientSession() as session:
        yield LlamaCppTransport(base_url=base_url, session=session)


def _props(
    *,
    n_ctx: int = 8_192,
    n_predict: int = -1,
    tools: bool = False,
    tool_calls: bool = False,
    preserve_reasoning: bool = False,
    modalities: dict[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        "default_generation_settings": {"n_ctx": n_ctx, "params": {"n_predict": n_predict}},
        "chat_template_caps": {
            "supports_tools": tools,
            "supports_tool_calls": tool_calls,
            "supports_preserve_reasoning": preserve_reasoning,
            "supports_string_content": True,
        },
        "modalities": modalities or {},
    }


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


def test_defaults_are_local_and_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    transport = LlamaCppTransport()
    assert transport.name == "llama.cpp"
    assert transport.base_url == LLAMA_CPP_BASE_URL
    assert transport.api_key == ""
    assert len(transport.models) == 0
    assert "fetch_models" in transport.model.id


def test_llama_environment_isolated_from_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.invalid/v1")
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://llama.invalid/v1")
    monkeypatch.setenv("LLAMA_API_KEY", "llama-key")
    transport = LlamaCppTransport()
    assert transport.base_url == "http://llama.invalid/v1"
    assert transport.api_key == "llama-key"


def test_stream_rejected_before_discovery() -> None:
    transport = LlamaCppTransport()
    with pytest.raises(RuntimeError, match="fetch_models"):
        transport.stream([], [], "")


async def test_current_single_model_contract_uses_props_and_v1_models(
    fake_server: tuple[FakeLlamaCppServer, str],
    transport: LlamaCppTransport,
) -> None:
    server, _ = fake_server
    server.single_response = {"data": [{"id": "qwen", "meta": {"n_ctx": 16_384, "n_ctx_train": 131_072}}]}
    server.single_props = _props(
        n_ctx=32_768,
        n_predict=2_048,
        tools=True,
        tool_calls=True,
        preserve_reasoning=True,
        modalities={"vision": True, "audio": True, "video": True},
    )

    await transport.fetch_models()

    model = transport.models["qwen"]
    assert model.context_window == 32_768
    assert model.max_output_tokens == 2_048
    assert model.capabilities == frozenset(
        {Capability.text, Capability.vision, Capability.audio, Capability.video, Capability.tool_use}
    )
    assert Capability.reasoning not in model.capabilities
    assert model.input_cost == model.output_cost == 0.0
    assert transport.model is model
    assert [(method, path) for method, path, _, _ in server.requests] == [
        ("GET", "/props"),
        ("GET", "/models"),
        ("GET", "/v1/models"),
    ]


async def test_discovery_auth_header_is_optional_and_scoped(
    fake_server: tuple[FakeLlamaCppServer, str],
) -> None:
    server, base_url = fake_server
    server.single_response = {"data": [{"id": "model"}]}
    server.single_props = _props()
    async with aiohttp.ClientSession() as session:
        without_key = LlamaCppTransport(base_url=base_url, api_key="", session=session)
        await without_key.fetch_models()
        assert all(auth is None for _, _, _, auth in server.requests)

        server.requests.clear()
        with_key = LlamaCppTransport(base_url=base_url, api_key="secret", session=session)
        await with_key.fetch_models()
        assert all(auth == "Bearer secret" for _, _, _, auth in server.requests)


async def test_training_context_and_conservative_output_are_fallbacks(
    fake_server: tuple[FakeLlamaCppServer, str],
    transport: LlamaCppTransport,
) -> None:
    server, _ = fake_server
    server.single_response = {"data": [{"id": "small", "meta": {"n_ctx_train": 3_000}}]}
    server.single_props = _props(n_ctx=0, n_predict=-1)

    await transport.fetch_models()

    assert transport.model.context_window == 3_000
    assert transport.model.max_output_tokens == 3_000


async def test_missing_output_and_template_metadata_does_not_imply_text(
    fake_server: tuple[FakeLlamaCppServer, str],
    transport: LlamaCppTransport,
) -> None:
    server, _ = fake_server
    server.single_response = {"data": [{"id": "unknown-kind"}]}
    server.single_props = {
        "default_generation_settings": {"n_ctx": 8_192, "params": {"n_predict": -1}},
    }

    await transport.fetch_models()

    assert transport.model.capabilities == frozenset()


async def test_explicit_text_output_adds_text_capability(
    fake_server: tuple[FakeLlamaCppServer, str],
    transport: LlamaCppTransport,
) -> None:
    server, _ = fake_server
    server.single_response = {
        "data": [
            {
                "id": "completion",
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            }
        ]
    }
    server.single_props = {
        "default_generation_settings": {"n_ctx": 8_192, "params": {"n_predict": -1}},
    }

    await transport.fetch_models()

    assert transport.model.capabilities == frozenset({Capability.text})


async def test_router_discovery_only_publishes_usable_models_without_side_effects(
    fake_server: tuple[FakeLlamaCppServer, str],
    transport: LlamaCppTransport,
) -> None:
    server, _ = fake_server
    server.router_response = {
        "data": [
            {
                "id": "z-loaded",
                "status": {"value": "loaded"},
                "meta": {"n_ctx": 12_000, "n_ctx_train": 100_000},
                "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            },
            {"id": "a-sleeping", "status": {"value": "sleeping"}},
            {"id": "unloaded", "status": {"value": "unloaded"}},
            {"id": "loading", "status": {"value": "loading"}},
            {"id": "downloading", "status": {"value": "downloading"}},
        ]
    }
    server.model_props = {
        "z-loaded": _props(n_ctx=24_000, tools=True, tool_calls=True),
        "a-sleeping": _props(n_ctx=4_000, n_predict=5_000),
    }

    await transport.fetch_models()

    assert transport.models.ids() == ["a-sleeping", "z-loaded"]
    assert transport.model.id == "a-sleeping"
    assert transport.models["a-sleeping"].max_output_tokens == 4_000
    assert transport.models["z-loaded"].context_window == 24_000
    assert Capability.vision in transport.models["z-loaded"].capabilities
    props_requests = [request for request in server.requests if request[1] == "/props" and "model" in request[2]]
    assert {query["model"] for _, _, query, _ in props_requests} == {"a-sleeping", "z-loaded"}
    assert all(query == {"model": query["model"], "autoload": "false"} for _, _, query, _ in props_requests)
    assert all("reload" not in query for _, _, query, _ in server.requests)
    assert all(method == "GET" for method, _, _, _ in server.requests)


async def test_tool_use_requires_both_published_template_capabilities(
    fake_server: tuple[FakeLlamaCppServer, str],
    transport: LlamaCppTransport,
) -> None:
    server, _ = fake_server
    server.router_props = {}
    server.router_response = {
        "data": [
            {"id": "both", "status": {"value": "loaded"}},
            {"id": "tools-only", "status": {"value": "loaded"}},
            {"id": "calls-only", "status": {"value": "loaded"}},
        ]
    }
    server.model_props = {
        "both": _props(tools=True, tool_calls=True),
        "tools-only": _props(tools=True),
        "calls-only": _props(tool_calls=True),
    }

    await transport.fetch_models()

    assert Capability.tool_use in transport.models["both"].capabilities
    assert Capability.tool_use not in transport.models["tools-only"].capabilities
    assert Capability.tool_use not in transport.models["calls-only"].capabilities


async def test_explicit_embedding_output_does_not_claim_completion_text(
    fake_server: tuple[FakeLlamaCppServer, str],
    transport: LlamaCppTransport,
) -> None:
    server, _ = fake_server
    server.router_response = {
        "data": [
            {
                "id": "embed",
                "status": {"value": "loaded"},
                "architecture": {"input_modalities": ["text"], "output_modalities": ["embedding"]},
            }
        ]
    }
    server.model_props = {"embed": _props()}

    await transport.fetch_models()

    assert transport.model.capabilities == frozenset({Capability.embedding})


async def test_failed_discovery_preserves_registry_and_active_model(
    fake_server: tuple[FakeLlamaCppServer, str],
) -> None:
    server, base_url = fake_server
    old = ModelSpec(id="old", context_window=1_000, max_output_tokens=100)
    registry = ModelRegistry([old])
    server.router_response = {
        "data": [
            {"id": "good", "status": {"value": "loaded"}},
            {"id": "bad", "status": {"value": "loaded"}},
        ]
    }
    server.model_props = {"good": _props()}
    server.props_status["bad"] = 500

    async with aiohttp.ClientSession() as session:
        transport = LlamaCppTransport(base_url=base_url, models=registry, model=old, session=session)
        with pytest.raises(StreamError, match="500"):
            await transport.fetch_models()

    assert transport.models is registry
    assert transport.models.ids() == ["old"]
    assert transport.model is old


async def test_no_usable_router_model_is_an_atomic_error(
    fake_server: tuple[FakeLlamaCppServer, str],
) -> None:
    server, base_url = fake_server
    old = ModelSpec(id="old")
    registry = ModelRegistry([old])
    server.router_response = {"data": [{"id": "cold", "status": {"value": "unloaded"}}]}

    async with aiohttp.ClientSession() as session:
        transport = LlamaCppTransport(base_url=base_url, models=registry, model=old, session=session)
        with pytest.raises(StreamError, match="no usable models"):
            await transport.fetch_models()

    assert transport.models.ids() == ["old"]
    assert transport.model is old
    assert server.requests == [
        ("GET", "/props", {}, None),
        ("GET", "/models", {}, None),
    ]


async def test_refresh_preserves_active_model_by_id(
    fake_server: tuple[FakeLlamaCppServer, str],
) -> None:
    server, base_url = fake_server
    old = ModelSpec(id="z-model", context_window=1_000)
    registry = ModelRegistry([old])
    server.router_response = {
        "data": [
            {"id": "a-model", "status": {"value": "loaded"}},
            {"id": "z-model", "status": {"value": "loaded"}},
        ]
    }
    server.model_props = {"a-model": _props(), "z-model": _props(n_ctx=9_000)}

    async with aiohttp.ClientSession() as session:
        transport = LlamaCppTransport(base_url=base_url, models=registry, model=old, session=session)
        await transport.fetch_models()

    assert transport.models is registry
    assert transport.model is registry["z-model"]
    assert transport.model.context_window == 9_000


async def test_shared_registry_consumer_resyncs_refreshed_model_metadata(
    fake_server: tuple[FakeLlamaCppServer, str],
) -> None:
    server, base_url = fake_server
    old = ModelSpec(
        id="shared",
        context_window=16_000,
        max_output_tokens=4_000,
        capabilities=frozenset({Capability.text}),
    )
    registry = ModelRegistry([old])
    server.router_response = {
        "data": [
            {
                "id": "shared",
                "status": {"value": "loaded"},
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            }
        ]
    }
    server.model_props = {"shared": _props(n_ctx=2_000, n_predict=512)}

    async with aiohttp.ClientSession() as session:
        refresher = LlamaCppTransport(base_url=base_url, models=registry, model=old, session=session)
        consumer = LlamaCppTransport(base_url=base_url, models=registry, model=old, session=session)
        await refresher.fetch_models()
        payload = consumer.build_payload([], [], "")

    assert consumer.model is registry["shared"]
    assert consumer.model.context_window == 2_000
    assert consumer.model.max_output_tokens == 512
    assert payload["max_completion_tokens"] == 512


def test_serialization_round_trip_preserves_active_model_and_empty_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    first = ModelSpec(id="first", capabilities=frozenset({Capability.text}))
    active = ModelSpec(id="active", context_window=32_000, max_output_tokens=2_000)
    transport = LlamaCppTransport(
        base_url="http://llama.local/v1",
        api_key="",
        models=ModelRegistry([first, active]),
        model=active,
    )

    restored = LlamaCppTransport.from_dict(transport.to_dict())

    assert restored.base_url == "http://llama.local/v1"
    assert restored.api_key == ""
    assert restored.model is restored.models["active"]


async def test_subclass_uses_inherited_chat_stream_and_reasoning_parser(
    fake_server: tuple[FakeLlamaCppServer, str],
    transport: LlamaCppTransport,
) -> None:
    server, _ = fake_server
    server.single_response = {"data": [{"id": "chat"}]}
    server.single_props = _props()
    await transport.fetch_models()

    events = await _collect(transport.stream([], [], ""))

    assert [event.delta for event in events if isinstance(event, ReasoningDelta)] == ["think"]
    assert [event.delta for event in events if isinstance(event, TextDelta)] == ["done"]
    assert len([event for event in events if isinstance(event, IterationEnd)]) == 1
    assert server.completion_payloads[0]["model"] == "chat"
    assert [(method, path) for method, path, _, _ in server.requests[:3]] == [
        ("GET", "/props"),
        ("GET", "/models"),
        ("GET", "/v1/models"),
    ]
    assert server.requests[3][0:2] == ("POST", "/v1/chat/completions")


def test_tui_export_and_entry_points() -> None:
    assert LlamaCppSettingsScreen.DEFAULT_BASE_URL == LLAMA_CPP_BASE_URL
    package_root = Path(__file__).parents[1]
    project = tomllib.loads((package_root / "pyproject.toml").read_text("utf-8"))
    assert project["project"]["entry-points"]["axio.transport"]["llama-cpp"].endswith(":LlamaCppTransport")
    assert project["project"]["entry-points"]["axio.transport.settings"]["llama-cpp"].endswith(
        ":LlamaCppSettingsScreen"
    )


async def test_tui_settings_defaults_and_preserves_explicit_empty_key() -> None:
    app = App[None]()
    saved: list[dict[str, str] | None] = []
    screen = LlamaCppSettingsScreen({})

    async with app.run_test() as pilot:
        await app.push_screen(screen, saved.append)
        assert screen.query_one("#base-url", Input).value == LLAMA_CPP_BASE_URL
        assert screen.query_one("#api-key", Input).value == ""
        await pilot.click("#btn-save")
        await pilot.pause()

    assert saved == [{"base_url": LLAMA_CPP_BASE_URL, "api_key": ""}]
