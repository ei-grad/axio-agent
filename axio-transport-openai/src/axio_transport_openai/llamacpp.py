"""llama.cpp transport with side-effect-free native model discovery."""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import aiohttp
from axio.events import StreamEvent
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool

from axio_transport_openai import OpenAITransport

logger = logging.getLogger(__name__)

LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_CONTEXT_WINDOW = 4_096
DEFAULT_MAX_OUTPUT_TOKENS = 4_096

_UNSET_MODEL = ModelSpec(
    id="<not initialized: call fetch_models() first>",
    context_window=0,
    max_output_tokens=0,
    pricing_available=False,
)
_USABLE_ROUTER_STATES = frozenset({"loaded", "sleeping"})


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _modalities(entry: Mapping[str, Any], props: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    architecture = _as_mapping(entry.get("architecture"))
    raw_inputs = architecture.get("input_modalities", [])
    raw_outputs = architecture.get("output_modalities", [])
    inputs = {str(value).casefold() for value in raw_inputs if isinstance(value, str)}
    outputs = {str(value).casefold() for value in raw_outputs if isinstance(value, str)}

    published = _as_mapping(props.get("modalities"))
    for name in ("vision", "image", "audio", "video"):
        if published.get(name) is True:
            inputs.add("image" if name == "vision" else name)

    raw_capabilities = entry.get("capabilities")
    if isinstance(raw_capabilities, list):
        explicit = {str(value).casefold() for value in raw_capabilities if isinstance(value, str)}
        if "embedding" in explicit:
            outputs.add("embedding")
    elif isinstance(raw_capabilities, Mapping) and raw_capabilities.get("embedding") is True:
        outputs.add("embedding")

    return inputs, outputs


def _capabilities(entry: Mapping[str, Any], props: Mapping[str, Any]) -> frozenset[Capability]:
    inputs, outputs = _modalities(entry, props)
    caps: set[Capability] = set()

    template_caps = _as_mapping(props.get("chat_template_caps"))
    if not template_caps:
        template_caps = _as_mapping(entry.get("chat_template_caps"))
    chat_template = props.get("chat_template")
    has_chat_template = isinstance(chat_template, str) and bool(chat_template.strip())
    has_content_template = (
        template_caps.get("supports_string_content") is True or template_caps.get("supports_typed_content") is True
    )

    if "text" in outputs or (not outputs and (has_chat_template or has_content_template)):
        caps.add(Capability.text)
    if "image" in inputs or "vision" in inputs:
        caps.add(Capability.vision)
    if "audio" in inputs:
        caps.add(Capability.audio)
    if "video" in inputs:
        caps.add(Capability.video)
    if "image" in outputs:
        caps.add(Capability.image_generation)
    if "video" in outputs:
        caps.add(Capability.video_generation)
    if "embedding" in outputs:
        caps.add(Capability.embedding)

    if template_caps.get("supports_tools") is True and template_caps.get("supports_tool_calls") is True:
        caps.add(Capability.tool_use)

    return frozenset(caps)


def _parse_model(entry: Mapping[str, Any], props: Mapping[str, Any]) -> ModelSpec:
    model_id = entry.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise StreamError("llama.cpp model entry has no usable id")

    meta = _as_mapping(entry.get("meta"))
    generation = _as_mapping(props.get("default_generation_settings"))
    params = _as_mapping(generation.get("params"))
    context_window = (
        _positive_int(generation.get("n_ctx"))
        or _positive_int(meta.get("n_ctx"))
        or _positive_int(entry.get("n_ctx"))
        or _positive_int(meta.get("n_ctx_train"))
        or _positive_int(entry.get("n_ctx_train"))
        or DEFAULT_CONTEXT_WINDOW
    )
    n_predict = _positive_int(params.get("n_predict"))
    max_output_tokens = min(n_predict or DEFAULT_MAX_OUTPUT_TOKENS, context_window)

    return ModelSpec(
        id=model_id,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        capabilities=_capabilities(entry, props),
        input_cost=0.0,
        output_cost=0.0,
        pricing_available=False,
    )


@dataclass(slots=True)
class LlamaCppTransport(OpenAITransport):
    """OpenAI chat streaming backed by side-effect-free llama.cpp discovery.

    Router entries that are not already loaded (or sleeping) are deliberately
    excluded. ``ModelSpec`` has no availability state, so including them would
    make the TUI present models whose first completion can silently autoload
    them. Axio never loads, unloads, downloads, or refreshes llama.cpp's cache.
    """

    name: str = "llama.cpp"
    base_url: str = field(default_factory=lambda: os.environ.get("LLAMA_CPP_BASE_URL", LLAMA_CPP_BASE_URL))
    api_key: str = field(default_factory=lambda: os.environ.get("LLAMA_API_KEY", ""))
    model: ModelSpec = field(default_factory=lambda: _UNSET_MODEL)
    models: ModelRegistry = field(default_factory=ModelRegistry)

    @property
    def native_base_url(self) -> str:
        base = self.base_url.rstrip("/")
        return base[:-3] if base.endswith("/v1") else base

    def _sync_active_model(self) -> bool:
        try:
            current = self.models[self.model.id]
        except KeyError:
            return False
        self.model = current
        return current.context_window > 0 and current.max_output_tokens > 0

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        if not self._sync_active_model():
            raise RuntimeError("LlamaCppTransport: call fetch_models() before streaming")
        return OpenAITransport.stream(self, messages, tools, system)

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        if not self._sync_active_model():
            raise RuntimeError("LlamaCppTransport: call fetch_models() before building a payload")
        return super().build_payload(messages, tools, system)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _get_json(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        assert self.session is not None, "session is required for fetch_models"
        async with self.session.get(url, params=params, headers=self._headers()) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise StreamError(f"llama.cpp API error {resp.status} from {url}: {body}")
            payload = await resp.json()
        if not isinstance(payload, dict):
            raise StreamError(f"llama.cpp API returned a non-object response from {url}")
        return payload

    async def fetch_models(self) -> None:
        """Discover loaded models without changing router or model state."""
        props_url = f"{self.native_base_url}/props"
        server_props = await self._get_json(props_url)
        native_models_url = f"{self.native_base_url}/models"
        native_models = await self._get_json(native_models_url)
        native_entries = native_models.get("data", [])
        if not isinstance(native_entries, list):
            raise StreamError("llama.cpp /models response has a non-list data field")

        is_router = server_props.get("role") == "router" or any(
            isinstance(_as_mapping(entry).get("status"), Mapping) for entry in native_entries
        )

        parsed: list[ModelSpec] = []
        if not is_router:
            models_url = f"{self.base_url.rstrip('/')}/models"
            models_payload = await self._get_json(models_url)
            entries = models_payload.get("data", [])
            if not isinstance(entries, list):
                raise StreamError("llama.cpp /v1/models response has a non-list data field")
            parsed = [_parse_model(_as_mapping(entry), server_props) for entry in entries]
            mode = "single-model"
        else:
            for raw_entry in native_entries:
                entry = _as_mapping(raw_entry)
                status = _as_mapping(entry.get("status")).get("value")
                if status not in _USABLE_ROUTER_STATES:
                    continue
                model_id = entry.get("id")
                if not isinstance(model_id, str) or not model_id:
                    raise StreamError("llama.cpp router model entry has no usable id")
                props = await self._get_json(
                    f"{self.native_base_url}/props",
                    params={"model": model_id, "autoload": "false"},
                )
                parsed.append(_parse_model(entry, props))
            mode = "router"

        if not parsed:
            raise StreamError(f"llama.cpp {mode} discovery returned no usable models")

        new_models = ModelRegistry(sorted(parsed, key=lambda spec: spec.id))
        selected_id = self.model.id if self.model.id in new_models else new_models.first().id
        selected_model = new_models[selected_id]

        # Keep the shared registry object used by TUI-created transports. There
        # are no await points between clear and update, so readers cannot observe
        # a partial replacement through the event loop.
        self.models.clear()
        self.models.update(new_models.items())
        self.model = selected_model
        logger.info("Loaded %d usable models from llama.cpp (%s mode)", len(self.models), mode)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.model.id in self.models:
            data["model"] = self.model.id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, session: aiohttp.ClientSession | None = None) -> Self:
        obj = super().from_dict(data, session=session)
        obj.models = ModelRegistry(
            dataclasses.replace(model, pricing_available=False) for model in obj.models.values()
        )
        obj = dataclasses.replace(
            obj,
            name=str(data.get("name", "llama.cpp")),
            base_url=str(data.get("base_url", LLAMA_CPP_BASE_URL)),
            api_key=str(data.get("api_key", "")),
        )
        model_id = data.get("model")
        if isinstance(model_id, str) and model_id in obj.models:
            obj.model = obj.models[model_id]
        elif obj.models:
            obj.model = obj.models.first()
        else:
            obj.model = _UNSET_MODEL
        return obj
