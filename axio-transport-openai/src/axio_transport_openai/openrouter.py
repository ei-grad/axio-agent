"""OpenRouter CompletionTransport - inherits from OpenAI-compatible transport."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from axio.effort import EFFORT_LEVELS, EffortMechanism, EffortState, PromptEffortAdapter, parse_effort
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability, ModelSpec
from axio.tool import Tool

from axio_transport_openai import ChatCompletionsTransport, ThinkingMixin

logger = logging.getLogger(__name__)

OPENROUTER_BROKEN_TOOL_ARGUMENT_PROVIDERS = (
    "streamlake/fp8",
    "relace/fp4",
    "cloudflare",
)
_BROKEN_PROVIDER_IDENTITIES = frozenset(
    {
        "streamlake",
        "streamlake/fp8",
        "relace",
        "relace/fp4",
        "cloudflare",
    }
)


def _split_provider(model_id: str) -> tuple[str, str | None]:
    """Split ``model@provider`` into its parts, leaving a bare id untouched."""
    base, sep, provider = model_id.rpartition("@")
    if not sep or not base or not provider:
        return model_id, None
    return base, provider


def _provider_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"OpenRouter provider.{field_name} must be an array of non-empty strings")
    return list(value)


def _dedupe_providers(providers: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for provider in providers:
        identity = provider.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        result.append(provider)
    return result


def _is_known_broken_provider(provider: str) -> bool:
    return provider.casefold() in _BROKEN_PROVIDER_IDENTITIES


def _merge_provider_routing(raw: object, pinned_provider: str | None) -> dict[str, Any]:
    if raw is None:
        routing: dict[str, Any] = {}
    elif isinstance(raw, Mapping):
        routing = dict(raw)
    else:
        raise ValueError("OpenRouter provider routing must be an object")

    only = _dedupe_providers(_provider_list(routing["only"], "only")) if "only" in routing else []
    order = _dedupe_providers(_provider_list(routing["order"], "order")) if "order" in routing else []
    ignored = _dedupe_providers(_provider_list(routing["ignore"], "ignore")) if "ignore" in routing else []

    for field_name, providers in (("only", only), ("order", order)):
        broken = [provider for provider in providers if _is_known_broken_provider(provider)]
        if broken:
            names = ", ".join(broken)
            raise ValueError(f"OpenRouter provider.{field_name} selects excluded provider(s): {names}")

    if pinned_provider is not None:
        if _is_known_broken_provider(pinned_provider):
            raise ValueError(f"OpenRouter model provider pin selects excluded provider: {pinned_provider}")
        if only and [provider.casefold() for provider in only] != [pinned_provider.casefold()]:
            raise ValueError("OpenRouter model provider pin conflicts with provider.only")
        if order and any(provider.casefold() != pinned_provider.casefold() for provider in order):
            raise ValueError("OpenRouter model provider pin conflicts with provider.order")
        only = [pinned_provider]

    ignored = _dedupe_providers([*ignored, *OPENROUTER_BROKEN_TOOL_ARGUMENT_PROVIDERS])
    ignored_ids = {provider.casefold() for provider in ignored}
    only_ids = {provider.casefold() for provider in only}
    order_ids = {provider.casefold() for provider in order}
    if conflict := sorted(only_ids & ignored_ids):
        raise ValueError(f"OpenRouter provider.only conflicts with provider.ignore: {', '.join(conflict)}")
    if conflict := sorted(order_ids & ignored_ids):
        raise ValueError(f"OpenRouter provider.order conflicts with provider.ignore: {', '.join(conflict)}")
    if only_ids and not order_ids.issubset(only_ids):
        raise ValueError("OpenRouter provider.order must be a subset of provider.only when both are set")

    if only:
        routing["only"] = only
    if order:
        routing["order"] = order
    routing["ignore"] = ignored
    return routing


@dataclass(slots=True)
class OpenRouterTransport(ThinkingMixin, ChatCompletionsTransport):
    name: str = "OpenRouter"
    api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    base_url: str = "https://openrouter.ai/api/v1"
    model: ModelSpec = ModelSpec(id="google/gemini-2.5-pro-preview")
    thinking: bool = False
    _reasoning_efforts: dict[str, tuple[str, ...] | None] = field(default_factory=dict, repr=False)
    _reasoning_mandatory: set[str] = field(default_factory=set, repr=False)

    def _reasoning_metadata_id(self) -> str:
        model_id, _provider = _split_provider(self.model.id)
        if model_id in self._reasoning_efforts or model_id in self._reasoning_mandatory:
            return model_id
        base_id, separator, _tier = model_id.rpartition(":")
        if separator and (base_id in self._reasoning_efforts or base_id in self._reasoning_mandatory):
            return base_id
        return model_id

    def configure_effort(self, requested: str | None) -> EffortState:
        level = parse_effort(requested)
        supports_reasoning = Capability.reasoning in self.model.capabilities
        metadata_id = self._reasoning_metadata_id()
        selector_present = metadata_id in self._reasoning_efforts
        advertised = self._reasoning_efforts.get(metadata_id)
        allowed = (
            EFFORT_LEVELS
            if selector_present and advertised is None
            else tuple(item for item in EFFORT_LEVELS if advertised is not None and item in advertised)
        )
        if metadata_id in self._reasoning_mandatory:
            allowed = tuple(item for item in allowed if item != "none")
        if level is None:
            self.thinking = False
            self.reasoning_effort = None
            mechanism = (
                EffortMechanism.native_effort
                if supports_reasoning and selector_present
                else EffortMechanism.prompt_fallback
            )
            effective_allowed = allowed if mechanism is EffortMechanism.native_effort else EFFORT_LEVELS
            return EffortState(None, mechanism, allowed=effective_allowed)
        if supports_reasoning and selector_present:
            if level not in allowed:
                valid = ", ".join(allowed) or "none of the Axio effort levels"
                raise ValueError(f"Effort {level!r} is not supported by {self.model.id}. Valid values: {valid}")
            self.thinking = False
            self.reasoning_effort = level
            return EffortState(level, EffortMechanism.native_effort, provider_value=level, allowed=allowed)
        self.thinking = False
        self.reasoning_effort = None
        return PromptEffortAdapter().configure_effort(level)

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        # Python 3.12 slots dataclasses replace the class, so zero-argument super() captures the discarded class.
        payload = super(OpenRouterTransport, self).build_payload(messages, tools, system)  # noqa: UP008
        # ThinkingMixin asks for enable_thinking, which OpenRouter does not
        # understand: it takes a reasoning object and returns the trace in
        # delta.reasoning.
        if payload.pop("enable_thinking", None):
            payload.setdefault("reasoning", {"enabled": True})
        if self.reasoning_effort is not None and Capability.reasoning in self.model.capabilities:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        model_id, provider = _split_provider(str(payload.get("model", "")))
        payload["model"] = model_id
        payload["provider"] = _merge_provider_routing(payload.get("provider"), provider)
        return payload

    def _provider_cost_usd(self, usage: Mapping[str, Any]) -> float | None:
        if "cost" not in usage:
            return None
        value = usage["cost"]
        if isinstance(value, bool) or not isinstance(value, int | float):
            logger.warning("Ignoring OpenRouter usage.cost with non-numeric type %s", type(value).__name__)
            return None
        cost = float(value)
        if not math.isfinite(cost) or cost < 0:
            logger.warning("Ignoring non-finite or negative OpenRouter usage.cost")
            return None
        return cost

    def resolve_model(self, model_id: str) -> ModelSpec:
        """Resolve ``[<lab>/]<model>[:tier][@<provider>]``.

        The tier (``:nitro``, ``:free``) is OpenRouter's own routing suffix and
        the registry may not list the variant, so its metadata comes from the
        base model. The ``@provider`` suffix is ours: it is not part of any
        model id, it pins serving to one provider and is peeled off into
        ``provider.only`` when the request is built. Both stay in the resolved
        id so the selection survives a round trip through ``/model``.
        """
        base_id, provider = _split_provider(model_id)
        try:
            spec = self.models[base_id]
        except KeyError:
            stripped, sep, _tier = base_id.rpartition(":")
            if not sep or not stripped:
                raise KeyError(model_id) from None
            try:
                spec = self.models[stripped]
            except KeyError:
                raise KeyError(model_id) from None
        return spec if spec.id == model_id else replace(spec, id=model_id)

    async def fetch_models(self) -> None:
        """Fetch available models from OpenRouter ``/v1/models``."""
        assert self.session is not None, "session is required for fetch_models"
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self.session.get(url, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise StreamError(f"OpenRouter API error {resp.status}: {body}")
            payload: dict[str, Any] = await resp.json()

        self.models.clear()
        self._reasoning_efforts.clear()
        self._reasoning_mandatory.clear()
        for entry in payload.get("data", []):
            m = self._parse_model(entry)
            self.models[m.id] = m
            reasoning = entry.get("reasoning")
            if isinstance(reasoning, dict):
                if "supported_efforts" in reasoning:
                    efforts = reasoning["supported_efforts"]
                    if efforts is None:
                        self._reasoning_efforts[m.id] = None
                    elif isinstance(efforts, list):
                        self._reasoning_efforts[m.id] = tuple(str(item) for item in efforts)
                if reasoning.get("mandatory") is True:
                    self._reasoning_mandatory.add(m.id)
        logger.info("Loaded %d models from %s", len(self.models), url)

    @staticmethod
    def _parse_model(entry: dict[str, Any]) -> ModelSpec:
        caps: set[Capability] = set()

        params: list[str] = entry.get("supported_parameters", [])
        if "tools" in params:
            caps.add(Capability.tool_use)
        # OpenRouter advertises reasoning support here; without reading it the
        # capability is never set, so thinking is never requested or announced.
        if "reasoning" in params or "include_reasoning" in params or isinstance(entry.get("reasoning"), dict):
            caps.add(Capability.reasoning)

        arch: dict[str, Any] = entry.get("architecture", {})
        input_modalities: list[str] = arch.get("input_modalities", [])
        output_modalities: list[str] = arch.get("output_modalities", [])
        if "image" in input_modalities:
            caps.add(Capability.vision)
        if "embedding" in output_modalities:
            caps.add(Capability.embedding)

        top: dict[str, Any] = entry.get("top_provider", {})
        context_window = int(entry.get("context_length") or top.get("context_length") or 128_000)
        # 49 of the ~400 models in the catalogue report the whole context window
        # as their output limit, meaning "no cap of our own". Kept as reported:
        # what actually fits is decided per request against the prompt.
        max_output_tokens = int(top.get("max_completion_tokens") or 8_000)

        pricing: dict[str, Any] = entry.get("pricing", {})
        return ModelSpec(
            id=entry["id"],
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            capabilities=frozenset(caps),
            input_cost=float(pricing.get("prompt", 0)) * 1_000_000,
            output_cost=float(pricing.get("completion", 0)) * 1_000_000,
        )
