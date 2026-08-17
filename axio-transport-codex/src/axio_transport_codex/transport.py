"""ChatGPT (Codex) transport - Responses API over SSE."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import platform
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from axio.blocks import AudioBlock, ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock, VideoBlock
from axio.effort import EFFORT_LEVELS, EffortLevel, EffortMechanism, EffortState, PromptEffortAdapter, parse_effort
from axio.events import IterationEnd, ReasoningDelta, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from axio.exceptions import StreamError
from axio.messages import (
    INPUT_PROVENANCE_FOOTER,
    Message,
    effective_input_provenance,
    input_provenance_header,
    model_visible_content,
)
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.transport import CompletionTransport
from axio.types import StopReason, Usage

from .oauth import CLIENT_ID, ORIGINATOR, TOKEN_URL, _decode_jwt_payload

logger = logging.getLogger(__name__)

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_USER_AGENT = f"codex_cli_rs/0.1.0 ({platform.system()} {platform.release()}; {platform.machine()})"

_VT = frozenset({Capability.text, Capability.vision, Capability.tool_use})
_RT = frozenset({Capability.text, Capability.reasoning, Capability.tool_use})
_TT = frozenset({Capability.text, Capability.tool_use})

CODEX_MODELS: ModelRegistry = ModelRegistry(
    {
        ModelSpec(id="o4-mini", context_window=200_000, max_output_tokens=100_000, capabilities=_RT),
        ModelSpec(id="gpt-4.1", context_window=1_047_576, max_output_tokens=32_768, capabilities=_VT),
        ModelSpec(id="gpt-4.1-mini", context_window=1_047_576, max_output_tokens=32_768, capabilities=_VT),
        ModelSpec(id="gpt-4.1-nano", context_window=1_047_576, max_output_tokens=32_768, capabilities=_TT),
        ModelSpec(id="gpt-4o", context_window=128_000, max_output_tokens=16_384, capabilities=_VT),
        ModelSpec(id="gpt-4o-mini", context_window=128_000, max_output_tokens=16_384, capabilities=_VT),
        ModelSpec(id="o3", context_window=200_000, max_output_tokens=100_000, capabilities=_RT),
        ModelSpec(id="o3-mini", context_window=200_000, max_output_tokens=100_000, capabilities=_RT),
    }
)

_STOP_REASON_MAP: dict[str, StopReason] = {
    "completed": StopReason.end_turn,
    "end_turn": StopReason.end_turn,
    "stop": StopReason.end_turn,
    "max_output_tokens": StopReason.max_tokens,
    "length": StopReason.max_tokens,
}


def _strip_title(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove pydantic 'title' keys from a JSON schema recursively."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if isinstance(value, dict):
            out[key] = _strip_title(value)
        elif isinstance(value, list):
            out[key] = [_strip_title(item) if isinstance(item, dict) else item for item in value]
        else:
            out[key] = value
    return out


def _convert_tools(tools: list[Tool[Any]]) -> list[dict[str, Any]]:
    """Convert axio Tool list to Responses API function tool dicts."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": _strip_title(tool.input_schema),
        }
        for tool in tools
    ]


def _convert_messages(messages: list[Message], system: str) -> tuple[str, list[dict[str, Any]]]:
    """Convert axio Message list to Responses API input array.

    Returns (instructions, input_items).
    """
    items: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            if tool_results:
                tool_image_parts: list[dict[str, Any]] = []
                for tr in tool_results:
                    if isinstance(tr.content, str):
                        content = tr.content
                        images: list[ImageBlock] = []
                    else:
                        unsupported = [block for block in tr.content if isinstance(block, (AudioBlock, VideoBlock))]
                        if unsupported:
                            media = ", ".join(sorted({block.media_type for block in unsupported}))
                            raise ValueError(f"Codex Responses does not support tool-result media: {media}")
                        content = "\n".join(block.text for block in tr.content if isinstance(block, TextBlock))
                        images = [block for block in tr.content if isinstance(block, ImageBlock)]
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": tr.tool_use_id,
                            "output": content,
                        }
                    )
                    if images:
                        tool_image_parts.append(
                            {"type": "input_text", "text": f"[Image from tool call {tr.tool_use_id}]"}
                        )
                        for image in images:
                            encoded = base64.b64encode(image.data).decode("ascii")
                            tool_image_parts.append(
                                {"type": "input_image", "image_url": f"data:{image.media_type};base64,{encoded}"}
                            )
                if tool_image_parts:
                    provenance = effective_input_provenance(msg)
                    tool_image_parts.insert(
                        0,
                        {"type": "input_text", "text": input_provenance_header(provenance)},
                    )
                    tool_image_parts.append({"type": "input_text", "text": INPUT_PROVENANCE_FOOTER})
                    items.append({"role": "user", "content": tool_image_parts})
                remaining_blocks = [b for b in model_visible_content(msg) if not isinstance(b, ToolResultBlock)]
            else:
                remaining_blocks = model_visible_content(msg)
            content_parts: list[dict[str, Any]] = []
            for b in remaining_blocks:
                if isinstance(b, TextBlock):
                    content_parts.append({"type": "input_text", "text": b.text})
                elif isinstance(b, ImageBlock):
                    encoded = base64.b64encode(b.data).decode("ascii")
                    data_uri = f"data:{b.media_type};base64,{encoded}"
                    content_parts.append({"type": "input_image", "image_url": data_uri})
            if content_parts:
                items.append({"role": "user", "content": content_parts})

        elif msg.role == "assistant":
            # Collect text and tool uses
            content_parts_a: list[dict[str, Any]] = []
            for b in msg.content:
                if isinstance(b, TextBlock):
                    content_parts_a.append({"type": "output_text", "text": b.text})
                elif isinstance(b, ToolUseBlock):
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": b.id,
                            "name": b.name,
                            "arguments": json.dumps(b.input),
                            "status": "completed",
                        }
                    )
            if content_parts_a:
                items.insert(
                    len(items) - sum(1 for b in msg.content if isinstance(b, ToolUseBlock)),
                    {
                        "role": "assistant",
                        "content": content_parts_a,
                    },
                )

    # Synthesize placeholder outputs for orphan function_calls (no corresponding output)
    output_ids = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    for item in list(items):
        if item.get("type") == "function_call" and item.get("call_id") not in output_ids:
            call_id = item.get("call_id", "")
            logger.warning("Synthesizing placeholder output for orphan function_call: call_id=%s", call_id)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "[Tool was not executed - context was interrupted or compacted]",
                }
            )

    return system, items


@dataclass(slots=True)
class CodexTransport(CompletionTransport):
    name: str = "ChatGPT (Codex)"
    api_key: str = ""
    refresh_token: str = ""
    expires_at: str = ""
    account_id: str = ""
    base_url: str = CODEX_BASE_URL
    model: ModelSpec = field(default_factory=lambda: CODEX_MODELS["o4-mini"])
    models: ModelRegistry = field(default_factory=lambda: ModelRegistry(CODEX_MODELS.values()))
    session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)
    on_auth_refresh: Callable[[dict[str, str]], Awaitable[None]] | None = field(
        default=None, repr=False, compare=False
    )
    max_retries: int = 10
    retry_base_delay: float = 5.0
    reasoning_effort: str | None = field(default=None, repr=False)
    _reasoning_efforts: dict[str, tuple[EffortLevel, ...]] = field(default_factory=dict, repr=False)

    def _get_retry_delay(self, resp: aiohttp.ClientResponse | None, attempt: int) -> float:
        """Return delay in seconds: prefer Retry-After header, fall back to exponential backoff."""
        if resp is not None:
            retry_after: str | None = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return max(0.0, float(retry_after))
                except (ValueError, TypeError):
                    pass
        return float(self.retry_base_delay * (2 ** (attempt - 1)))

    def configure_effort(self, requested: str | None) -> EffortState:
        level = parse_effort(requested)
        supported = self._reasoning_efforts.get(self.model.id, ())
        if level is None:
            self.reasoning_effort = None
            mechanism = EffortMechanism.native_effort if supported else EffortMechanism.prompt_fallback
            allowed = supported if mechanism is EffortMechanism.native_effort else EFFORT_LEVELS
            return EffortState(None, mechanism, allowed=allowed)
        if supported:
            if level not in supported:
                raise ValueError(
                    f"Effort {level!r} is not supported by {self.model.id}. Valid values: {', '.join(supported)}"
                )
            self.reasoning_effort = level
            return EffortState(level, EffortMechanism.native_effort, provider_value=level, allowed=supported)
        self.reasoning_effort = None
        return PromptEffortAdapter().configure_effort(level)

    async def _ensure_token(self) -> None:
        """Refresh access token if expired or about to expire."""
        if not self.refresh_token or not self.expires_at:
            return
        try:
            expires_at = int(self.expires_at)
        except ValueError:
            return
        if time.time() < expires_at - 30:
            return

        logger.info("Access token expired or expiring soon, refreshing...")
        await self._refresh()

    async def _refresh(self) -> None:
        """Refresh the access token using the refresh token."""
        payload = {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": self.refresh_token,
        }

        async with aiohttp.ClientSession() as sess:
            async with sess.post(TOKEN_URL, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise StreamError(f"Token refresh failed ({resp.status}): {body}")
                data: dict[str, Any] = await resp.json()

        self.api_key = data["access_token"]
        self.refresh_token = data.get("refresh_token", self.refresh_token)
        expires_in: int = data.get("expires_in", 3600)
        self.expires_at = str(int(time.time()) + expires_in)

        jwt_payload = _decode_jwt_payload(self.api_key)
        orgs = jwt_payload.get("organizations", [])
        if orgs and isinstance(orgs, list) and isinstance(orgs[0], dict):
            self.account_id = orgs[0].get("id", self.account_id)

        logger.info("Token refreshed, expires_at=%s", self.expires_at)

        if self.on_auth_refresh is not None:
            try:
                await self.on_auth_refresh(
                    {
                        "api_key": self.api_key,
                        "refresh_token": self.refresh_token,
                        "expires_at": self.expires_at,
                        "account_id": self.account_id,
                    }
                )
            except Exception:
                logger.warning("Failed to persist refreshed tokens", exc_info=True)

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        instructions, input_items = _convert_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self.model.id,
            "input": input_items,
            "stream": True,
            "store": False,
        }
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = _convert_tools(tools)
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        # Log input items summary for debugging
        fc = [i for i in input_items if i.get("type") == "function_call"]
        fco = [i for i in input_items if i.get("type") == "function_call_output"]
        if fc or fco:
            fc_ids = [i.get("call_id") for i in fc]
            fco_ids = [i.get("call_id") for i in fco]
            logger.info("Input: %d function_calls %s, %d outputs %s", len(fc), fc_ids, len(fco), fco_ids)

        return payload

    async def _parse_sse(self, resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
        """Parse Responses API SSE events into axio StreamEvents."""
        usage = Usage(0, 0)
        stop_reason: StopReason | None = None
        # Map item_id → call_id so ToolInputDelta uses the same ID as ToolUseStart
        item_to_call: dict[str, str] = {}

        buffer = b""
        async for chunk in resp.content.iter_any():
            buffer += chunk
            while b"\n" in buffer:
                # Yield to event loop between SSE lines to keep the TUI responsive.
                # Codex sends many event types that don't produce StreamEvents,
                # so without this the inner loop can starve the event loop.
                await asyncio.sleep(0)
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue

                raw = line[6:]
                if raw == "[DONE]":
                    continue

                try:
                    data: dict[str, Any] = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse SSE JSON: %s", raw[:200])
                    continue

                event_type = data.get("type", "")

                if event_type == "response.output_text.delta":
                    yield TextDelta(index=0, delta=data.get("delta", ""))

                elif event_type == "response.reasoning_summary_text.delta":
                    yield ReasoningDelta(index=0, delta=data.get("delta", ""))

                elif event_type == "response.output_item.added":
                    item = data.get("item", {})
                    item_type = item.get("type", "")
                    if item_type == "function_call":
                        call_id = item.get("call_id", "")
                        item_id = item.get("id", "")
                        name = item.get("name", "")
                        if item_id:
                            item_to_call[item_id] = call_id
                        logger.info("Tool call started: %s (call_id=%s, item_id=%s)", name, call_id, item_id)
                        yield ToolUseStart(
                            index=data.get("output_index", 0),
                            tool_use_id=call_id,
                            name=name,
                        )
                    else:
                        logger.info("Output item added: type=%s", item_type)

                elif event_type == "response.function_call_arguments.delta":
                    item_id = data.get("item_id", "")
                    resolved_id = item_to_call.get(item_id, item_id)
                    delta = data.get("delta", "")
                    logger.debug("Tool args delta: call_id=%s, +%d chars", resolved_id, len(delta))
                    yield ToolInputDelta(
                        index=data.get("output_index", 0),
                        tool_use_id=resolved_id,
                        partial_json=delta,
                    )

                elif event_type == "response.function_call_arguments.done":
                    item_id = data.get("item_id", "")
                    resolved_id = item_to_call.get(item_id, item_id)
                    args = data.get("arguments", "")
                    logger.info("Tool args complete: call_id=%s, args=%s", resolved_id, args[:200])

                elif event_type == "response.completed":
                    response = data.get("response", {})
                    resp_usage = response.get("usage", {})
                    usage = Usage(
                        input_tokens=resp_usage.get("input_tokens", 0),
                        output_tokens=resp_usage.get("output_tokens", 0),
                    )
                    status = response.get("status", "completed")
                    stop_reason = _STOP_REASON_MAP.get(status, StopReason.end_turn)
                    output = response.get("output", [])
                    if any(item.get("type") == "function_call" for item in output):
                        stop_reason = StopReason.tool_use
                    logger.info(
                        "Response completed: status=%s, stop=%s, in=%d, out=%d",
                        status,
                        stop_reason,
                        usage.input_tokens,
                        usage.output_tokens,
                    )

                elif event_type == "response.failed":
                    response = data.get("response", {})
                    err = response.get("error", {})
                    msg = err.get("message", "Unknown error")
                    logger.error("Response failed: %s", msg)
                    raise StreamError(f"Codex API error: {msg}")

        if stop_reason is None:
            stop_reason = StopReason.end_turn

        logger.debug(
            "Stream complete: stop_reason=%s, input_tokens=%d, output_tokens=%d",
            stop_reason,
            usage.input_tokens,
            usage.output_tokens,
        )
        yield IterationEnd(iteration=0, stop_reason=stop_reason, usage=usage)

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        return self._do_stream(messages, tools, system)

    async def _do_stream(
        self, messages: list[Message], tools: list[Tool[Any]], system: str
    ) -> AsyncIterator[StreamEvent]:
        assert self.session is not None, "session is required for streaming"

        logger.debug("Stream start: model=%s, messages=%d, tools=%d", self.model.id, len(messages), len(tools))

        await self._ensure_token()
        logger.debug("Token check passed")

        url = f"{self.base_url}/responses"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": CODEX_USER_AGENT,
            "originator": ORIGINATOR,
        }
        if self.account_id:
            headers["ChatGPT-Account-ID"] = self.account_id

        payload = self.build_payload(messages, tools, system)
        logger.debug("Payload built: %d input items", len(payload.get("input", [])))

        if logger.getEffectiveLevel() <= logging.DEBUG:
            dumped = json.dumps(payload, indent=2)
            if len(dumped) > 4000:
                dumped = dumped[:4000] + f"\n... truncated ({len(dumped)} chars total)"
            logger.debug("Request payload:\n%s", dumped)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            retry_resp: aiohttp.ClientResponse | None = None
            try:
                logger.debug("POST %s (attempt %d/%d)", url, attempt, self.max_retries)
                async with self.session.post(url, json=payload, headers=headers) as resp:
                    logger.debug("HTTP response: status=%d content_type=%s", resp.status, resp.content_type)
                    if resp.status == 200:
                        logger.debug("SSE parsing started")
                        async for event in self._parse_sse(resp):
                            yield event
                        logger.debug("SSE parsing finished")
                        return

                    body = await resp.text()
                    if resp.status == 429 or resp.status >= 500:
                        retry_resp = resp
                        last_exc = StreamError(f"Codex API error {resp.status}: {body}")
                        logger.warning(
                            "Retryable HTTP %d (attempt %d/%d): %s",
                            resp.status,
                            attempt,
                            self.max_retries,
                            body,
                        )
                    else:
                        logger.error("HTTP %d from %s: %s", resp.status, url, body)
                        raise StreamError(f"Codex API error {resp.status}: {body}")
            except aiohttp.ClientError as exc:
                last_exc = StreamError(str(exc))
                logger.warning("Connection error (attempt %d/%d): %s", attempt, self.max_retries, exc)

            if attempt < self.max_retries:
                delay = self._get_retry_delay(retry_resp, attempt)
                logger.info("Retrying in %.1fs...", delay)
                await asyncio.sleep(delay)

        raise last_exc or StreamError("Max retries exceeded")

    async def fetch_models(self) -> None:
        """Fetch available models from the Codex API."""
        if not self.api_key or self.session is None:
            self.models = ModelRegistry(CODEX_MODELS.values())
            return

        await self._ensure_token()

        url = f"{self.base_url}/models?client_version=0.0.1"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": CODEX_USER_AGENT,
            "originator": ORIGINATOR,
        }
        if self.account_id:
            headers["ChatGPT-Account-ID"] = self.account_id

        try:
            async with self.session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    logger.warning("Failed to fetch models (%d), using defaults", resp.status)
                    self.models = ModelRegistry(CODEX_MODELS.values())
                    return
                data: dict[str, Any] = await resp.json()
        except Exception:
            logger.warning("Failed to fetch models, using defaults", exc_info=True)
            self.models = ModelRegistry(CODEX_MODELS.values())
            return

        # Parse model list - the WHAM endpoint may return different formats
        model_list = data.get("data", data.get("models", []))
        if not model_list:
            self.models = ModelRegistry(CODEX_MODELS.values())
            return

        specs: list[ModelSpec] = []
        reasoning_efforts: dict[str, tuple[EffortLevel, ...]] = {}
        for item in model_list:
            model_id = item.get("id", item.get("slug", ""))
            if not model_id:
                continue
            advertised = item.get("supported_reasoning_efforts", item.get("supported_reasoning_levels", []))
            advertised_efforts = {
                str(option.get("effort", "")) if isinstance(option, dict) else str(option)
                for option in advertised
                if (isinstance(option, str) and option) or (isinstance(option, dict) and option.get("effort"))
            }
            parsed_efforts = tuple(level for level in EFFORT_LEVELS if level in advertised_efforts)
            if parsed_efforts:
                reasoning_efforts[model_id] = parsed_efforts
            # Use known spec if available, otherwise build one from API data.
            if model_id in CODEX_MODELS:
                specs.append(CODEX_MODELS[model_id])
            else:
                specs.append(
                    ModelSpec(
                        id=model_id,
                        capabilities=_RT if parsed_efforts else _TT,
                        context_window=item.get("context_window", 128_000),
                        max_output_tokens=item.get("max_output_tokens", 8_192),
                    )
                )

        self.models = ModelRegistry(specs) if specs else ModelRegistry(CODEX_MODELS.values())
        self._reasoning_efforts = reasoning_efforts

        # If the currently selected model was dropped from the API list, switch to
        # the first available one so the transport stays usable out of the box.
        if self.model.id not in self.models:
            self.model = self.models.first()
