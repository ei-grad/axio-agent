"""OpenAI transport speaking the native ``/v1/responses`` protocol.

Not a preference: on chat/completions the newer models refuse to combine
function tools with a reasoning effort — "use /v1/responses or set
reasoning_effort to 'none'" — which for an agent means giving up either its
tools or the reasoning that makes the strong models worth using.

The wire format differs enough to be worth naming: messages become a flat
`input` array where tool calls and their outputs are items in their own right
rather than fields on a message, the system prompt moves to `instructions`, and
the stream is a sequence of named events instead of choice deltas.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar

import aiohttp
from axio.blocks import AudioBlock, ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock, VideoBlock
from axio.effort import EFFORT_LEVELS, EffortMechanism, EffortState, PromptEffortAdapter, parse_effort
from axio.events import IterationEnd, ReasoningDelta, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from axio.exceptions import StreamError
from axio.messages import Message
from axio.models import Capability
from axio.tool import Tool
from axio.types import StopReason, Usage

from .chat import _fit_output_limit, _openai_reasoning_efforts, _OpenAIHTTPTransport, _strip_title

logger = logging.getLogger(__name__)

_ORPHAN_OUTPUT = "[Tool was not executed - context was interrupted or compacted]"
_RESERVED_RESPONSE_PARAMS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "stream",
        "store",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "max_output_tokens",
    }
)


def _convert_tools(tools: list[Tool[Any]]) -> list[dict[str, Any]]:
    """Function tools are flat here — no nested "function" object."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": _strip_title(tool.input_schema),
        }
        for tool in tools
    ]


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            if tool_results:
                tool_image_parts: list[dict[str, Any]] = []
                for tr in tool_results:
                    if isinstance(tr.content, str):
                        output = tr.content
                        images: list[ImageBlock] = []
                    else:
                        unsupported = [b for b in tr.content if isinstance(b, (AudioBlock, VideoBlock))]
                        if unsupported:
                            media = ", ".join(sorted({b.media_type for b in unsupported}))
                            raise ValueError(f"OpenAI Responses does not support tool-result media: {media}")
                        output = "\n".join(b.text for b in tr.content if isinstance(b, TextBlock))
                        images = [b for b in tr.content if isinstance(b, ImageBlock)]
                    items.append({"type": "function_call_output", "call_id": tr.tool_use_id, "output": output})
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
                    items.append({"role": "user", "content": tool_image_parts})
                remaining_blocks = [b for b in msg.content if not isinstance(b, ToolResultBlock)]
            else:
                remaining_blocks = msg.content
            parts: list[dict[str, Any]] = []
            for b in remaining_blocks:
                if isinstance(b, TextBlock):
                    parts.append({"type": "input_text", "text": b.text})
                elif isinstance(b, ImageBlock):
                    encoded = base64.b64encode(b.data).decode("ascii")
                    parts.append({"type": "input_image", "image_url": f"data:{b.media_type};base64,{encoded}"})
                elif isinstance(b, (AudioBlock, VideoBlock)):
                    raise ValueError(f"OpenAI Responses does not support input media: {b.media_type}")
            if parts:
                items.append({"role": "user", "content": parts})

        elif msg.role == "system":
            unsupported_system = [b for b in msg.content if isinstance(b, (ImageBlock, AudioBlock, VideoBlock))]
            if unsupported_system:
                media = ", ".join(sorted({b.media_type for b in unsupported_system}))
                raise ValueError(f"OpenAI Responses does not support media in system messages: {media}")
            parts_system = [{"type": "input_text", "text": b.text} for b in msg.content if isinstance(b, TextBlock)]
            if parts_system:
                items.append({"role": "system", "content": parts_system})

        elif msg.role == "assistant":
            unsupported_assistant = [b for b in msg.content if isinstance(b, (ImageBlock, AudioBlock, VideoBlock))]
            if unsupported_assistant:
                media = ", ".join(sorted({b.media_type for b in unsupported_assistant}))
                raise ValueError(f"OpenAI Responses does not support media in assistant history: {media}")
            calls = [b for b in msg.content if isinstance(b, ToolUseBlock)]
            parts_out = [{"type": "output_text", "text": b.text} for b in msg.content if isinstance(b, TextBlock)]
            if parts_out:
                items.append({"role": "assistant", "content": parts_out})
            for b in calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": b.id,
                        "name": b.name,
                        "arguments": json.dumps(b.input),
                        "status": "completed",
                    }
                )

    # A call without its output makes the API reject the whole request, which
    # happens whenever a turn was interrupted or the context was compacted.
    answered = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    for item in list(items):
        if item.get("type") == "function_call" and item.get("call_id") not in answered:
            logger.warning("Synthesising placeholder output for unanswered call %s", item.get("call_id"))
            items.append({"type": "function_call_output", "call_id": item["call_id"], "output": _ORPHAN_OUTPUT})

    return items


@dataclass(slots=True)
class OpenAITransport(_OpenAIHTTPTransport):
    """Official OpenAI transport using only the Responses API."""

    name: str = "OpenAI"

    stream_path: ClassVar[str] = "responses"

    def configure_effort(self, requested: str | None) -> EffortState:
        level = parse_effort(requested)
        supported = _openai_reasoning_efforts(self.model.id) if Capability.reasoning in self.model.capabilities else ()
        if level is None:
            self.reasoning_effort = None
            mechanism = EffortMechanism.native_effort if supported else EffortMechanism.prompt_fallback
            return EffortState(None, mechanism, allowed=supported or EFFORT_LEVELS)
        if supported:
            if level not in supported:
                raise ValueError(
                    f"Effort {level!r} is not supported by {self.model.id}. Valid values: {', '.join(supported)}"
                )
            self.reasoning_effort = level
            return EffortState(level, EffortMechanism.native_effort, provider_value=level, allowed=supported)
        self.reasoning_effort = None
        return PromptEffortAdapter().configure_effort(level)

    def build_payload(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> dict[str, Any]:
        conflicts = _RESERVED_RESPONSE_PARAMS.intersection(self.extra_params)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"extra_params cannot override OpenAI Responses fields: {names}")
        payload: dict[str, Any] = {
            "model": self.model.id,
            "input": _convert_messages(messages),
            "stream": True,
            # Server-side conversation state is not used: axio keeps the history.
            "store": False,
        }
        if system:
            payload["instructions"] = system
        if tools:
            payload["tools"] = _convert_tools(tools)
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        if self.reasoning_effort and Capability.reasoning in self.model.capabilities:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        payload["max_output_tokens"] = _fit_output_limit(payload, self.model)
        if self.extra_params:
            payload.update(self.extra_params)
        return payload

    async def _parse_sse(self, resp: aiohttp.ClientResponse) -> AsyncIterator[StreamEvent]:
        usage = Usage(0, 0)
        stop_reason: StopReason | None = None
        saw_terminal = False
        # Argument deltas arrive keyed by item id, while the call was announced
        # under its call_id; without this map the two halves never join up.
        item_to_call: dict[str, str] = {}

        async def parse_line(line: str) -> AsyncIterator[StreamEvent]:
            nonlocal saw_terminal, stop_reason, usage
            if not line.startswith("data: "):
                return
            raw = line[6:]
            if raw == "[DONE]":
                return
            try:
                data: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise StreamError(f"Invalid OpenAI Responses SSE payload: {raw[:200]}") from exc

            event_type = data.get("type", "")

            if event_type == "response.output_text.delta":
                yield TextDelta(index=0, delta=data.get("delta", ""))

            elif event_type == "response.refusal.delta":
                yield TextDelta(index=0, delta=data.get("delta", ""))

            elif event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                yield ReasoningDelta(index=0, delta=data.get("delta", ""))

            elif event_type == "response.output_item.added":
                item = data.get("item", {})
                if item.get("type") == "function_call":
                    call_id = item.get("call_id", "")
                    if item.get("id"):
                        item_to_call[item["id"]] = call_id
                    yield ToolUseStart(
                        index=data.get("output_index", 0),
                        tool_use_id=call_id,
                        name=item.get("name", ""),
                    )

            elif event_type == "response.function_call_arguments.delta":
                item_id = data.get("item_id", "")
                yield ToolInputDelta(
                    index=data.get("output_index", 0),
                    tool_use_id=item_to_call.get(item_id, item_id),
                    partial_json=data.get("delta", ""),
                )

            elif event_type in ("response.completed", "response.incomplete"):
                saw_terminal = True
                response = data.get("response", {})
                resp_usage = response.get("usage") or {}
                usage = Usage(
                    input_tokens=resp_usage.get("input_tokens", 0),
                    output_tokens=resp_usage.get("output_tokens", 0),
                )
                output = response.get("output", [])
                if event_type == "response.incomplete" or response.get("status") == "incomplete":
                    stop_reason = StopReason.max_tokens
                elif any(i.get("type") == "function_call" for i in output):
                    stop_reason = StopReason.tool_use
                else:
                    stop_reason = StopReason.end_turn

            elif event_type in ("response.failed", "error"):
                response = data.get("response", data)
                error = response.get("error")
                message = (
                    error.get("message", "unknown error")
                    if isinstance(error, dict)
                    else str(error or response.get("message") or "unknown error")
                )
                code = response.get("code")
                detail = f"{code}: {message}" if code else message
                raise StreamError(f"OpenAI Responses API error: {detail}")

        buffer = b""
        async for chunk in resp.content.iter_any():
            buffer += chunk
            while b"\n" in buffer:
                await asyncio.sleep(0)
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8", "replace").strip()
                async for event in parse_line(line):
                    yield event

        if buffer.strip():
            line = buffer.decode("utf-8", "replace").strip()
            async for event in parse_line(line):
                yield event

        if not saw_terminal:
            raise StreamError("OpenAI Responses stream ended without a terminal response event")
        assert stop_reason is not None
        yield IterationEnd(iteration=0, stop_reason=stop_reason, usage=usage)
