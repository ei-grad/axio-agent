"""Agent: the core agentic loop orchestrating transport, tools, and context."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, Self

from . import background, notify
from ._asyncio import CancellationCause, shield_until_done
from .blocks import AudioBlock, ContentBlock, ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock, VideoBlock
from .context import ContextStore
from .events import (
    AudioOutput,
    Error,
    ImageOutput,
    IterationEnd,
    ReasoningDelta,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
    VideoOutput,
)
from .exceptions import (
    GuardCrash,
    HandlerCrash,
    ProviderOutputLimitError,
    ToolError,
)
from .messages import InputProvenance, Message
from .models import Capability
from .provider_output import ProviderOutputGuard, ProviderOutputPolicy, snapshot_output_token_limit
from .selector import ToolSelector
from .stream import AgentStream
from .tool import BACKGROUND_PARAM, CURRENT_TOOL_CALL, Tool, ToolCallContext
from .transport import CompletionTransport
from .types import StopReason, Usage

logger = logging.getLogger(__name__)


@contextmanager
def _tool_call_scope(block: ToolUseBlock, iteration: int) -> Iterator[None]:
    token = CURRENT_TOOL_CALL.set(ToolCallContext(tool_use_id=block.id, tool_name=block.name, iteration=iteration))
    try:
        yield
    finally:
        CURRENT_TOOL_CALL.reset(token)


class _RepetitionDetector:
    """Detects when model output is stuck in a repetitive loop.

    Two complementary checks run periodically on accumulated text:

    1. **Short-period**: counts trailing consecutive repetitions of
       patterns from 1 to ``max_period`` chars.  Triggers when repetitions
       span >= ``min_repeat_span`` chars.  Catches single-token and
       short-phrase loops quickly.

    2. **Long-period**: checks whether the last ``long_window`` chars
       appear verbatim earlier in the output.  Catches paragraph-level
       repetition that the short-period check would miss.
    """

    __slots__ = (
        "_parts",
        "_total_len",
        "_last_check",
        "_interval",
        "_min_len",
        "_max_period",
        "_min_repeat_span",
        "_long_window",
    )

    def __init__(
        self,
        interval: int = 200,
        min_len: int = 800,
        max_period: int = 150,
        min_repeat_span: int = 200,
        long_window: int = 500,
    ) -> None:
        self._parts: list[str] = []
        self._total_len = 0
        self._last_check = 0
        self._interval = interval
        self._min_len = min_len
        self._max_period = max_period
        self._min_repeat_span = min_repeat_span
        self._long_window = long_window

    def feed(self, delta: str) -> bool:
        """Feed a text delta.  Returns ``True`` when a loop is detected."""
        self._parts.append(delta)
        self._total_len += len(delta)

        if self._total_len < self._min_len:
            return False
        if self._total_len - self._last_check < self._interval:
            return False
        self._last_check = self._total_len

        full = "".join(self._parts)
        self._parts = [full]
        n = len(full)

        # --- Short-period: trailing repetition of a small pattern ---
        max_p = min(self._max_period, n // 3)
        for p in range(1, max_p + 1):
            chunk = full[n - p : n]
            count = 1
            pos = n - 2 * p
            while pos >= 0 and full[pos : pos + p] == chunk:
                count += 1
                pos -= p
            if count >= 3 and count * p >= self._min_repeat_span:
                return True

        # --- Long-period: trailing window found earlier verbatim ---
        w = min(self._long_window, n // 2)
        if w >= self._min_repeat_span:
            window = full[-w:]
            if full.find(window, 0, n - w) >= 0:
                return True

        return False


def _log_tool_failure(name: str, exc: Exception) -> None:
    """Log a failed tool call, distinguishing expected failures from crashes.

    An expected failure (a handler reporting a missing file, a guard denying the
    call, invalid model-supplied input) is ordinary agent control flow and gets no
    traceback. Anything else escaped the code that should have handled it.
    """
    if isinstance(exc, ToolError) and not isinstance(exc, HandlerCrash | GuardCrash):
        logger.info("Tool %s failed: %s", name, exc)
    else:
        logger.error("Tool %s raised %s: %s", name, type(exc).__name__, exc, exc_info=exc)


async def _tool_result_text(tool: Tool[Any], args: dict[str, Any]) -> str:
    result = await tool(**args)
    return result if isinstance(result, str) else str(result)


@dataclass(frozen=True, slots=True)
class ToolExecutionTiming:
    """Wall-clock bounds plus monotonic duration for one actual tool execution."""

    started_at: datetime
    finished_at: datetime
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class ToolDispatch:
    """One concurrent tool batch whose task can outlive its originating turn."""

    blocks: tuple[ToolUseBlock, ...]
    task: asyncio.Task[list[ToolResultBlock]]
    owner: str | None
    timings: dict[str, ToolExecutionTiming] = field(default_factory=dict)


class DeferredToolSink(Protocol):
    def dispatch_started(self, dispatch: ToolDispatch) -> None: ...

    def dispatch_finished(self, dispatch: ToolDispatch) -> None: ...

    def defer(self, dispatch: ToolDispatch) -> None: ...

    def should_defer(self, dispatch: ToolDispatch) -> bool: ...

    def protocol_closed(self, dispatch: ToolDispatch) -> None: ...


@dataclass(slots=True)
class Agent:
    system: str
    transport: CompletionTransport
    tools: list[Tool[Any]] = field(default_factory=list)
    selector: ToolSelector | None = field(default=None)
    max_iterations: int = field(default=50)
    last_iteration_message: Message | None = field(default=None)
    deferred_tool_sink: DeferredToolSink | None = field(default=None)
    before_next_provider_request: Callable[[], Awaitable[None]] | None = field(default=None)
    provider_output_policy: ProviderOutputPolicy = field(default_factory=ProviderOutputPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.provider_output_policy, ProviderOutputPolicy):
            raise TypeError("provider_output_policy must be a ProviderOutputPolicy")

    def copy(self, **overrides: Any) -> Self:
        """Return a new Agent with *overrides* applied."""
        return dataclasses.replace(self, **overrides)

    def run_stream(
        self,
        user_message: str,
        context: ContextStore,
        *,
        provenance: InputProvenance | None = None,
    ) -> AgentStream:
        ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        message = Message(
            role="user",
            content=[TextBlock(text=f"[{ts}] {user_message}")],
            provenance=provenance or InputProvenance(human_authored=True, source="direct", author="human"),
        )
        return self.run_stream_messages((message,), context)

    async def run(
        self,
        user_message: str,
        context: ContextStore,
        *,
        provenance: InputProvenance | None = None,
    ) -> str:
        return await self.run_stream(user_message, context, provenance=provenance).get_final_text()

    def run_stream_messages(
        self,
        messages: Sequence[Message],
        context: ContextStore,
        *,
        on_input_committed: Callable[[], Awaitable[None]] | None = None,
    ) -> AgentStream:
        """Start one model operation after appending an ordered Message batch."""

        if not messages:
            raise ValueError("messages must not be empty")
        initial_messages = tuple(copy.deepcopy(messages))
        return AgentStream(self._run_loop(initial_messages, context, on_input_committed))

    async def run_messages(
        self,
        messages: Sequence[Message],
        context: ContextStore,
        *,
        on_input_committed: Callable[[], Awaitable[None]] | None = None,
    ) -> str:
        return await self.run_stream_messages(
            messages,
            context,
            on_input_committed=on_input_committed,
        ).get_final_text()

    async def dispatch_tools(self, blocks: list[ToolUseBlock], iteration: int) -> list[ToolResultBlock]:
        logger.info("Dispatching %d tool(s): %r", len(blocks), blocks)

        async def _run_one(block: ToolUseBlock) -> ToolResultBlock:
            tool = self._find_tool(block.name)
            if tool is None:
                logger.warning("Unknown tool requested: %s", block.name)
                return ToolResultBlock(tool_use_id=block.id, content=f"Unknown tool: {block.name}", is_error=True)
            logger.debug("Tool %s (id=%s) args=%s", block.name, block.id, json.dumps(block.input)[:200])
            args = {k: v for k, v in block.input.items() if k != BACKGROUND_PARAM}
            if block.input.get(BACKGROUND_PARAM):
                if not tool.detachable:
                    return ToolResultBlock(
                        tool_use_id=block.id,
                        content=f"Tool {block.name} does not support background execution.",
                        is_error=True,
                    )
                with _tool_call_scope(block, iteration):
                    handle = background.start(block.name, _tool_result_text(tool, args))
                return ToolResultBlock(tool_use_id=block.id, content=background.started_message(block.name, handle))
            with _tool_call_scope(block, iteration):
                try:
                    result = await tool(**args)
                    if isinstance(result, str):
                        content: str | list[TextBlock | ImageBlock | AudioBlock | VideoBlock] = result
                    elif isinstance(result, list) and all(isinstance(b, ContentBlock) for b in result):
                        content = result
                    else:
                        content = str(result)
                except Exception as exc:
                    _log_tool_failure(block.name, exc)
                    return ToolResultBlock(tool_use_id=block.id, content=str(exc), is_error=True)
            return ToolResultBlock(tool_use_id=block.id, content=content)

        tasks = [asyncio.create_task(_run_one(block)) for block in blocks]
        try:
            results = list(await asyncio.gather(*tasks))
        except BaseException:
            # Tool coroutines may terminate with cancellation or another
            # BaseException; either way no sibling dispatch may outlive this call.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        error_count = sum(1 for r in results if r.is_error)
        logger.info("Tools complete: %d total, %d errors", len(results), error_count)
        return results

    async def _dispatch_tools_streaming(
        self,
        blocks: list[ToolUseBlock],
        iteration: int,
        output_queue: asyncio.Queue[ToolOutputDelta | None],
        execution_timings: dict[str, ToolExecutionTiming] | None = None,
    ) -> list[ToolResultBlock]:
        """Like dispatch_tools but pushes ToolOutputDelta events for streaming tools."""
        logger.info("Dispatching %d tool(s) with streaming: %r", len(blocks), blocks)
        timings = execution_timings if execution_timings is not None else {}

        async def _execute_one(block: ToolUseBlock) -> ToolResultBlock:
            tool = self._find_tool(block.name)
            if tool is None:
                logger.warning("Unknown tool requested: %s", block.name)
                return ToolResultBlock(tool_use_id=block.id, content=f"Unknown tool: {block.name}", is_error=True)
            logger.debug("Tool %s (id=%s) args=%s", block.name, block.id, json.dumps(block.input)[:200])
            args = {k: v for k, v in block.input.items() if k != BACKGROUND_PARAM}
            if block.input.get(BACKGROUND_PARAM):
                if not tool.detachable:
                    return ToolResultBlock(
                        tool_use_id=block.id,
                        content=f"Tool {block.name} does not support background execution.",
                        is_error=True,
                    )
                with _tool_call_scope(block, iteration):
                    handle = background.start(block.name, _tool_result_text(tool, args))
                return ToolResultBlock(tool_use_id=block.id, content=background.started_message(block.name, handle))

            if tool.supports_streaming:
                chunks: list[tuple[float, str, str]] = []
                t0 = time.monotonic()
                with _tool_call_scope(block, iteration):
                    try:
                        async for key, text in tool.call_streaming(**args):
                            chunks.append((time.monotonic() - t0, key, text))
                            await output_queue.put(
                                ToolOutputDelta(tool_use_id=block.id, name=block.name, key=key, delta=text)
                            )
                    except Exception as exc:
                        _log_tool_failure(block.name, exc)
                        return ToolResultBlock(tool_use_id=block.id, content=str(exc), is_error=True)
                return ToolResultBlock(tool_use_id=block.id, content=tool.format_stream_result(chunks))
            else:
                with _tool_call_scope(block, iteration):
                    try:
                        result = await tool(**args)
                        if isinstance(result, str):
                            content: str | list[TextBlock | ImageBlock | AudioBlock | VideoBlock] = result
                        elif isinstance(result, list) and all(isinstance(b, ContentBlock) for b in result):
                            content = result
                        else:
                            content = str(result)
                    except Exception as exc:
                        _log_tool_failure(block.name, exc)
                        return ToolResultBlock(tool_use_id=block.id, content=str(exc), is_error=True)
                return ToolResultBlock(tool_use_id=block.id, content=content)

        async def _run_one(block: ToolUseBlock) -> ToolResultBlock:
            tool = self._find_tool(block.name)
            if tool is None or block.input.get(BACKGROUND_PARAM):
                return await _execute_one(block)
            started_at = datetime.now().astimezone()
            started_monotonic = time.monotonic()
            try:
                result = await _execute_one(block)
            except BaseException:
                finished_at = datetime.now().astimezone()
                timings[block.id] = ToolExecutionTiming(
                    started_at,
                    finished_at,
                    time.monotonic() - started_monotonic,
                )
                raise
            finished_at = datetime.now().astimezone()
            timing = ToolExecutionTiming(
                started_at,
                finished_at,
                time.monotonic() - started_monotonic,
            )
            timings[block.id] = timing
            return dataclasses.replace(
                result,
                started_at=timing.started_at,
                finished_at=timing.finished_at,
                duration_seconds=timing.duration_seconds,
            )

        tasks = [asyncio.create_task(_run_one(block)) for block in blocks]
        try:
            results = list(await asyncio.gather(*tasks))
        except BaseException:
            # Streaming tools have the same structured-lifetime requirement as
            # regular tools, including failures outside Exception.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        error_count = sum(1 for r in results if r.is_error)
        logger.info("Tools complete: %d total, %d errors", len(results), error_count)
        return results

    def _find_tool(self, name: str) -> Tool[Any] | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    async def _append(self, context: ContextStore, message: Message) -> None:
        await context.append(message)

    @staticmethod
    def _accumulate_text(
        content: list[TextBlock | ImageBlock | AudioBlock | VideoBlock | ToolUseBlock],
        delta: str,
    ) -> None:
        """Append text delta — merge into last TextBlock or start a new one."""
        if content and isinstance(content[-1], TextBlock):
            content[-1] = TextBlock(text=content[-1].text + delta)
        else:
            content.append(TextBlock(text=delta))

    @staticmethod
    def _result_events(
        result: ToolResultBlock,
        block: ToolUseBlock | None,
    ) -> Iterator[StreamEvent]:
        if isinstance(result.content, str):
            result_content = result.content
        else:
            result_content = "\n".join(item.text for item in result.content if isinstance(item, TextBlock))
            for media_block in result.content:
                if isinstance(media_block, ImageBlock):
                    yield ImageOutput(index=0, data=media_block.data, media_type=media_block.media_type)
                elif isinstance(media_block, AudioBlock):
                    yield AudioOutput(index=0, data=media_block.data, media_type=media_block.media_type)
                elif isinstance(media_block, VideoBlock):
                    yield VideoOutput(index=0, data=media_block.data, media_type=media_block.media_type)
        yield ToolResult(
            tool_use_id=result.tool_use_id,
            name=block.name if block else "",
            is_error=result.is_error,
            content=result_content,
            input=block.input if block else {},
            started_at=result.started_at,
            finished_at=result.finished_at,
            duration_seconds=result.duration_seconds,
        )

    @staticmethod
    def _finalize_pending_tools(
        pending: dict[str, dict[str, Any]],
        usage: Usage,
    ) -> tuple[list[ToolUseBlock], set[str]]:
        """Convert streamed tool-call fragments into ToolUseBlocks.

        Returns (blocks, malformed_ids).  Malformed IDs arise when
        max_tokens truncates the response mid-tool-call, producing
        incomplete JSON (expected with eager_input_streaming).  The
        caller is responsible for not executing malformed tools.
        """
        blocks: list[ToolUseBlock] = []
        malformed: set[str] = set()
        for tid, info in pending.items():
            raw = "".join(info["json_parts"])
            if not raw:
                logger.warning(
                    "Tool %s (id=%s) received empty arguments (output may be truncated, output_tokens=%d)",
                    info["name"],
                    tid,
                    usage.output_tokens,
                )
                inp: dict[str, Any] = {}
            else:
                try:
                    inp = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Tool %s (id=%s) has malformed JSON arguments: %s\nRaw: %s",
                        info["name"],
                        tid,
                        exc,
                        raw,
                    )
                    malformed.add(tid)
                    inp = {}
            blocks.append(ToolUseBlock(id=tid, name=info["name"], input=inp))
        return blocks, malformed

    async def _select_tools(self, history: list[Message], tools: list[Tool[Any]]) -> Iterable[Tool[Any]]:
        if not tools:
            return []
        if not self.selector:
            return tools
        return await self.selector.select(history, tools)

    async def _run_loop(
        self,
        initial_messages: tuple[Message, ...],
        context: ContextStore,
        on_input_committed: Callable[[], Awaitable[None]] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        total_usage = Usage(0, 0)
        session_end_emitted = False
        owner = notify.current_owner()
        await context.append_many(list(initial_messages))
        if on_input_committed is not None:
            await on_input_committed()

        try:
            with notify.turn_scope(owner):
                for iteration in range(1, self.max_iterations + 1):
                    # Notifications that arrived mid-turn join the conversation as
                    # their own user message: appending never rewrites what is
                    # already there, so no tool_use/tool_result pair is disturbed.
                    if notifications := notify.drain(owner):
                        await self._append(
                            context,
                            Message(
                                role="user",
                                content=[TextBlock(text="\n\n".join(notifications))],
                                provenance=InputProvenance(
                                    human_authored=False,
                                    source="notification",
                                    author="axio",
                                ),
                            ),
                        )
                    history = await context.get_history()
                    logger.info("Iteration %d, history length=%d", iteration, len(history))
                    effective_history = (
                        [*history, self.last_iteration_message]
                        if self.last_iteration_message and iteration == self.max_iterations
                        else history
                    )
                    model = getattr(self.transport, "model", None)
                    model_caps = getattr(model, "capabilities", None)
                    if model_caps is not None and Capability.tool_use not in model_caps:
                        active_tools: list[Tool[Any]] = []
                    else:
                        active_tools = list(await self._select_tools(effective_history, self.tools))

                    content: list[TextBlock | ImageBlock | AudioBlock | VideoBlock | ToolUseBlock] = []
                    pending: dict[str, dict[str, Any]] = {}
                    stop_reason = StopReason.end_turn
                    malformed: set[str] = set()
                    repetition_detected = False
                    output_limit_error: ProviderOutputLimitError | None = None
                    text_rep_detector = _RepetitionDetector()
                    reasoning_rep_detector = _RepetitionDetector()

                    try:
                        output_guard = ProviderOutputGuard(
                            self.provider_output_policy,
                            effective_output_tokens=snapshot_output_token_limit(
                                self.transport,
                                effective_history,
                                active_tools,
                                self.system,
                            ),
                        )
                        provider_stream = self.transport.stream(effective_history, active_tools, self.system)
                        try:
                            async for event in provider_stream:
                                output_limit_error = output_guard.inspect(event)
                                if output_limit_error is not None:
                                    logger.error(
                                        "Provider output circuit breaker stopped the response: %s",
                                        output_limit_error,
                                    )
                                    self._accumulate_text(content, output_limit_error.note)
                                    break
                                yield event
                                match event:
                                    case TextDelta(delta=delta):
                                        self._accumulate_text(content, delta)
                                        if text_rep_detector.feed(delta):
                                            note = "\n\n[Output truncated: repetitive content detected]"
                                            self._accumulate_text(content, note)
                                            repetition_detected = True
                                            yield TextDelta(index=0, delta=note)
                                            break
                                    case ReasoningDelta(delta=delta):
                                        if reasoning_rep_detector.feed(delta):
                                            note = "\n\n[Output truncated: repetitive content detected]"
                                            self._accumulate_text(content, note)
                                            repetition_detected = True
                                            yield TextDelta(index=0, delta=note)
                                            break
                                    case ImageOutput(data=data, media_type=mt):
                                        content.append(ImageBlock(media_type=mt, data=data))
                                    case VideoOutput(data=data, media_type=mt):
                                        content.append(VideoBlock(media_type=mt, data=data))
                                    case ToolUseStart(tool_use_id=tid, name=name):
                                        pending[tid] = {"name": name, "json_parts": []}
                                    case ToolInputDelta(tool_use_id=tid, partial_json=pj):
                                        if tid in pending:
                                            pending[tid]["json_parts"].append(pj)
                                    case IterationEnd(usage=usage, stop_reason=sr):
                                        blocks, malformed = self._finalize_pending_tools(pending, usage)
                                        content.extend(blocks)
                                        pending.clear()
                                        total_usage = total_usage + usage
                                        await context.add_context_tokens(usage.input_tokens, usage.output_tokens)
                                        stop_reason = sr
                        finally:
                            if repetition_detected or output_limit_error is not None:
                                close_provider_stream = getattr(provider_stream, "aclose", None)
                                if close_provider_stream is not None:
                                    _, cancellation = await shield_until_done(close_provider_stream())
                                    if cancellation is not None:
                                        raise cancellation
                    except Exception as exc:
                        logger.error("Transport error: %s", exc, exc_info=True)
                        yield Error(exception=exc)
                        yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                        session_end_emitted = True
                        return

                    if repetition_detected or output_limit_error is not None:
                        await self._append(context, Message(role="assistant", content=list(content)))
                        # The provider did not emit IterationEnd, so this call has no
                        # trustworthy usage or provider stop reason to account.
                        if output_limit_error is not None:
                            yield TextDelta(index=0, delta=output_limit_error.note)
                            yield Error(exception=output_limit_error)
                        yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                        session_end_emitted = True
                        return

                    tool_blocks = [b for b in content if isinstance(b, ToolUseBlock)]

                    if tool_blocks:
                        if stop_reason != StopReason.tool_use:
                            logger.warning(
                                "Dispatching %d tool(s) despite stop_reason=%s",
                                len(tool_blocks),
                                stop_reason,
                            )

                        # Dispatch tools BEFORE appending to context - cancellation
                        # between here and the two appends below cannot leave orphan
                        # ToolUseBlocks in the persistent context store.
                        valid = [b for b in tool_blocks if b.id not in malformed]
                        error_results = [
                            ToolResultBlock(
                                tool_use_id=b.id,
                                content=(
                                    f"Malformed JSON arguments for tool {b.name}."
                                    f" Raw input could not be parsed. Please retry the tool call"
                                    f" with valid JSON arguments."
                                ),
                                is_error=True,
                            )
                            for b in tool_blocks
                            if b.id in malformed
                        ]

                        partial_output: dict[str, list[tuple[float, str, str]]] = {}
                        t0_map: dict[str, float] = {}
                        dispatch_task: asyncio.Task[list[ToolResultBlock]] | None = None
                        dispatch: ToolDispatch | None = None
                        try:
                            if valid:
                                execution_timings: dict[str, ToolExecutionTiming] = {}
                                output_queue: asyncio.Queue[ToolOutputDelta | None] = asyncio.Queue()

                                async def _dispatch_and_signal() -> list[ToolResultBlock]:
                                    try:
                                        return await self._dispatch_tools_streaming(
                                            valid,
                                            iteration,
                                            output_queue,
                                            execution_timings,
                                        )
                                    finally:
                                        output_queue.put_nowait(None)

                                dispatch_task = asyncio.create_task(_dispatch_and_signal())
                                dispatch = ToolDispatch(tuple(valid), dispatch_task, owner, execution_timings)
                                if self.deferred_tool_sink is not None:
                                    self.deferred_tool_sink.dispatch_started(dispatch)
                                while True:
                                    ev = await output_queue.get()
                                    if ev is None:
                                        break
                                    if isinstance(ev, ToolOutputDelta):
                                        if ev.tool_use_id not in t0_map:
                                            t0_map[ev.tool_use_id] = time.monotonic()
                                        partial_output.setdefault(ev.tool_use_id, []).append(
                                            (time.monotonic() - t0_map[ev.tool_use_id], ev.key, ev.delta)
                                        )
                                    yield ev
                                await asyncio.sleep(0)
                                current_task = asyncio.current_task()
                                if current_task is not None and current_task.cancelling():
                                    raise asyncio.CancelledError
                                dispatched = await asyncio.shield(dispatch_task)
                            else:
                                dispatched = []
                            results = dispatched + error_results
                        except asyncio.CancelledError as cancellation:
                            cancellation_note = "[interrupted by user]"
                            if cancellation.args and isinstance(cancellation.args[0], CancellationCause):
                                cancellation_note = f"[cancelled: {cancellation.args[0].reason}]"

                            async def finalize_interrupted_dispatch() -> list[tuple[ToolUseBlock, ToolResultBlock]]:
                                completed_results: dict[str, ToolResultBlock] = {}
                                deferred = False
                                if dispatch is not None and dispatch_task is not None:
                                    prefer_deferral = (
                                        self.deferred_tool_sink is not None
                                        and self.deferred_tool_sink.should_defer(dispatch)
                                    )
                                    if prefer_deferral and not dispatch_task.done():
                                        # An immediate handler completes through its child task and the
                                        # dispatch gather task on successive scheduler turns. Let runnable
                                        # completion propagate before turning it into deferred work.
                                        for _ in range(2):
                                            await asyncio.sleep(0)
                                            if dispatch_task.done():
                                                break
                                    if dispatch_task.done() and not dispatch_task.cancelled():
                                        try:
                                            completed_results = {
                                                result.tool_use_id: result for result in dispatch_task.result()
                                            }
                                        except BaseException:
                                            completed_results = {}
                                        if self.deferred_tool_sink is not None:
                                            self.deferred_tool_sink.dispatch_finished(dispatch)
                                    elif self.deferred_tool_sink is not None and self.deferred_tool_sink.should_defer(
                                        dispatch
                                    ):
                                        self.deferred_tool_sink.defer(dispatch)
                                        deferred = True
                                    else:
                                        if not dispatch_task.done():
                                            dispatch_task.cancel()
                                            await asyncio.gather(dispatch_task, return_exceptions=True)
                                        if self.deferred_tool_sink is not None:
                                            self.deferred_tool_sink.dispatch_finished(dispatch)
                                interrupted_results: list[ToolResultBlock] = []
                                visible_results: list[tuple[ToolUseBlock, ToolResultBlock]] = []
                                for b in tool_blocks:
                                    completed = completed_results.get(b.id)
                                    if completed is not None:
                                        interrupted_results.append(completed)
                                        visible_results.append((b, completed))
                                        continue
                                    if deferred and b.id not in malformed:
                                        interrupted_results.append(
                                            ToolResultBlock(
                                                tool_use_id=b.id,
                                                content=(
                                                    f"Tool {b.name} continues after the model turn was preempted. "
                                                    "Its actual result will arrive as a later user message."
                                                ),
                                            )
                                        )
                                        continue
                                    malformed_result = next(
                                        (result for result in error_results if result.tool_use_id == b.id),
                                        None,
                                    )
                                    if malformed_result is not None:
                                        interrupted_results.append(malformed_result)
                                        visible_results.append((b, malformed_result))
                                        continue
                                    chunks = partial_output.get(b.id, [])
                                    tool = self._find_tool(b.name)
                                    if chunks and tool:
                                        msg = tool.format_stream_result(chunks) + f"\n{cancellation_note}"
                                    elif chunks:
                                        msg = "".join(text for _, _, text in chunks) + f"\n{cancellation_note}"
                                    else:
                                        msg = cancellation_note
                                    timing = dispatch.timings.get(b.id) if dispatch is not None else None
                                    interrupted = ToolResultBlock(
                                        tool_use_id=b.id,
                                        content=msg,
                                        is_error=True,
                                        started_at=timing.started_at if timing is not None else None,
                                        finished_at=timing.finished_at if timing is not None else None,
                                        duration_seconds=timing.duration_seconds if timing is not None else None,
                                    )
                                    interrupted_results.append(interrupted)
                                    visible_results.append((b, interrupted))
                                await context.append_many(
                                    [
                                        Message(role="assistant", content=list(content)),
                                        Message(
                                            role="user",
                                            content=list(interrupted_results),
                                            provenance=InputProvenance(
                                                human_authored=False,
                                                source="tool-result",
                                                author="axio",
                                            ),
                                        ),
                                    ]
                                )
                                if deferred and dispatch is not None and self.deferred_tool_sink is not None:
                                    self.deferred_tool_sink.protocol_closed(dispatch)
                                return visible_results

                            visible_results, _ = await shield_until_done(finalize_interrupted_dispatch())
                            for visible_block, result in visible_results:
                                for result_event in self._result_events(result, visible_block):
                                    yield result_event
                            raise
                        except BaseException:
                            if dispatch_task is not None and not dispatch_task.done():
                                dispatch_task.cancel()
                                await asyncio.gather(dispatch_task, return_exceptions=True)
                            if dispatch is not None and self.deferred_tool_sink is not None:
                                self.deferred_tool_sink.dispatch_finished(dispatch)
                            raise

                        if dispatch is not None and self.deferred_tool_sink is not None:
                            self.deferred_tool_sink.dispatch_finished(dispatch)

                        await context.append_many(
                            [
                                Message(role="assistant", content=list(content)),
                                Message(
                                    role="user",
                                    content=list(results),
                                    provenance=InputProvenance(
                                        human_authored=False,
                                        source="tool-result",
                                        author="axio",
                                    ),
                                ),
                            ]
                        )

                        # Gemini stops generating (~20 tokens, end_turn) after receiving
                        # media as sibling inlineData parts alongside functionResponse.
                        # A "Proceed." user message nudges it to actually analyze the content.
                        if getattr(self.transport, "nudge_on_media_tool_result", False) and any(
                            not isinstance(r.content, str)
                            and any(isinstance(b, (AudioBlock, ImageBlock, VideoBlock)) for b in r.content)
                            for r in results
                        ):
                            await self._append(
                                context,
                                Message(
                                    role="user",
                                    content=[
                                        TextBlock(
                                            text="You now have the media file above in your context. Proceed.",
                                        )
                                    ],
                                    provenance=InputProvenance(
                                        human_authored=False,
                                        source="transport-nudge",
                                        author="axio",
                                    ),
                                ),
                            )

                        # Yield ToolResult events + media output events.
                        # Non-streaming tools return full content (str or list of
                        # TextBlock/ImageBlock/VideoBlock) — no information is lost.
                        # Images/videos are yielded as separate ImageOutput/VideoOutput
                        # events so the REPL can save them to disk; the model sees the
                        # actual pixel data via ImageBlock/VideoBlock in the tool result.
                        by_id = {b.id: b for b in tool_blocks}
                        for r in results:
                            block = by_id.get(r.tool_use_id)
                            for result_event in self._result_events(r, block):
                                yield result_event
                        if self.before_next_provider_request is not None:
                            await self.before_next_provider_request()
                        continue

                    await self._append(context, Message(role="assistant", content=list(content)))

                    match stop_reason:
                        case StopReason.end_turn:
                            logger.debug("End turn: total_usage=%s", total_usage)
                            yield SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=total_usage)
                            session_end_emitted = True
                            return
                        case StopReason.max_tokens | StopReason.error:
                            yield Error(exception=RuntimeError(f"Transport stopped with: {stop_reason}"))
                            yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                            session_end_emitted = True
                            return

                logger.warning("Max iterations (%d) reached", self.max_iterations)
                # Announced, not only logged: an agent that ran out of iterations
                # produced no answer, and a caller watching it from another process
                # otherwise sees the same "finished" as one that succeeded.
                yield Error(
                    exception=RuntimeError(f"Stopped after {self.max_iterations} iterations without finishing")
                )
                yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                session_end_emitted = True

        except GeneratorExit:
            return
        except BaseException:
            if not session_end_emitted:
                yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
            raise
