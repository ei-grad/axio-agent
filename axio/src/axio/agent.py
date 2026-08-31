"""Agent: the core agentic loop orchestrating transport, tools, and context."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
from contextlib import aclosing, contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Protocol, Self

from . import background, notify
from ._asyncio import CancellationCause, shield_until_done
from .blocks import (
    AudioBlock,
    ContentBlock,
    ImageBlock,
    ProviderBlock,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
)
from .context import ContextStore
from .events import (
    AudioOutput,
    Error,
    ImageOutput,
    IterationEnd,
    ProviderOutput,
    ReasoningDelta,
    ReasoningSignature,
    Refusal,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    TextSignature,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
    VideoOutput,
)
from .exceptions import GuardCrash, HandlerCrash, ProviderOutputLimitError, StreamError, ToolError
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
    """Log expected tool failures without a traceback and crashes with one."""
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


def _last_unsigned(blocks: list[TurnBlock], kind: type[TextBlock] | type[ReasoningBlock]) -> int | None:
    """Where the nearest block of this kind still waiting for its proof sits, or None.

    Read as ``blocks[-1]`` alone, a proof that arrived after a block of another kind appended a
    signed empty one, which the next request replays as content the provider never produced.
    """
    for at in range(len(blocks) - 1, -1, -1):
        block = blocks[at]
        if isinstance(block, kind):
            return None if block.signature else at
    return None


#: The only reasons that vouch for the calls a turn produced. Named as the reasons to refuse
#: instead, a reason added later would let a failed turn's half-written arguments run.
_DISPATCH = frozenset(
    {
        StopReason.end_turn,
        StopReason.tool_use,
        #: A paused server-side tool loop. The turn did not fail, so its calls stand.
        StopReason.pause_turn,
    }
)


#: Sent after a tool returns media. Gemini otherwise ends the turn in about twenty tokens.
_MEDIA_NUDGE = "You now have the media file above in your context. Proceed."


def _malformed_results(blocks: list[ToolUseBlock], malformed: set[str]) -> list[ToolResultBlock]:
    """What a call whose arguments would not parse gets back instead of a run."""
    return [
        ToolResultBlock(
            tool_use_id=block.id,
            content=(
                f"Malformed JSON arguments for tool {block.name}. Raw input could not be parsed."
                f" Please retry the tool call with valid JSON arguments."
            ),
            is_error=True,
        )
        for block in blocks
        if block.id in malformed
    ]


def _refused_results(blocks: list[ToolUseBlock], stop_reason: StopReason | None) -> list[ToolResultBlock]:
    """What a call the turn did not vouch for gets back instead of a run.

    Dropped from the history instead, the next request holds no trace of the attempt: the model
    cannot tell a refused call from a turn that called nothing, and makes the same call again.
    """
    # str, not .value: a transport may hand the loop a reason this enum never named.
    named = str(stop_reason) if stop_reason is not None else "no stated reason"
    return [
        ToolResultBlock(
            tool_use_id=block.id,
            content=(
                f"Tool {block.name} was not run: the turn ended with {named}, which does not vouch"
                f" for the calls it produced. Nothing was executed and nothing changed."
            ),
            is_error=True,
        )
        for block in blocks
    ]


def _carries_media(results: list[ToolResultBlock]) -> bool:
    """Whether any result holds a picture, a sound or a video rather than only text."""
    return any(
        not isinstance(r.content, str) and any(isinstance(b, (AudioBlock, ImageBlock, VideoBlock)) for b in r.content)
        for r in results
    )


def _result_events(results: list[ToolResultBlock], by_id: dict[str, ToolUseBlock]) -> Iterator[StreamEvent]:
    """What the caller sees for each finished call.

    Media travels twice: as its own output event, which is how a caller saves it to disk, and
    inside the result, which is how the model sees the pixels.
    """
    for result in results:
        block = by_id.get(result.tool_use_id)
        if isinstance(result.content, str):
            text = result.content
        else:
            text = "\n".join(b.text for b in result.content if isinstance(b, TextBlock))
            for part in result.content:
                if isinstance(part, ImageBlock):
                    yield ImageOutput(index=0, data=part.data, media_type=part.media_type)
                elif isinstance(part, AudioBlock):
                    yield AudioOutput(index=0, data=part.data, media_type=part.media_type)
                elif isinstance(part, VideoBlock):
                    yield VideoOutput(index=0, data=part.data, media_type=part.media_type)
        yield ToolResult(
            tool_use_id=result.tool_use_id,
            name=block.name if block else "",
            is_error=result.is_error,
            content=text,
            input=block.input if block else {},
            started_at=result.started_at,
            finished_at=result.finished_at,
            duration_seconds=result.duration_seconds,
        )


@dataclass(slots=True)
class _PendingCall:
    """One tool call while its arguments are still arriving."""

    name: str
    signature: str = ""
    provider: str = ""
    json_parts: list[str] = field(default_factory=list)


#: What one iteration accumulates while the transport streams it.
type TurnBlock = TextBlock | ReasoningBlock | ImageBlock | AudioBlock | VideoBlock | ToolUseBlock | ProviderBlock


@dataclass(slots=True)
class _Turn:
    """What one iteration accumulates from the transport's stream."""

    content: list[TurnBlock] = field(default_factory=list)
    pending: dict[str, _PendingCall] = field(default_factory=dict)
    #: Calls whose arguments would not parse, so nothing may be dispatched from them.
    malformed: set[str] = field(default_factory=set)
    usage: Usage = Usage(0, 0)
    #: None until the transport says why it stopped. Defaulted to end_turn, a stream that simply
    #: ended stored half an answer as a whole one.
    stop_reason: StopReason | None = None
    repeated: bool = False
    output_limit_error: ProviderOutputLimitError | None = None
    #: Which block the last proof was for. A repeated index continues that proof.
    signed_at: int | None = None
    #: Which part the last text delta came from, and where that part's text starts inside the
    #: block it was merged into. A proof covers one part, so that offset is where to cut.
    text_at: int | None = None
    text_from: int = 0


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

    async def _call_one(self, block: ToolUseBlock, iteration: int) -> ToolResultBlock:
        """Run one call and shape whatever it returned, or the exception it raised, as a result."""
        tool = self._find_tool(block.name)
        if tool is None:
            logger.warning("Unknown tool requested: %s", block.name)
            return ToolResultBlock(tool_use_id=block.id, content=f"Unknown tool: {block.name}", is_error=True)
        logger.debug("Tool %s (id=%s) args=%s", block.name, block.id, json.dumps(block.input)[:200])
        args = {key: value for key, value in block.input.items() if key != BACKGROUND_PARAM}
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
            except Exception as exc:
                _log_tool_failure(block.name, exc)
                return ToolResultBlock(tool_use_id=block.id, content=str(exc), is_error=True)
        content: str | list[TextBlock | ImageBlock | AudioBlock | VideoBlock]
        if isinstance(result, str):
            content = result
        elif isinstance(result, list) and all(isinstance(b, ContentBlock) for b in result):
            content = result
        else:
            content = str(result)
        return ToolResultBlock(tool_use_id=block.id, content=content)

    async def dispatch_tools(self, blocks: list[ToolUseBlock], iteration: int) -> list[ToolResultBlock]:
        logger.info("Dispatching %d tool(s): %r", len(blocks), blocks)
        tasks = [asyncio.create_task(self._call_one(block, iteration)) for block in blocks]
        try:
            results = list(await asyncio.gather(*tasks))
        except BaseException:
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
            if tool is None or not tool.supports_streaming or block.input.get(BACKGROUND_PARAM):
                return await self._call_one(block, iteration)
            logger.debug("Tool %s (id=%s) args=%s", block.name, block.id, json.dumps(block.input)[:200])
            args = {key: value for key, value in block.input.items() if key != BACKGROUND_PARAM}
            chunks: list[tuple[float, str, str]] = []
            started = time.monotonic()
            with _tool_call_scope(block, iteration):
                try:
                    async for key, text in tool.call_streaming(**args):
                        chunks.append((time.monotonic() - started, key, text))
                        await output_queue.put(
                            ToolOutputDelta(tool_use_id=block.id, name=block.name, key=key, delta=text)
                        )
                except Exception as exc:
                    _log_tool_failure(block.name, exc)
                    return ToolResultBlock(tool_use_id=block.id, content=str(exc), is_error=True)
            return ToolResultBlock(tool_use_id=block.id, content=tool.format_stream_result(chunks))

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
        content: list[TurnBlock],
        delta: str,
    ) -> None:
        """Append a text delta, merging into the block still being built.

        A signed block is finished. The provider computed the signature over the text it had, so a
        later delta starts a new block rather than leaving a proof that disagrees with its text.
        """
        last = content[-1] if content else None
        if isinstance(last, TextBlock) and not last.signature:
            content[-1] = replace(last, text=last.text + delta)
        else:
            content.append(TextBlock(text=delta))

    @staticmethod
    def _accumulate_reasoning(
        content: list[TurnBlock],
        delta: str,
    ) -> None:
        """Append a reasoning delta, merging into the block still being built.

        A signed block is finished. The provider computed the signature over the text it had. A
        later delta therefore starts a new block, rather than leaving a signature that disagrees
        with the text beside it.
        """
        last = content[-1] if content else None
        if isinstance(last, ReasoningBlock) and not last.signature:
            content[-1] = replace(last, text=last.text + delta)
        else:
            content.append(ReasoningBlock(text=delta))

    @staticmethod
    def _sign_reasoning(
        content: list[TurnBlock],
        data: str,
        redacted: bool,
        id: str = "",
        joining: bool = False,
        provider: str = "",
    ) -> None:
        """Attach the provider's proof to the reasoning it belongs to.

        The proof is kept because the turn has to be replayable. A provider refuses a thinking block
        whose signature is missing, and a redacted block carries a signature and no text at all.

        ``joining`` says this proof continues the one before it. The transports index a signature
        by the block it proves, so a repeated index is the rest of one proof rather than another.
        Stored apart, the block replays with half a signature and the turn after it is refused.
        """
        last = content[-1] if content else None
        if joining and isinstance(last, ReasoningBlock) and last.signature:
            content[-1] = replace(last, signature=last.signature + data)
            return
        at = _last_unsigned(content, ReasoningBlock)
        if at is None:
            # A provider that sends the proof before the reasoning, which none does today.
            content.append(ReasoningBlock(signature=data, redacted=redacted, id=id, provider=provider))
            return
        # The provider travels with the proof: stored without it, no converter can tell whether the
        # value is one it issued, and a session that changed transport replays it to a stranger.
        signed = replace(content[at], signature=data, redacted=redacted, id=id, provider=provider)  # type: ignore[arg-type]
        content[at] = signed

    @staticmethod
    def _sign_text(
        content: list[TurnBlock],
        data: str,
        provider: str = "",
        signs_from: int = 0,
    ) -> None:
        """Attach the provider's proof to the answer text it belongs to.

        The proof arrives after the text it signs, so it goes on the nearest text still unsigned.
        Put on a reasoning block or a call instead, it is replayed on a part nobody signed.

        ``signs_from`` is where the signed part's text begins inside that block. Google signs the
        part it issued the proof for, and an answer reaches this vocabulary as a run of parts
        merged into one block, so a proof stored over the whole run covers text the provider never
        signed. The block is cut there: what came before keeps no proof, and what was signed
        carries its own.
        """
        at = _last_unsigned(content, TextBlock)
        if at is None:
            # Better to lose the proof than to invent a block for it: a signed empty text part is
            # replayed to the provider as content it never produced.
            logger.warning("Dropping a text signature with no text to attach it to")
            return
        block = content[at]
        assert isinstance(block, TextBlock)  # noqa: S101 - _last_unsigned looked for this kind
        if 0 < signs_from < len(block.text):
            content[at] = replace(block, text=block.text[:signs_from])
            content.insert(at + 1, TextBlock(text=block.text[signs_from:], signature=data, provider=provider))
            return
        content[at] = replace(block, signature=data, provider=provider)

    @staticmethod
    def _finalize_pending_tools(
        pending: dict[str, _PendingCall],
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
            raw = "".join(info.json_parts)
            if not raw:
                logger.warning(
                    "Tool %s (id=%s) received empty arguments (output may be truncated, output_tokens=%d)",
                    info.name,
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
                        info.name,
                        tid,
                        exc,
                        raw,
                    )
                    malformed.add(tid)
                    inp = {}
            blocks.append(
                ToolUseBlock(id=tid, name=info.name, input=inp, signature=info.signature, provider=info.provider)
            )
        return blocks, malformed

    async def _select_tools(self, history: list[Message], tools: list[Tool[Any]]) -> Iterable[Tool[Any]]:
        if not tools:
            return []
        if not self.selector:
            return tools
        return await self.selector.select(history, tools)

    def _interrupted_results(
        self, blocks: list[ToolUseBlock], partial: dict[str, list[tuple[float, str, str]]]
    ) -> list[ToolResultBlock]:
        """What each call reports when the user stops the run while it is still working.

        Whatever a streaming tool had already produced is kept, because a cancelled call still did
        part of its work and the model has to be told which part.
        """
        made: list[ToolResultBlock] = []
        for block in blocks:
            chunks = partial.get(block.id, [])
            tool = self._find_tool(block.name)
            if chunks and tool:
                text = tool.format_stream_result(chunks) + "\n[interrupted by user]"
            elif chunks:
                text = "".join(text for _, _, text in chunks) + "\n[interrupted by user]"
            else:
                text = "[interrupted by user]"
            made.append(ToolResultBlock(tool_use_id=block.id, content=text, is_error=True))
        return made

    async def _consume(
        self,
        stream: AsyncIterator[StreamEvent],
        turn: _Turn,
        context: ContextStore,
        output_guard: ProviderOutputGuard,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Pass every event through, building the turn's content as they go."""
        text_detector = _RepetitionDetector()
        reasoning_detector = _RepetitionDetector()
        try:
            async for event in stream:
                if (limit_error := output_guard.inspect(event)) is not None:
                    logger.error("Provider output circuit breaker stopped the response: %s", limit_error)
                    turn.output_limit_error = limit_error
                    self._accumulate_text(turn.content, limit_error.note)
                    return
                yield event
                match event:
                    case TextDelta(index=at, delta=delta):
                        if at != turn.text_at:
                            # Where this part starts is where a proof for it has to cut.
                            last = turn.content[-1] if turn.content else None
                            merging = isinstance(last, TextBlock) and not last.signature
                            turn.text_from = len(last.text) if isinstance(last, TextBlock) and merging else 0
                            turn.text_at = at
                        self._accumulate_text(turn.content, delta)
                        if text_detector.feed(delta):
                            note = "\n\n[Output truncated: repetitive content detected]"
                            self._accumulate_text(turn.content, note)
                            yield TextDelta(index=0, delta=note)
                            turn.repeated = True
                            return
                    case Refusal(text=text) if text:
                        # Stored whoever wrote it: a turn with no content is refused next time.
                        self._accumulate_text(turn.content, text)
                    case TextSignature(signature=proof, provider=provider):
                        self._sign_text(turn.content, proof, provider, turn.text_from)
                        turn.text_from = 0
                    case ReasoningDelta(delta=delta):
                        self._accumulate_reasoning(turn.content, delta)
                        if reasoning_detector.feed(delta):
                            note = "\n\n[Output truncated: repetitive content detected]"
                            self._accumulate_text(turn.content, note)
                            yield TextDelta(index=0, delta=note)
                            turn.repeated = True
                            return
                    case ReasoningSignature(
                        index=at, signature=proof, redacted=redacted, id=block_id, provider=provider
                    ):
                        self._sign_reasoning(
                            turn.content, proof, redacted, block_id, joining=at == turn.signed_at, provider=provider
                        )
                        turn.signed_at = at
                    case ProviderOutput(provider=provider, kind=kind, data=item, id=item_id):
                        # Stored, not watched: the endpoint that produced it keeps no history of
                        # its own, so the next request is incomplete without this item back.
                        turn.content.append(ProviderBlock(provider=provider, kind=kind, data=item, id=item_id))
                    case ImageOutput(data=data, media_type=mt):
                        turn.content.append(ImageBlock(media_type=mt, data=data))
                    case AudioOutput(data=data, media_type=mt):
                        turn.content.append(AudioBlock(media_type=mt, data=data))
                    case VideoOutput(data=data, media_type=mt):
                        turn.content.append(VideoBlock(media_type=mt, data=data))
                    case ToolUseStart(tool_use_id=tid, name=name, signature=proof, provider=provider):
                        turn.pending[tid] = _PendingCall(name=name, signature=proof, provider=provider)
                    case ToolInputDelta(tool_use_id=tid, partial_json=chunk) if tid in turn.pending:
                        turn.pending[tid].json_parts.append(chunk)
                    case IterationEnd(usage=usage, stop_reason=reason):
                        blocks, turn.malformed = self._finalize_pending_tools(turn.pending, usage)
                        turn.content.extend(blocks)
                        turn.pending.clear()
                        turn.usage = turn.usage + usage
                        turn.stop_reason = reason
                        await context.add_context_tokens(usage.input_tokens, usage.output_tokens)
        finally:
            # Left suspended, the HTTP response under it is released by the collector rather
            # than when the turn ended.
            if (close := getattr(stream, "aclose", None)) is not None:
                if turn.repeated or turn.output_limit_error is not None:
                    _, cancellation = await shield_until_done(close())
                    if cancellation is not None:
                        raise cancellation
                else:
                    await close()

    async def _dispatch_phase(
        self, blocks: list[ToolUseBlock], turn: _Turn, context: ContextStore, iteration: int
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run the turn's calls, store the turn beside their results, and report what happened."""
        if turn.stop_reason != StopReason.tool_use:
            logger.warning("Dispatching %d tool(s) despite stop_reason=%s", len(blocks), turn.stop_reason)

        runnable = [b for b in blocks if b.id not in turn.malformed]
        partial: dict[str, list[tuple[float, str, str]]] = {}
        started: dict[str, float] = {}
        malformed_results = _malformed_results(blocks, turn.malformed)
        execution_timings: dict[str, ToolExecutionTiming] = {}
        dispatch_task: asyncio.Task[list[ToolResultBlock]] | None = None
        dispatch: ToolDispatch | None = None
        try:
            if runnable:
                output_queue: asyncio.Queue[ToolOutputDelta | None] = asyncio.Queue()

                async def dispatch_and_signal() -> list[ToolResultBlock]:
                    try:
                        return await self._dispatch_tools_streaming(
                            runnable,
                            iteration,
                            output_queue,
                            execution_timings,
                        )
                    finally:
                        output_queue.put_nowait(None)

                dispatch_task = asyncio.create_task(dispatch_and_signal())
                dispatch = ToolDispatch(tuple(runnable), dispatch_task, notify.current_owner(), execution_timings)
                if self.deferred_tool_sink is not None:
                    self.deferred_tool_sink.dispatch_started(dispatch)
                while (delta := await output_queue.get()) is not None:
                    at = started.setdefault(delta.tool_use_id, time.monotonic())
                    partial.setdefault(delta.tool_use_id, []).append((time.monotonic() - at, delta.key, delta.delta))
                    yield delta
                await asyncio.sleep(0)
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise asyncio.CancelledError
                dispatched = await asyncio.shield(dispatch_task)
            else:
                dispatched = []
            results = dispatched + malformed_results
        except asyncio.CancelledError as cancellation:
            cancellation_note = "[interrupted by user]"
            if cancellation.args and isinstance(cancellation.args[0], CancellationCause):
                cancellation_note = f"[cancelled: {cancellation.args[0].reason}]"

            async def finalize_interrupted_dispatch() -> list[tuple[ToolUseBlock, ToolResultBlock]]:
                completed_results: dict[str, ToolResultBlock] = {}
                deferred = False
                if dispatch is not None and dispatch_task is not None:
                    prefer_deferral = self.deferred_tool_sink is not None and self.deferred_tool_sink.should_defer(
                        dispatch
                    )
                    if prefer_deferral and not dispatch_task.done():
                        for _ in range(2):
                            await asyncio.sleep(0)
                            if dispatch_task.done():
                                break
                    if dispatch_task.done() and not dispatch_task.cancelled():
                        try:
                            completed_results = {result.tool_use_id: result for result in dispatch_task.result()}
                        except BaseException:
                            completed_results = {}
                        if self.deferred_tool_sink is not None:
                            self.deferred_tool_sink.dispatch_finished(dispatch)
                    elif self.deferred_tool_sink is not None and self.deferred_tool_sink.should_defer(dispatch):
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
                malformed_by_id = {result.tool_use_id: result for result in malformed_results}
                for block in blocks:
                    if (completed := completed_results.get(block.id)) is not None:
                        interrupted_results.append(completed)
                        visible_results.append((block, completed))
                        continue
                    if deferred and block.id not in turn.malformed:
                        interrupted_results.append(
                            ToolResultBlock(
                                tool_use_id=block.id,
                                content=(
                                    f"Tool {block.name} continues after the model turn was preempted. "
                                    "Its actual result will arrive as a later user message."
                                ),
                            )
                        )
                        continue
                    if (malformed := malformed_by_id.get(block.id)) is not None:
                        interrupted_results.append(malformed)
                        visible_results.append((block, malformed))
                        continue
                    chunks = partial.get(block.id, [])
                    tool = self._find_tool(block.name)
                    if chunks and tool:
                        message = tool.format_stream_result(chunks) + f"\n{cancellation_note}"
                    elif chunks:
                        message = "".join(text for _, _, text in chunks) + f"\n{cancellation_note}"
                    else:
                        message = cancellation_note
                    timing = dispatch.timings.get(block.id) if dispatch is not None else None
                    interrupted = ToolResultBlock(
                        tool_use_id=block.id,
                        content=message,
                        is_error=True,
                        started_at=timing.started_at if timing is not None else None,
                        finished_at=timing.finished_at if timing is not None else None,
                        duration_seconds=timing.duration_seconds if timing is not None else None,
                    )
                    interrupted_results.append(interrupted)
                    visible_results.append((block, interrupted))

                await context.append_many(
                    [
                        Message(role="assistant", content=list(turn.content)),
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
            for block, result in visible_results:
                for event in _result_events([result], {block.id: block}):
                    yield event
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

        answer: list[ContentBlock] = list(results)
        if getattr(self.transport, "nudge_on_media_tool_result", False) and _carries_media(results):
            answer.append(TextBlock(text=_MEDIA_NUDGE))
        await context.append_many(
            [
                Message(role="assistant", content=list(turn.content)),
                Message(
                    role="user",
                    content=answer,
                    provenance=InputProvenance(
                        human_authored=False,
                        source="tool-result",
                        author="axio",
                    ),
                ),
            ]
        )

        for event in _result_events(results, {b.id: b for b in blocks}):
            yield event
        if self.before_next_provider_request is not None:
            await self.before_next_provider_request()

    async def _run_tools(
        self,
        runnable: list[ToolUseBlock],
        iteration: int,
        partial: dict[str, list[tuple[float, str, str]]],
        into: list[ToolResultBlock],
    ) -> AsyncGenerator[ToolOutputDelta, None]:
        """Run the calls, yielding what a streaming tool produces while it produces it.

        Results land in ``into`` rather than being returned: an async generator's return value is
        not reachable from ``async for``. ``partial`` keeps what each tool produced, so a
        cancellation can still report the part of the work that was done.
        """
        if not runnable:
            return
        if not any((t := self._find_tool(b.name)) is not None and t.supports_streaming for b in runnable):
            into += await self.dispatch_tools(runnable, iteration)
            return

        queue: asyncio.Queue[ToolOutputDelta | None] = asyncio.Queue()

        async def run() -> list[ToolResultBlock]:
            try:
                return await self._dispatch_tools_streaming(runnable, iteration, queue)
            finally:
                # Put only after a normal return, a dispatch that raised left the loop below
                # waiting on a queue nothing would fill again.
                await queue.put(None)

        task = asyncio.create_task(run())
        started: dict[str, float] = {}
        try:
            while (delta := await queue.get()) is not None:
                at = started.setdefault(delta.tool_use_id, time.monotonic())
                partial.setdefault(delta.tool_use_id, []).append((time.monotonic() - at, delta.key, delta.delta))
                yield delta
            into += await task
        finally:
            # However this generator ends, the task stops with it. Closing a stream throws
            # GeneratorExit, which is not a CancelledError.
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task

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
        turn_scope = notify.turn_scope(owner)
        turn_scope.__enter__()

        try:
            for iteration in range(1, self.max_iterations + 1):
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

                turn = _Turn()
                output_guard = ProviderOutputGuard(
                    self.provider_output_policy,
                    effective_output_tokens=snapshot_output_token_limit(
                        self.transport,
                        effective_history,
                        active_tools,
                        self.system,
                    ),
                )
                stream = self.transport.stream(effective_history, active_tools, self.system)
                try:
                    # `_consume` closes the stream in its `finally`, which runs only on close.
                    # Left to the collector, a caller that walked away held the response open.
                    async with aclosing(self._consume(stream, turn, context, output_guard)) as consumed:
                        async for event in consumed:
                            yield event
                except Exception as exc:
                    # The turn was billed for whatever it reported before it broke, and the caller
                    # is owed that figure. Counted only below, a failure after IterationEnd told
                    # them the turn was free.
                    total_usage = total_usage + turn.usage
                    logger.error("Transport error: %s", exc, exc_info=True)
                    yield Error(exception=exc)
                    yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                    session_end_emitted = True
                    return

                content = turn.content
                stop_reason = turn.stop_reason
                total_usage = total_usage + turn.usage

                if turn.output_limit_error is not None:
                    await self._append(context, Message(role="assistant", content=list(content)))
                    yield TextDelta(index=0, delta=turn.output_limit_error.note)
                    yield Error(exception=turn.output_limit_error)
                    yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                    session_end_emitted = True
                    return

                if turn.repeated:
                    # No IterationEnd arrived, so the turn carries neither its usage nor a reason.
                    # What the transport reported while it streamed is all there is.
                    partial = getattr(self.transport, "last_usage", None)
                    if partial is None:
                        logger.warning(
                            "Repetition cut the turn short and %s reports no running usage, so this"
                            " iteration's tokens are missing from the total",
                            type(self.transport).__name__,
                        )
                        partial = Usage(input_tokens=0, output_tokens=0)
                    total_usage = total_usage + partial
                    await self._append(context, Message(role="assistant", content=list(content)))
                    # Counted here as well: the message is stored either way, and a store that
                    # never counted it drifts further from the real context size every time,
                    # which is how autocompaction comes late.
                    await context.add_context_tokens(partial.input_tokens, partial.output_tokens)
                    yield SessionEndEvent(stop_reason=StopReason.repetition, total_usage=total_usage)
                    session_end_emitted = True
                    return

                tool_blocks = [b for b in content if isinstance(b, ToolUseBlock)]

                refused: list[ToolResultBlock] = []
                by_id: dict[str, ToolUseBlock] = {}
                if tool_blocks and stop_reason not in _DISPATCH:
                    # The run ends immediately after this regardless.
                    logger.warning(
                        "Not dispatching %d tool(s): the turn ended with stop_reason=%s",
                        len(tool_blocks),
                        stop_reason,
                    )
                    # The calls stay in the turn, each with a result beside it saying why it did
                    # not run: a stored call with no result is refused by the provider next, and a
                    # call with neither leaves the model no way to know it was ever attempted.
                    refused = _refused_results(tool_blocks, stop_reason)
                    by_id = {b.id: b for b in tool_blocks}
                    tool_blocks = []

                if tool_blocks:
                    async for event in self._dispatch_phase(tool_blocks, turn, context, iteration):
                        yield event
                    continue

                if refused:
                    await context.append_many(
                        [
                            Message(role="assistant", content=list(content)),
                            Message(
                                role="user",
                                content=list(refused),
                                provenance=InputProvenance(
                                    human_authored=False,
                                    source="tool-result",
                                    author="axio",
                                ),
                            ),
                        ]
                    )
                    # The transport already reported these calls as started, so a caller left with
                    # no result for them shows a call that never resolves.
                    for event in _result_events(refused, by_id):
                        yield event
                else:
                    await self._append(context, Message(role="assistant", content=list(content)))

                match stop_reason:
                    case None:
                        # The transport never said why it stopped, so the stream was cut. Read as
                        # end_turn, half an answer is stored and returned as a whole one.
                        yield Error(exception=StreamError("Transport ended without an IterationEnd"))
                        yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                        session_end_emitted = True
                        return
                    case StopReason.end_turn:
                        logger.debug("End turn: total_usage=%s", total_usage)
                        yield SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=total_usage)
                        session_end_emitted = True
                        return
                    case (
                        StopReason.refusal
                        | StopReason.cancelled
                        | StopReason.max_tokens
                        | StopReason.context_window_exceeded
                        | StopReason.unknown
                    ):
                        # Terminal, and none of them the transport breaking. Reported as an error, a
                        # caller retries what can never work.
                        logger.info("Ending on %s: total_usage=%s", stop_reason, total_usage)
                        yield SessionEndEvent(stop_reason=stop_reason, total_usage=total_usage)
                        session_end_emitted = True
                        return
                    case StopReason.tool_use if content:
                        # The turn added something, so the next request differs and may parse.
                        logger.warning("Asked for a tool and produced no call; re-prompting")
                        continue
                    case StopReason.tool_use:
                        # Nothing was added, so the next request is byte-identical to the last.
                        yield Error(exception=RuntimeError("Transport asked for a tool but produced no call"))
                        yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                        session_end_emitted = True
                        return
                    case StopReason.pause_turn:
                        # The provider expects the assistant content back, which was just appended.
                        logger.debug("Paused turn, resuming: total_usage=%s", total_usage)
                        continue
                    case _:
                        # Wildcard on purpose: a reason added later would otherwise fall through and
                        # the loop would run again until max_iterations.
                        yield Error(exception=RuntimeError(f"Transport stopped with: {stop_reason}"))
                        yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
                        session_end_emitted = True
                        return

            logger.warning("Max iterations (%d) reached", self.max_iterations)
            # With the event, like every other error, or `get_final_text()` returns half a turn.
            yield Error(
                exception=RuntimeError(
                    f"Stopped after {self.max_iterations} iterations without finishing (max_iterations reached)"
                )
            )
            yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
            session_end_emitted = True

        except GeneratorExit:
            return
        except BaseException:
            if not session_end_emitted:
                yield SessionEndEvent(stop_reason=StopReason.error, total_usage=total_usage)
            raise
        finally:
            turn_scope.__exit__(None, None, None)
