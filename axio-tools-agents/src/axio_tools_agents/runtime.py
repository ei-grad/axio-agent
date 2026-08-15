"""Typed observation and lifecycle primitives for in-process agent runs."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import logging
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from uuid import uuid4

from axio.agent import Agent
from axio.blocks import TextBlock
from axio.context import ContextStore, SessionInfo
from axio.events import Error, SessionEndEvent, StreamEvent, TextDelta, ToolResult
from axio.messages import Message
from axio.types import StopReason

logger = logging.getLogger(__name__)


class ExecutionMode(StrEnum):
    FOREGROUND = "foreground"
    BACKGROUND = "background"


class TurnStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AgentStarted:
    name: str
    kind: str


@dataclass(frozen=True, slots=True)
class AgentStopped:
    status: TurnStatus


@dataclass(frozen=True, slots=True)
class TurnStarted:
    prompt: str


@dataclass(frozen=True, slots=True)
class TurnFinished:
    status: TurnStatus
    stop_reason: StopReason | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ForegroundEntered:
    parent_agent_id: str | None


@dataclass(frozen=True, slots=True)
class ForegroundExited:
    status: TurnStatus


@dataclass(frozen=True, slots=True)
class OutcomeDelivered:
    recipient_agent_id: str | None
    route: str


@dataclass(frozen=True, slots=True)
class InputReceived:
    text: str
    source: str


@dataclass(frozen=True, slots=True)
class InputBuffered:
    input_id: str
    text: str
    intended_target_agent_id: str


@dataclass(frozen=True, slots=True)
class InputRecalled:
    input_ids: tuple[str, ...]
    editor_text: str


@dataclass(frozen=True, slots=True)
class InputClaimed:
    input_ids: tuple[str, ...]
    target_agent_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class InputDelivered:
    input_ids: tuple[str, ...]
    target_agent_id: str


@dataclass(frozen=True, slots=True)
class InterruptionRequested:
    target_agent_id: str
    captured_turn_id: str | None


@dataclass(frozen=True, slots=True)
class InterruptionCommitted:
    request_seq: int
    target_agent_id: str
    captured_turn_id: str | None
    reason: str
    claimed_input_ids: tuple[str, ...]
    partial_text: str


@dataclass(frozen=True, slots=True)
class EditorSnapshot:
    text: str


@dataclass(frozen=True, slots=True)
class ShutdownRecorded:
    reason: str
    pending_input_ids: tuple[str, ...]
    deferred_tool_use_ids: tuple[str, ...]
    interrupted_turn_id: str | None = None
    partial_text: str = ""
    deferred_tool_agent_ids: tuple[str, ...] = ()
    deferred_tool_turn_ids: tuple[str | None, ...] = ()
    deferred_tool_phases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecoveryApplied:
    source_session_id: str
    recovery_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConfigurationChanged:
    name: str
    value: object
    source: str


@dataclass(frozen=True, slots=True)
class MessageCommitted:
    message: Message
    source_input_id: str | None = None

    def __post_init__(self) -> None:
        if self.source_input_id == "":
            raise ValueError("source_input_id must not be empty")


@dataclass(frozen=True, slots=True)
class ContextForked:
    source_context_id: str
    child_context_id: str


@dataclass(frozen=True, slots=True)
class ContextCleared:
    pass


type RuntimeEvent = (
    StreamEvent
    | AgentStarted
    | AgentStopped
    | TurnStarted
    | TurnFinished
    | ForegroundEntered
    | ForegroundExited
    | OutcomeDelivered
    | InputReceived
    | InputBuffered
    | InputRecalled
    | InputClaimed
    | InputDelivered
    | InterruptionRequested
    | InterruptionCommitted
    | EditorSnapshot
    | ShutdownRecorded
    | RecoveryApplied
    | ConfigurationChanged
    | MessageCommitted
    | ContextForked
    | ContextCleared
)


@dataclass(frozen=True, slots=True)
class AgentEventEnvelope:
    seq: int
    session_id: str
    run_id: str
    agent_id: str
    parent_agent_id: str | None
    turn_id: str | None
    execution_mode: ExecutionMode
    parent_tool_use_id: str | None
    event: RuntimeEvent
    context_id: str | None = None


@dataclass(frozen=True, slots=True)
class TurnIdentity:
    run_id: str
    agent_id: str
    parent_agent_id: str | None
    turn_id: str
    execution_mode: ExecutionMode
    parent_tool_use_id: str | None = None
    context_id: str | None = None


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    identity: TurnIdentity
    status: TurnStatus
    text: str
    stop_reason: StopReason | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is TurnStatus.SUCCEEDED


EventSubscriber = Callable[[AgentEventEnvelope], Awaitable[None]]
Unsubscribe = Callable[[], None]


@dataclass(slots=True)
class _PendingPublication:
    envelope: AgentEventEnvelope
    subscribers: tuple[EventSubscriber, ...]
    result: asyncio.Future[AgentEventEnvelope]


_current_turn_identity: contextvars.ContextVar[TurnIdentity | None] = contextvars.ContextVar(
    "axio_tools_agents_current_turn_identity",
    default=None,
)
_initial_message_input_ids: contextvars.ContextVar[tuple[str | None, ...] | None] = contextvars.ContextVar(
    "axio_tools_agents_initial_message_input_ids",
    default=None,
)


def current_turn_identity() -> TurnIdentity | None:
    """Return the turn whose agent code is executing in this task context."""

    return _current_turn_identity.get()


class SessionEventHub:
    """Assign a session order and fan every runtime event out to all observers."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid4().hex
        self._seq = 0
        self._sequence_lock = threading.Lock()
        self._reserved_sequences: set[int] = set()
        self._next_publication_seq = 1
        self._pending_publications: dict[int, _PendingPublication] = {}
        self._subscribers: list[EventSubscriber] = []
        self._publish_lock = asyncio.Lock()

    def reserve_sequence(self) -> int:
        """Reserve the next logical sequence from a synchronous ingress callback."""

        with self._sequence_lock:
            self._seq += 1
            sequence = self._seq
            self._reserved_sequences.add(sequence)
            return sequence

    def subscribe(self, subscriber: EventSubscriber) -> Unsubscribe:
        self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(subscriber)
            except ValueError:
                pass

        return unsubscribe

    async def publish(
        self,
        event: RuntimeEvent,
        *,
        run_id: str,
        agent_id: str,
        parent_agent_id: str | None,
        turn_id: str | None,
        execution_mode: ExecutionMode,
        parent_tool_use_id: str | None = None,
        context_id: str | None = None,
        reserved_seq: int | None = None,
    ) -> AgentEventEnvelope:
        async with self._publish_lock:
            sequence = self._claim_sequence(reserved_seq)
            envelope = AgentEventEnvelope(
                seq=sequence,
                session_id=self.session_id,
                run_id=run_id,
                agent_id=agent_id,
                parent_agent_id=parent_agent_id,
                turn_id=turn_id,
                execution_mode=execution_mode,
                parent_tool_use_id=parent_tool_use_id,
                event=event,
                context_id=context_id,
            )
            result: asyncio.Future[AgentEventEnvelope] = asyncio.get_running_loop().create_future()
            publication = _PendingPublication(envelope, tuple(self._subscribers), result)
            if sequence in self._pending_publications or sequence < self._next_publication_seq:
                raise RuntimeError(f"logical sequence {sequence} was already published")
            self._pending_publications[sequence] = publication
            await self._drain_publications()
        return await asyncio.shield(result)

    async def publish_for(
        self,
        identity: TurnIdentity,
        event: RuntimeEvent,
        *,
        reserved_seq: int | None = None,
    ) -> AgentEventEnvelope:
        return await self.publish(
            event,
            run_id=identity.run_id,
            agent_id=identity.agent_id,
            parent_agent_id=identity.parent_agent_id,
            turn_id=identity.turn_id,
            execution_mode=identity.execution_mode,
            parent_tool_use_id=identity.parent_tool_use_id,
            context_id=identity.context_id,
            reserved_seq=reserved_seq,
        )

    def _claim_sequence(self, reserved_seq: int | None) -> int:
        with self._sequence_lock:
            if reserved_seq is None:
                self._seq += 1
                return self._seq
            if reserved_seq not in self._reserved_sequences:
                raise ValueError(f"logical sequence {reserved_seq} is not reserved")
            self._reserved_sequences.remove(reserved_seq)
            return reserved_seq

    async def _drain_publications(self) -> None:
        failure: BaseException | None = None
        while publication := self._pending_publications.pop(self._next_publication_seq, None):
            try:
                await self._deliver(publication)
            except BaseException as exc:
                if not publication.result.done():
                    publication.result.cancel()
                self._next_publication_seq += 1
                failure = failure or exc
                continue
            if not publication.result.done():
                publication.result.set_result(publication.envelope)
            self._next_publication_seq += 1
        if failure is not None:
            raise failure

    @staticmethod
    async def _deliver(publication: _PendingPublication) -> None:
        if not publication.subscribers:
            return
        results = await asyncio.gather(
            *(subscriber(publication.envelope) for subscriber in publication.subscribers),
            return_exceptions=True,
        )
        for subscriber, result in zip(publication.subscribers, results, strict=True):
            if isinstance(result, BaseException):
                logger.error("Agent event subscriber %r failed: %s", subscriber, result, exc_info=result)


def new_turn_identity(
    *,
    agent_id: str,
    parent_agent_id: str | None,
    execution_mode: ExecutionMode,
    parent_tool_use_id: str | None = None,
    run_id: str | None = None,
    context_id: str | None = None,
) -> TurnIdentity:
    return TurnIdentity(
        run_id=run_id or uuid4().hex,
        agent_id=agent_id,
        parent_agent_id=parent_agent_id,
        turn_id=uuid4().hex,
        execution_mode=execution_mode,
        parent_tool_use_id=parent_tool_use_id,
        context_id=context_id,
    )


class ObservedContextStore(ContextStore):
    """Publish successful context mutations through a session event hub."""

    def __init__(self, store: ContextStore, hub: SessionEventHub) -> None:
        self._store = store
        self._hub = hub
        self._default_identity: TurnIdentity | None = None

    @property
    def session_id(self) -> str:
        return self._store.session_id

    def bind_identity(self, identity: TurnIdentity) -> None:
        self._default_identity = replace(identity, context_id=self.session_id)

    def _identity(self) -> TurnIdentity | None:
        identity = current_turn_identity() or self._default_identity
        if identity is None:
            return None
        return replace(identity, context_id=self.session_id)

    async def _publish(self, event: RuntimeEvent) -> None:
        identity = self._identity()
        if identity is not None:
            await self._hub.publish_for(identity, event)

    async def append(self, message: Message) -> None:
        input_ids = _initial_message_input_ids.get()
        if input_ids is not None and len(input_ids) != 1:
            raise ValueError("initial message/input correlation count does not match append")
        source_input_id = input_ids[0] if input_ids is not None else None
        if input_ids is not None:
            _initial_message_input_ids.set(None)
        await self._store.append(message)
        await self._publish(MessageCommitted(message=copy.deepcopy(message), source_input_id=source_input_id))

    async def append_many(self, messages: list[Message]) -> None:
        committed = copy.deepcopy(messages)
        input_ids = _initial_message_input_ids.get()
        if input_ids is not None and len(input_ids) != len(committed):
            raise ValueError("initial message/input correlation count does not match append_many")
        if input_ids is not None:
            _initial_message_input_ids.set(None)
        else:
            input_ids = (None,) * len(committed)
        await self._store.append_many(messages)
        for message, source_input_id in zip(committed, input_ids, strict=True):
            await self._publish(MessageCommitted(message=message, source_input_id=source_input_id))

    async def get_history(self) -> list[Message]:
        return await self._store.get_history()

    async def clear(self) -> None:
        await self._store.clear()
        await self._publish(ContextCleared())

    async def fork(self) -> ObservedContextStore:
        child = ObservedContextStore(await self._store.fork(), self._hub)
        await self._publish(
            ContextForked(
                source_context_id=self.session_id,
                child_context_id=child.session_id,
            )
        )
        return child

    async def set_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        await self._store.set_context_tokens(input_tokens, output_tokens)

    async def get_context_tokens(self) -> tuple[int, int]:
        return await self._store.get_context_tokens()

    async def close(self) -> None:
        await self._store.close()

    async def list_sessions(self) -> list[SessionInfo]:
        return await self._store.list_sessions()


async def observe_agent_turn(
    *,
    agent: Agent,
    context: ContextStore,
    prompt: str,
    identity: TurnIdentity,
    hub: SessionEventHub,
) -> TurnOutcome:
    """Consume a complete turn and derive its outcome after the iterator closes."""

    if isinstance(context, ObservedContextStore):
        context.bind_identity(identity)
    identity_token = _current_turn_identity.set(identity)
    try:
        return await _observe_agent_turn_current(
            agent=agent,
            context=context,
            prompt=prompt,
            messages=None,
            identity=identity,
            hub=hub,
        )
    finally:
        _current_turn_identity.reset(identity_token)


async def observe_agent_turn_messages(
    *,
    agent: Agent,
    context: ContextStore,
    messages: Sequence[Message],
    identity: TurnIdentity,
    hub: SessionEventHub,
    on_input_committed: Callable[[], Awaitable[None]] | None = None,
    source_input_ids: Sequence[str | None] | None = None,
) -> TurnOutcome:
    """Consume one turn started from a distinct ordered Message batch."""

    if not messages:
        raise ValueError("messages must not be empty")
    captured = tuple(copy.deepcopy(messages))
    captured_input_ids = tuple(source_input_ids) if source_input_ids is not None else None
    if captured_input_ids is not None:
        if len(captured_input_ids) != len(captured):
            raise ValueError("source_input_ids must align with messages")
        if any(source_input_id == "" for source_input_id in captured_input_ids):
            raise ValueError("source_input_ids must not contain empty strings")
    if isinstance(context, ObservedContextStore):
        context.bind_identity(identity)
    identity_token = _current_turn_identity.set(identity)
    input_ids_token = _initial_message_input_ids.set(captured_input_ids)
    try:
        return await _observe_agent_turn_current(
            agent=agent,
            context=context,
            prompt=_summarize_input_messages(captured),
            messages=captured,
            identity=identity,
            hub=hub,
            on_input_committed=on_input_committed,
        )
    finally:
        _initial_message_input_ids.reset(input_ids_token)
        _current_turn_identity.reset(identity_token)


def _summarize_input_messages(messages: Sequence[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
    return "\n\n".join(parts)


async def _observe_agent_turn_current(
    *,
    agent: Agent,
    context: ContextStore,
    prompt: str,
    messages: tuple[Message, ...] | None,
    identity: TurnIdentity,
    hub: SessionEventHub,
    on_input_committed: Callable[[], Awaitable[None]] | None = None,
) -> TurnOutcome:
    history_boundary = len(await context.get_history())
    await hub.publish_for(identity, TurnStarted(prompt=prompt))
    stream = (
        agent.run_stream(prompt, context)
        if messages is None
        else agent.run_stream_messages(
            messages,
            context,
            on_input_committed=on_input_committed,
        )
    )
    text: list[str] = []
    current_iteration_text: list[str] = []
    observed_error: str | None = None
    session_end: SessionEndEvent | None = None
    raised: BaseException | None = None
    cancelled: asyncio.CancelledError | None = None

    try:
        async for event in stream:
            if isinstance(event, TextDelta):
                text.append(event.delta)
                current_iteration_text.append(event.delta)
            elif isinstance(event, ToolResult):
                current_iteration_text.clear()
            elif isinstance(event, Error):
                observed_error = f"{type(event.exception).__name__}: {event.exception}"
            elif isinstance(event, SessionEndEvent):
                session_end = event
            await hub.publish_for(identity, event)
    except asyncio.CancelledError as exc:
        cancelled = exc
    except BaseException as exc:
        raised = exc
    finally:
        try:
            await stream.aclose()
        except asyncio.CancelledError as exc:
            if cancelled is None:
                cancelled = exc
        except BaseException as exc:
            if raised is None:
                raised = exc

    result_text = "".join(text)
    if cancelled is not None:
        await _commit_partial_text(
            context,
            "".join(current_iteration_text),
            history_boundary=history_boundary,
        )
        await hub.publish_for(
            identity,
            TurnFinished(status=TurnStatus.CANCELLED, stop_reason=None, error="turn cancelled"),
        )
        raise cancelled

    if raised is not None:
        error = f"{type(raised).__name__}: {raised}"
        if observed_error != error:
            await hub.publish_for(identity, Error(exception=raised))
        outcome = TurnOutcome(
            identity=identity,
            status=TurnStatus.FAILED,
            text=result_text,
            stop_reason=session_end.stop_reason if session_end is not None else None,
            error=error,
        )
    elif observed_error is not None:
        outcome = TurnOutcome(
            identity=identity,
            status=TurnStatus.FAILED,
            text=result_text,
            stop_reason=session_end.stop_reason if session_end is not None else None,
            error=observed_error,
        )
    elif session_end is None:
        outcome = TurnOutcome(
            identity=identity,
            status=TurnStatus.FAILED,
            text=result_text,
            stop_reason=None,
            error="agent stream ended without SessionEndEvent",
        )
    elif session_end.stop_reason is not StopReason.end_turn:
        outcome = TurnOutcome(
            identity=identity,
            status=TurnStatus.FAILED,
            text=result_text,
            stop_reason=session_end.stop_reason,
            error=f"agent stopped with {session_end.stop_reason.value}",
        )
    else:
        outcome = TurnOutcome(
            identity=identity,
            status=TurnStatus.SUCCEEDED,
            text=result_text,
            stop_reason=session_end.stop_reason,
        )

    await hub.publish_for(
        identity,
        TurnFinished(status=outcome.status, stop_reason=outcome.stop_reason, error=outcome.error),
    )
    return outcome


async def _commit_partial_text(context: ContextStore, text: str, *, history_boundary: int) -> None:
    if not text:
        return
    history = await context.get_history()
    for message in history[history_boundary:]:
        if message.role != "assistant":
            continue
        committed_text = "".join(block.text for block in message.content if isinstance(block, TextBlock))
        if committed_text == text:
            return
    await context.append(Message(role="assistant", content=[TextBlock(text=text)]))
