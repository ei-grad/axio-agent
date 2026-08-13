"""Typed observation and lifecycle primitives for in-process agent runs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from axio.agent import Agent
from axio.context import ContextStore
from axio.events import Error, SessionEndEvent, StreamEvent, TextDelta
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


type RuntimeEvent = (
    StreamEvent
    | AgentStarted
    | AgentStopped
    | TurnStarted
    | TurnFinished
    | ForegroundEntered
    | ForegroundExited
    | OutcomeDelivered
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


@dataclass(frozen=True, slots=True)
class TurnIdentity:
    run_id: str
    agent_id: str
    parent_agent_id: str | None
    turn_id: str
    execution_mode: ExecutionMode
    parent_tool_use_id: str | None = None


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


class SessionEventHub:
    """Assign a session order and fan every runtime event out to all observers."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid4().hex
        self._seq = 0
        self._subscribers: list[EventSubscriber] = []
        self._publish_lock = asyncio.Lock()

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
    ) -> AgentEventEnvelope:
        async with self._publish_lock:
            self._seq += 1
            envelope = AgentEventEnvelope(
                seq=self._seq,
                session_id=self.session_id,
                run_id=run_id,
                agent_id=agent_id,
                parent_agent_id=parent_agent_id,
                turn_id=turn_id,
                execution_mode=execution_mode,
                parent_tool_use_id=parent_tool_use_id,
                event=event,
            )
            subscribers = tuple(self._subscribers)
            if subscribers:
                results = await asyncio.gather(
                    *(subscriber(envelope) for subscriber in subscribers),
                    return_exceptions=True,
                )
                for subscriber, result in zip(subscribers, results, strict=True):
                    if isinstance(result, BaseException):
                        logger.error("Agent event subscriber %r failed: %s", subscriber, result, exc_info=result)
            return envelope

    async def publish_for(self, identity: TurnIdentity, event: RuntimeEvent) -> AgentEventEnvelope:
        return await self.publish(
            event,
            run_id=identity.run_id,
            agent_id=identity.agent_id,
            parent_agent_id=identity.parent_agent_id,
            turn_id=identity.turn_id,
            execution_mode=identity.execution_mode,
            parent_tool_use_id=identity.parent_tool_use_id,
        )


def new_turn_identity(
    *,
    agent_id: str,
    parent_agent_id: str | None,
    execution_mode: ExecutionMode,
    parent_tool_use_id: str | None = None,
    run_id: str | None = None,
) -> TurnIdentity:
    return TurnIdentity(
        run_id=run_id or uuid4().hex,
        agent_id=agent_id,
        parent_agent_id=parent_agent_id,
        turn_id=uuid4().hex,
        execution_mode=execution_mode,
        parent_tool_use_id=parent_tool_use_id,
    )


async def observe_agent_turn(
    *,
    agent: Agent,
    context: ContextStore,
    prompt: str,
    identity: TurnIdentity,
    hub: SessionEventHub,
) -> TurnOutcome:
    """Consume a complete turn and derive its outcome after the iterator closes."""

    await hub.publish_for(identity, TurnStarted(prompt=prompt))
    stream = agent.run_stream(prompt, context)
    text: list[str] = []
    observed_error: str | None = None
    session_end: SessionEndEvent | None = None
    raised: BaseException | None = None
    cancelled: asyncio.CancelledError | None = None

    try:
        async for event in stream:
            if isinstance(event, TextDelta):
                text.append(event.delta)
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
