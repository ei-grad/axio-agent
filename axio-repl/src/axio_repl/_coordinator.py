"""Pure chronological input state for the interactive REPL coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from axio.blocks import TextBlock
from axio.messages import InputProvenance, Message
from axio_tools_agents.runtime import (
    AgentEventEnvelope,
    InputBuffered,
    InputClaimed,
    InputDelivered,
    InputRecalled,
    RuntimeEvent,
)


class PendingInputStatus(StrEnum):
    PENDING = "pending"
    RECALLED = "recalled"
    CLAIMED = "claimed"
    DELIVERED = "delivered"


class ForegroundPhase(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ForegroundOperation:
    target_agent_id: str
    turn_id: str

    def __post_init__(self) -> None:
        if not self.target_agent_id:
            raise ValueError("foreground target_agent_id must not be empty")
        if not self.turn_id:
            raise ValueError("foreground turn_id must not be empty")


@dataclass(frozen=True, slots=True)
class ForegroundCoordinatorState:
    """Pure lifecycle reducer for the one foreground operation owner."""

    phase: ForegroundPhase = ForegroundPhase.IDLE
    operation: ForegroundOperation | None = None
    interrupted_turns: frozenset[tuple[str, str]] = frozenset()
    shutdown_reason: str | None = None

    def __post_init__(self) -> None:
        if self.phase in {ForegroundPhase.RUNNING, ForegroundPhase.CANCELLING} and self.operation is None:
            raise ValueError(f"{self.phase.value} coordinator state requires an operation")
        if self.phase in {ForegroundPhase.IDLE, ForegroundPhase.STOPPED} and self.operation is not None:
            raise ValueError(f"{self.phase.value} coordinator state cannot retain an operation")
        if self.phase in {ForegroundPhase.STOPPING, ForegroundPhase.STOPPED}:
            if not self.shutdown_reason:
                raise ValueError(f"{self.phase.value} coordinator state requires a shutdown reason")
        elif self.shutdown_reason is not None:
            raise ValueError("non-shutdown coordinator state cannot carry a shutdown reason")

    def active_turn_id(self, target_agent_id: str) -> str | None:
        operation = self.operation
        if operation is None or operation.target_agent_id != target_agent_id:
            return None
        return operation.turn_id

    def start(self, target_agent_id: str, turn_id: str) -> ForegroundCoordinatorState:
        if self.phase is not ForegroundPhase.IDLE:
            raise RuntimeError(f"cannot start foreground operation from {self.phase.value} state")
        return ForegroundCoordinatorState(
            phase=ForegroundPhase.RUNNING,
            operation=ForegroundOperation(target_agent_id, turn_id),
            interrupted_turns=self.interrupted_turns,
        )

    def request_interrupt(
        self,
        target_agent_id: str,
        captured_turn_id: str | None,
    ) -> tuple[ForegroundCoordinatorState, bool]:
        """Return the next state and whether this key event has semantic effects."""

        if not target_agent_id:
            raise ValueError("interrupt target_agent_id must not be empty")
        if captured_turn_id is None:
            return self, True
        key = (target_agent_id, captured_turn_id)
        if key in self.interrupted_turns:
            return self, False
        interrupted = self.interrupted_turns | {key}
        operation = self.operation
        if (
            self.phase is ForegroundPhase.RUNNING
            and operation is not None
            and operation.target_agent_id == target_agent_id
            and operation.turn_id == captured_turn_id
        ):
            return (
                ForegroundCoordinatorState(
                    phase=ForegroundPhase.CANCELLING,
                    operation=operation,
                    interrupted_turns=interrupted,
                ),
                True,
            )
        return replace(self, interrupted_turns=interrupted), True

    def complete(self, target_agent_id: str, turn_id: str) -> ForegroundCoordinatorState:
        operation = self.operation
        if operation is None or operation != ForegroundOperation(target_agent_id, turn_id):
            return self
        if self.phase is ForegroundPhase.STOPPING:
            return replace(self, operation=None)
        return ForegroundCoordinatorState(
            phase=ForegroundPhase.IDLE,
            interrupted_turns=self.interrupted_turns,
        )

    def request_shutdown(self, reason: str) -> ForegroundCoordinatorState:
        if not reason:
            raise ValueError("shutdown reason must not be empty")
        if self.phase is ForegroundPhase.STOPPED:
            return self
        if self.phase is ForegroundPhase.STOPPING:
            if reason != self.shutdown_reason:
                raise RuntimeError("shutdown reason cannot change after stopping begins")
            return self
        return ForegroundCoordinatorState(
            phase=ForegroundPhase.STOPPING,
            operation=self.operation,
            interrupted_turns=self.interrupted_turns,
            shutdown_reason=reason,
        )

    def mark_stopped(self) -> ForegroundCoordinatorState:
        if self.phase is not ForegroundPhase.STOPPING:
            raise RuntimeError(f"cannot stop foreground coordinator from {self.phase.value} state")
        return ForegroundCoordinatorState(
            phase=ForegroundPhase.STOPPED,
            interrupted_turns=self.interrupted_turns,
            shutdown_reason=self.shutdown_reason,
        )


@dataclass(frozen=True, slots=True)
class PendingUserEntry:
    id: str
    arrival_seq: int
    text: str
    intended_target_agent_id: str
    submitted_at: datetime | None = None
    author: str | None = None
    status: PendingInputStatus = PendingInputStatus.PENDING
    claimed_target_agent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("pending input id must not be empty")
        if self.arrival_seq < 1:
            raise ValueError("arrival_seq must be positive")
        if not self.text:
            raise ValueError("pending input text must not be empty")
        if not self.intended_target_agent_id:
            raise ValueError("intended_target_agent_id must not be empty")
        if self.submitted_at is not None and (
            self.submitted_at.tzinfo is None or self.submitted_at.utcoffset() is None
        ):
            raise ValueError("pending input submitted_at must be timezone-aware")
        if self.author == "":
            raise ValueError("pending input author must not be empty")
        if self.status in {PendingInputStatus.CLAIMED, PendingInputStatus.DELIVERED}:
            if not self.claimed_target_agent_id:
                raise ValueError("claimed entries require claimed_target_agent_id")
        elif self.claimed_target_agent_id is not None:
            raise ValueError("unclaimed entries cannot have claimed_target_agent_id")


@dataclass(frozen=True, slots=True)
class RecallBatch:
    source_ids: tuple[str, ...]
    editor_text: str


@dataclass(frozen=True, slots=True)
class ClaimBatch:
    entries: tuple[PendingUserEntry, ...]
    target_agent_id: str

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("claim batch must not be empty")
        if not self.target_agent_id:
            raise ValueError("target_agent_id must not be empty")
        if any(entry.status is not PendingInputStatus.CLAIMED for entry in self.entries):
            raise ValueError("claim batch entries must be claimed")
        if any(entry.claimed_target_agent_id != self.target_agent_id for entry in self.entries):
            raise ValueError("claim batch entries must share the batch target")


@dataclass(frozen=True, slots=True)
class PendingInputState:
    """Append-only pending-input history with explicit tombstone states."""

    entries: tuple[PendingUserEntry, ...] = ()

    def __post_init__(self) -> None:
        ids = [entry.id for entry in self.entries]
        sequences = [entry.arrival_seq for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("pending input ids must be unique")
        if len(sequences) != len(set(sequences)):
            raise ValueError("pending input sequences must be unique")
        if sequences != sorted(sequences):
            raise ValueError("pending input entries must be ordered by arrival_seq")

    @property
    def pending(self) -> tuple[PendingUserEntry, ...]:
        return tuple(entry for entry in self.entries if entry.status is PendingInputStatus.PENDING)

    def admit(self, entry: PendingUserEntry) -> PendingInputState:
        if entry.status is not PendingInputStatus.PENDING:
            raise ValueError("only pending entries can be admitted")
        if any(existing.id == entry.id for existing in self.entries):
            raise ValueError(f"duplicate pending input id: {entry.id}")
        if self.entries and entry.arrival_seq <= self.entries[-1].arrival_seq:
            raise ValueError("arrival_seq must increase monotonically")
        return PendingInputState((*self.entries, entry))

    def recall_all(self) -> tuple[PendingInputState, RecallBatch | None]:
        selected = self.pending
        if not selected:
            return self, None
        selected_ids = frozenset(entry.id for entry in selected)
        entries = tuple(
            replace(entry, status=PendingInputStatus.RECALLED) if entry.id in selected_ids else entry
            for entry in self.entries
        )
        return PendingInputState(entries), RecallBatch(
            source_ids=tuple(entry.id for entry in selected),
            editor_text="\n\n".join(entry.text for entry in selected),
        )

    def claim_for_target(self, target_agent_id: str) -> tuple[PendingInputState, ClaimBatch | None]:
        return self._claim(
            tuple(entry for entry in self.pending if entry.intended_target_agent_id == target_agent_id),
            target_agent_id,
        )

    def claim_oldest(self) -> tuple[PendingInputState, ClaimBatch | None]:
        pending = self.pending
        if not pending:
            return self, None
        entry = pending[0]
        return self._claim((entry,), entry.intended_target_agent_id)

    def claim_all_for_interrupt(self, target_agent_id: str) -> tuple[PendingInputState, ClaimBatch | None]:
        return self._claim(self.pending, target_agent_id)

    def mark_delivered(self, source_ids: tuple[str, ...]) -> PendingInputState:
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("delivered source ids must be unique")
        selected = frozenset(source_ids)
        known = {entry.id for entry in self.entries}
        unknown = selected - known
        if unknown:
            raise ValueError(f"unknown pending input ids: {', '.join(sorted(unknown))}")
        for entry in self.entries:
            if entry.id in selected and entry.status is not PendingInputStatus.CLAIMED:
                raise ValueError(f"pending input {entry.id} is not claimed")
        return PendingInputState(
            tuple(
                replace(entry, status=PendingInputStatus.DELIVERED) if entry.id in selected else entry
                for entry in self.entries
            )
        )

    def _claim(
        self,
        selected: tuple[PendingUserEntry, ...],
        target_agent_id: str,
    ) -> tuple[PendingInputState, ClaimBatch | None]:
        if not target_agent_id:
            raise ValueError("target_agent_id must not be empty")
        if not selected:
            return self, None
        selected_ids = frozenset(entry.id for entry in selected)
        entries = tuple(
            replace(
                entry,
                status=PendingInputStatus.CLAIMED,
                claimed_target_agent_id=target_agent_id,
            )
            if entry.id in selected_ids
            else entry
            for entry in self.entries
        )
        claimed = tuple(entry for entry in entries if entry.id in selected_ids)
        return PendingInputState(entries), ClaimBatch(claimed, target_agent_id)


EventPublisher = Callable[[RuntimeEvent], Awaitable[AgentEventEnvelope]]
ReservedEventPublisher = Callable[[RuntimeEvent, int], Awaitable[AgentEventEnvelope]]


class PendingInputCoordinator:
    """Serialize durable pending-input transitions around the pure reducer."""

    def __init__(
        self,
        publish: EventPublisher,
        publish_reserved: ReservedEventPublisher | None = None,
    ) -> None:
        self._publish = publish
        self._publish_reserved = publish_reserved
        self._state = PendingInputState()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> PendingInputState:
        return self._state

    @property
    def pending_count(self) -> int:
        return len(self._state.pending)

    def pending_count_for_target(self, target_agent_id: str) -> int:
        return sum(entry.intended_target_agent_id == target_agent_id for entry in self._state.pending)

    async def wait_for_transitions(self) -> None:
        """Wait for a currently publishing input transition to update local state."""

        async with self._lock:
            pass

    async def admit(
        self,
        text: str,
        target_agent_id: str,
        *,
        reserved_seq: int | None = None,
        submitted_at: datetime | None = None,
        author: str | None = None,
    ) -> PendingUserEntry:
        input_id = uuid4().hex
        async with self._lock:
            event = InputBuffered(
                input_id=input_id,
                text=text,
                intended_target_agent_id=target_agent_id,
                submitted_at=submitted_at,
                author=author,
            )
            if reserved_seq is None:
                publication = self._publish(event)
            elif self._publish_reserved is None:
                raise RuntimeError("pending input coordinator has no reserved-sequence publisher")
            else:
                publication = self._publish_reserved(event, reserved_seq)
            envelope, cancellation = await self._await_publication(publication)
            entry = PendingUserEntry(
                id=input_id,
                arrival_seq=envelope.seq,
                text=text,
                intended_target_agent_id=target_agent_id,
                submitted_at=submitted_at,
                author=author,
            )
            self._state = self._state.admit(entry)
            if cancellation is not None:
                raise cancellation
            return entry

    async def recall_all(self) -> RecallBatch | None:
        async with self._lock:
            next_state, batch = self._state.recall_all()
            if batch is None:
                return None
            _, cancellation = await self._await_publication(
                self._publish(InputRecalled(input_ids=batch.source_ids, editor_text=batch.editor_text))
            )
            self._state = next_state
            if cancellation is not None:
                raise cancellation
            return batch

    async def claim_for_target(self, target_agent_id: str, *, reason: str) -> ClaimBatch | None:
        async with self._lock:
            next_state, batch = self._state.claim_for_target(target_agent_id)
            return await self._commit_claim(next_state, batch, reason)

    async def claim_oldest(self) -> ClaimBatch | None:
        async with self._lock:
            next_state, batch = self._state.claim_oldest()
            return await self._commit_claim(next_state, batch, "boundary")

    async def claim_all_for_interrupt(self, target_agent_id: str) -> ClaimBatch | None:
        async with self._lock:
            next_state, batch = self._state.claim_all_for_interrupt(target_agent_id)
            return await self._commit_claim(next_state, batch, "interrupt")

    async def mark_delivered(self, batch: ClaimBatch) -> None:
        source_ids = tuple(entry.id for entry in batch.entries)
        async with self._lock:
            next_state = self._state.mark_delivered(source_ids)
            _, cancellation = await self._await_publication(
                self._publish(InputDelivered(input_ids=source_ids, target_agent_id=batch.target_agent_id))
            )
            self._state = next_state
            if cancellation is not None:
                raise cancellation

    async def _commit_claim(
        self,
        next_state: PendingInputState,
        batch: ClaimBatch | None,
        reason: str,
    ) -> ClaimBatch | None:
        if batch is None:
            return None
        source_ids = tuple(entry.id for entry in batch.entries)
        _, cancellation = await self._await_publication(
            self._publish(
                InputClaimed(
                    input_ids=source_ids,
                    target_agent_id=batch.target_agent_id,
                    reason=reason,
                )
            )
        )
        self._state = next_state
        if cancellation is not None:
            raise cancellation
        return batch

    async def _await_publication(
        self,
        publication: Awaitable[AgentEventEnvelope],
    ) -> tuple[AgentEventEnvelope, asyncio.CancelledError | None]:
        publication_task = asyncio.ensure_future(publication)
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                envelope = await asyncio.shield(publication_task)
                return envelope, cancellation
            except asyncio.CancelledError as exc:
                if publication_task.done():
                    return publication_task.result(), cancellation or exc
                cancellation = cancellation or exc


@dataclass(frozen=True, slots=True)
class ContextArrival:
    seq: int
    target_agent_id: str
    message: Message
    source: str
    source_input_id: str | None = None

    def __post_init__(self) -> None:
        if self.seq < 1:
            raise ValueError("context arrival seq must be positive")
        if not self.target_agent_id:
            raise ValueError("target_agent_id must not be empty")
        if not self.source:
            raise ValueError("context arrival source must not be empty")
        if self.source_input_id == "":
            raise ValueError("source_input_id must not be empty")


def claim_batch_arrivals(batch: ClaimBatch) -> tuple[ContextArrival, ...]:
    """Preserve every claimed input as a distinct Message and arrival."""

    return tuple(
        ContextArrival(
            seq=entry.arrival_seq,
            target_agent_id=batch.target_agent_id,
            message=Message(
                role="user",
                content=[TextBlock(text=entry.text)],
                provenance=InputProvenance(
                    human_authored=True,
                    source="interactive",
                    author=entry.author or "human",
                    submitted_at=entry.submitted_at,
                ),
            ),
            source="interactive",
            source_input_id=entry.id,
        )
        for entry in batch.entries
    )


def ordered_messages(
    arrivals: tuple[ContextArrival, ...],
    target_agent_id: str,
    *,
    through_seq: int | None = None,
) -> tuple[Message, ...]:
    """Materialize one target's arrivals without joining or reordering Messages."""

    return tuple(arrival.message for arrival in ordered_arrivals(arrivals, target_agent_id, through_seq=through_seq))


def ordered_arrivals(
    arrivals: tuple[ContextArrival, ...],
    target_agent_id: str,
    *,
    through_seq: int | None = None,
) -> tuple[ContextArrival, ...]:
    """Select one target's arrivals in logical session order."""

    selected = [
        arrival
        for arrival in arrivals
        if arrival.target_agent_id == target_agent_id and (through_seq is None or arrival.seq <= through_seq)
    ]
    sequences = [arrival.seq for arrival in selected]
    if len(sequences) != len(set(sequences)):
        raise ValueError("context arrival sequences must be unique per target")
    return tuple(sorted(selected, key=lambda arrival: arrival.seq))
