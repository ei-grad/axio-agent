"""Typed interactive input events and EOF arming state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class SubmissionDisposition(StrEnum):
    PENDING = "pending"
    COMMAND = "command"
    RETAINED = "retained"


@dataclass(frozen=True, slots=True)
class InputSubmitted:
    """Text submitted by Enter for later delivery."""

    text: str
    target_agent_id: str
    disposition: SubmissionDisposition
    input_id: str | None = None
    arrival_seq: int | None = None
    submitted_at: datetime | None = None
    author: str | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("submitted input must not be empty")
        if not self.target_agent_id:
            raise ValueError("target_agent_id must not be empty")
        correlated = self.input_id is not None and self.arrival_seq is not None
        if self.disposition is SubmissionDisposition.PENDING and not correlated:
            raise ValueError("pending submission requires input_id and arrival_seq")
        if self.disposition is SubmissionDisposition.COMMAND and self.input_id is not None:
            raise ValueError("command submissions may carry only an arrival sequence")
        if self.disposition is SubmissionDisposition.RETAINED and (
            self.input_id is not None or self.arrival_seq is not None
        ):
            raise ValueError("retained submissions cannot carry input correlation")
        if self.input_id == "":
            raise ValueError("input_id must not be empty")
        if self.arrival_seq is not None and self.arrival_seq < 1:
            raise ValueError("arrival_seq must be positive")
        if self.submitted_at is not None and (
            self.submitted_at.tzinfo is None or self.submitted_at.utcoffset() is None
        ):
            raise ValueError("submitted_at must be timezone-aware")
        if self.author == "":
            raise ValueError("submission author must not be empty")


@dataclass(frozen=True, slots=True)
class PendingRecallRequested:
    """Request that every still-pending user input be returned to the editor."""


@dataclass(frozen=True, slots=True)
class InterruptRequested:
    """Interrupt a turn without submitting or mutating editor text."""

    target_agent_id: str
    captured_turn_id: str | None

    def __post_init__(self) -> None:
        if not self.target_agent_id:
            raise ValueError("target_agent_id must not be empty")


@dataclass(frozen=True, slots=True)
class EndOfInput:
    """Ctrl-D received while the editor is empty."""

    monotonic_at: float


type PromptEvent = InputSubmitted | PendingRecallRequested | InterruptRequested | EndOfInput


@dataclass(frozen=True, slots=True)
class ExitArmingState:
    """Two-press Ctrl-D state driven by a monotonic clock."""

    deadline: float | None = None

    def press(self, now: float, *, window_seconds: float = 2.0) -> tuple[ExitArmingState, bool]:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.deadline is not None and now <= self.deadline:
            return ExitArmingState(), True
        return ExitArmingState(deadline=now + window_seconds), False

    def expire(self, now: float) -> ExitArmingState:
        if self.deadline is None or now <= self.deadline:
            return self
        return ExitArmingState()
