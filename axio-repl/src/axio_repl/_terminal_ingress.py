"""Thread-safe bounded ingress for serialized terminal output."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

RESET = "\033[0m"
MAX_PENDING_CHARS = 256 * 1024
MAX_BATCH_CHARS = 32 * 1024
MAX_LATE_CHARS = 64 * 1024

type OutputStream = Literal["stdout", "stderr"]


@dataclass(frozen=True, slots=True)
class OutputFrame:
    """One atomic write accepted by the terminal owner."""

    content: str
    stream: OutputStream = "stdout"


class IngressPhase(StrEnum):
    OPEN = "open"
    SEALED = "sealed"
    CLOSED = "closed"


class IngressDestination(StrEnum):
    ACTIVE = "active"
    LATE = "late"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class IngressAdmission:
    destination: IngressDestination
    wake_consumer: bool = False


@dataclass(frozen=True, slots=True)
class LateDrain:
    frames: tuple[OutputFrame, ...]
    dropped_frames: int
    dropped_chars: int


class TerminalIngress:
    """Own output pressure, wake scheduling, and the close barrier."""

    def __init__(
        self,
        *,
        max_pending_chars: int = MAX_PENDING_CHARS,
        max_batch_chars: int = MAX_BATCH_CHARS,
        max_late_chars: int = MAX_LATE_CHARS,
        reset: str = RESET,
    ) -> None:
        if min(max_pending_chars, max_batch_chars, max_late_chars) < 1:
            raise ValueError("terminal ingress limits must be positive")
        self._max_pending_chars = max_pending_chars
        self._max_batch_chars = max_batch_chars
        self._max_late_chars = max_late_chars
        self._reset = reset
        self._lock = threading.RLock()
        self._phase = IngressPhase.OPEN
        self._pending: deque[OutputFrame] = deque()
        self._pending_chars = 0
        self._dropped_frames = 0
        self._dropped_chars = 0
        self._late: deque[OutputFrame] = deque()
        self._late_chars = 0
        self._late_dropped_frames = 0
        self._late_dropped_chars = 0
        self._wake_scheduled = False
        self._writing = False

    @property
    def phase(self) -> IngressPhase:
        with self._lock:
            return self._phase

    @property
    def pending_char_count(self) -> int:
        with self._lock:
            return self._pending_chars

    @property
    def idle(self) -> bool:
        with self._lock:
            return self._idle_locked()

    @property
    def consumer_should_stop(self) -> bool:
        with self._lock:
            return self._phase is not IngressPhase.OPEN and self._idle_locked()

    def submit(self, frame: OutputFrame) -> IngressAdmission:
        if not frame.content:
            return IngressAdmission(IngressDestination.ACTIVE)
        with self._lock:
            if self._phase is IngressPhase.CLOSED:
                return IngressAdmission(IngressDestination.FALLBACK)
            if self._phase is IngressPhase.SEALED:
                self._enqueue_late_locked(frame)
                return IngressAdmission(IngressDestination.LATE)
            self._enqueue_active_locked(frame)
            wake = not self._wake_scheduled
            if wake:
                self._wake_scheduled = True
            return IngressAdmission(IngressDestination.ACTIVE, wake_consumer=wake)

    def wake_delivered(self) -> None:
        with self._lock:
            self._wake_scheduled = False

    def seal(self) -> bool:
        """Close active ingress and return whether the consumer needs waking."""

        with self._lock:
            if self._phase is not IngressPhase.OPEN:
                return False
            self._phase = IngressPhase.SEALED
            wake = not self._wake_scheduled
            if wake:
                self._wake_scheduled = True
            return wake

    def next_batch(self) -> str | None:
        with self._lock:
            if self._pending:
                batch = [self._pending.popleft()]
                size = len(batch[0].content)
                while self._pending and size + len(self._pending[0].content) <= self._max_batch_chars:
                    frame = self._pending.popleft()
                    batch.append(frame)
                    size += len(frame.content)
                self._pending_chars -= size
                self._writing = True
                return "".join(frame.content for frame in batch)
            if self._dropped_frames:
                dropped_frames = self._dropped_frames
                dropped_chars = self._dropped_chars
                self._dropped_frames = 0
                self._dropped_chars = 0
                self._writing = True
                return (
                    f"{self._reset}\n[terminal output skipped: {dropped_frames} frame(s), "
                    f"{dropped_chars} character(s)]\n"
                )
            self._writing = False
            return None

    def finish_batch(self) -> None:
        with self._lock:
            self._writing = False

    def fail(self) -> None:
        """Stop active delivery while retaining later writes for close fallback."""

        with self._lock:
            if self._phase is IngressPhase.CLOSED:
                return
            self._phase = IngressPhase.SEALED
            self._pending.clear()
            self._pending_chars = 0
            self._dropped_frames = 0
            self._dropped_chars = 0
            self._wake_scheduled = False
            self._writing = False

    def close_late(self) -> LateDrain:
        """Atomically switch retained wrappers to fallback and return late output."""

        with self._lock:
            if self._phase is IngressPhase.OPEN:
                raise RuntimeError("terminal ingress must be sealed before close")
            if self._phase is IngressPhase.CLOSED:
                return LateDrain((), 0, 0)
            self._phase = IngressPhase.CLOSED
            drain = LateDrain(
                frames=tuple(self._late),
                dropped_frames=self._late_dropped_frames,
                dropped_chars=self._late_dropped_chars,
            )
            self._late.clear()
            self._late_chars = 0
            self._late_dropped_frames = 0
            self._late_dropped_chars = 0
            return drain

    def _enqueue_active_locked(self, frame: OutputFrame) -> None:
        size = len(frame.content)
        if self._dropped_frames or self._pending_chars + size > self._max_pending_chars:
            self._dropped_frames += 1
            self._dropped_chars += size
            return
        if (
            self._pending
            and self._pending[-1].stream == frame.stream
            and len(self._pending[-1].content) + size <= self._max_batch_chars
        ):
            previous = self._pending[-1]
            self._pending[-1] = OutputFrame(previous.content + frame.content, frame.stream)
        else:
            self._pending.append(frame)
        self._pending_chars += size

    def _enqueue_late_locked(self, frame: OutputFrame) -> None:
        size = len(frame.content)
        if self._late_dropped_frames or self._late_chars + size > self._max_late_chars:
            self._late_dropped_frames += 1
            self._late_dropped_chars += size
            return
        self._late.append(frame)
        self._late_chars += size

    def _idle_locked(self) -> bool:
        return not self._pending and not self._dropped_frames and not self._writing and not self._wake_scheduled
