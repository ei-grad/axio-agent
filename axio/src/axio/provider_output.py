"""Bounded safety checks for one completion-provider response stream."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .events import ReasoningDelta, Refusal, StreamEvent, TextDelta, ToolInputDelta
from .exceptions import ProviderOutputLimitError
from .messages import Message
from .tool import Tool
from .transport import CompletionTransport, OutputTokenLimitSource

type _SemanticStreamKey = tuple[Literal["text", "reasoning", "refusal"], int] | tuple[Literal["tool-input"], int, str]


def snapshot_output_token_limit(
    transport: CompletionTransport,
    messages: list[Message],
    tools: list[Tool[Any]],
    system: str,
) -> int | None:
    """Snapshot the effective request limit before consuming its stream."""

    def positive_int(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    if isinstance(transport, OutputTokenLimitSource):
        if limit := positive_int(transport.output_token_limit(messages, tools, system)):
            return limit
    if limit := positive_int(getattr(transport, "max_output_tokens", None)):
        return limit
    return positive_int(getattr(getattr(transport, "model", None), "max_output_tokens", None))


@dataclass(frozen=True, slots=True)
class ProviderOutputPolicy:
    """Circuit-breaker thresholds applied independently to each provider call.

    Byte limits count the UTF-8 representation of text, reasoning, and raw tool
    argument deltas. Tool execution output is outside the provider response and
    is deliberately not counted.
    """

    max_response_bytes: int | None = 512 * 1024
    max_bytes_per_output_token: int = 32
    output_token_overhead_bytes: int = 16 * 1024
    sustained_rate_bytes_per_second: int | None = 64 * 1024
    rate_burst_bytes: int = 256 * 1024
    cumulative_snapshot_min_prefix_chars: int = 2 * 1024
    cumulative_snapshot_prefix_ratio: float = 0.9
    cumulative_snapshot_streak: int = 2

    def __post_init__(self) -> None:
        optional_positive = {
            "max_response_bytes": self.max_response_bytes,
            "sustained_rate_bytes_per_second": self.sustained_rate_bytes_per_second,
        }
        for name, value in optional_positive.items():
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be positive or None")
        positive = {
            "max_bytes_per_output_token": self.max_bytes_per_output_token,
            "rate_burst_bytes": self.rate_burst_bytes,
            "cumulative_snapshot_min_prefix_chars": self.cumulative_snapshot_min_prefix_chars,
            "cumulative_snapshot_streak": self.cumulative_snapshot_streak,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            not isinstance(self.output_token_overhead_bytes, int)
            or isinstance(self.output_token_overhead_bytes, bool)
            or self.output_token_overhead_bytes < 0
        ):
            raise ValueError("output_token_overhead_bytes must be non-negative")
        if not 0 < self.cumulative_snapshot_prefix_ratio <= 1:
            raise ValueError("cumulative_snapshot_prefix_ratio must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class _SnapshotCandidate:
    key: _SemanticStreamKey
    value: str
    streak: int


class ProviderOutputGuard:
    """Inspect provider deltas before callers observe or retain them."""

    __slots__ = (
        "_accepted_events",
        "_clock",
        "_effective_output_tokens",
        "_last_rate_check",
        "_policy",
        "_rate_tokens",
        "_snapshot_streaks",
        "_snapshot_values",
        "_total_bytes",
    )

    def __init__(
        self,
        policy: ProviderOutputPolicy,
        *,
        effective_output_tokens: int | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy = policy
        self._effective_output_tokens = (
            effective_output_tokens
            if effective_output_tokens is not None
            and not isinstance(effective_output_tokens, bool)
            and effective_output_tokens > 0
            else None
        )
        self._clock = clock
        self._total_bytes = 0
        self._accepted_events = 0
        self._snapshot_values: dict[_SemanticStreamKey, str] = {}
        self._snapshot_streaks: dict[_SemanticStreamKey, int] = {}
        self._last_rate_check = clock()
        self._rate_tokens = float(policy.rate_burst_bytes)

    @property
    def accepted_bytes(self) -> int:
        return self._total_bytes

    def inspect(self, event: StreamEvent) -> ProviderOutputLimitError | None:
        """Accept an event or return the bounded error that rejects it."""

        extracted = self._extract_delta(event)
        if extracted is None:
            return None
        value, key = extracted
        if not value:
            return None
        event_bytes = len(value.encode("utf-8", errors="surrogatepass"))
        candidate_total = self._total_bytes + event_bytes

        if self._policy.max_response_bytes is not None and candidate_total > self._policy.max_response_bytes:
            return ProviderOutputLimitError(
                "Provider response exceeded the configured decoded-byte safety limit "
                f"({candidate_total} > {self._policy.max_response_bytes} bytes)",
                note="\n\n[Output truncated: provider response exceeded the safety size limit]",
            )

        token_byte_limit = self._token_byte_limit()
        if token_byte_limit is not None and candidate_total > token_byte_limit:
            assert self._effective_output_tokens is not None
            return ProviderOutputLimitError(
                "Provider response exceeded the expected envelope for the request's output token limit "
                f"({candidate_total} > {token_byte_limit} decoded bytes for "
                f"{self._effective_output_tokens} output tokens)",
                note="\n\n[Output truncated: provider response exceeded the expected request size]",
            )

        snapshot = self._snapshot_candidate(key, value)
        if snapshot.streak >= self._policy.cumulative_snapshot_streak:
            return ProviderOutputLimitError(
                "Provider response emitted cumulative snapshot deltas on one semantic output stream",
                note="\n\n[Output truncated: provider sent cumulative response snapshots]",
            )

        now = self._clock()
        rate_tokens = self._available_rate_tokens(now)
        if (
            self._policy.sustained_rate_bytes_per_second is not None
            and self._accepted_events > 0
            and event_bytes > rate_tokens
        ):
            return ProviderOutputLimitError(
                "Provider response exceeded the sustained decoded-byte rate safety limit "
                f"({self._policy.sustained_rate_bytes_per_second} bytes/s with a "
                f"{self._policy.rate_burst_bytes}-byte burst)",
                note="\n\n[Output truncated: provider response arrived at an unsafe rate]",
            )

        self._total_bytes = candidate_total
        self._accepted_events += 1
        self._commit_rate(now, rate_tokens, event_bytes)
        self._snapshot_values[snapshot.key] = snapshot.value
        self._snapshot_streaks[snapshot.key] = snapshot.streak
        return None

    def _token_byte_limit(self) -> int | None:
        if self._effective_output_tokens is None:
            return None
        return (
            self._policy.output_token_overhead_bytes
            + self._effective_output_tokens * self._policy.max_bytes_per_output_token
        )

    def _available_rate_tokens(self, now: float) -> float:
        rate = self._policy.sustained_rate_bytes_per_second
        if rate is None:
            return self._rate_tokens
        elapsed = max(0.0, now - self._last_rate_check)
        return min(float(self._policy.rate_burst_bytes), self._rate_tokens + elapsed * rate)

    def _commit_rate(self, now: float, available: float, event_bytes: int) -> None:
        if self._policy.sustained_rate_bytes_per_second is None:
            return
        self._last_rate_check = max(self._last_rate_check, now)
        # A provider may buffer its first frame. Accept that one frame subject to
        # the response-size limits, but let it consume the whole available burst.
        self._rate_tokens = max(0.0, available - event_bytes)

    def _snapshot_candidate(self, key: _SemanticStreamKey, value: str) -> _SnapshotCandidate:
        previous = self._snapshot_values.get(key)
        streak = 0
        if (
            previous is not None
            and len(previous) >= self._policy.cumulative_snapshot_min_prefix_chars
            and len(value) > len(previous)
        ):
            shared = self._shared_prefix_length(previous, value)
            if shared / len(previous) >= self._policy.cumulative_snapshot_prefix_ratio:
                streak = self._snapshot_streaks.get(key, 0) + 1
        return _SnapshotCandidate(key=key, value=value, streak=streak)

    @staticmethod
    def _shared_prefix_length(left: str, right: str) -> int:
        limit = min(len(left), len(right))
        for index in range(limit):
            if left[index] != right[index]:
                return index
        return limit

    @staticmethod
    def _extract_delta(event: StreamEvent) -> tuple[str, _SemanticStreamKey] | None:
        if isinstance(event, TextDelta):
            return event.delta, ("text", event.index)
        if isinstance(event, ReasoningDelta):
            return event.delta, ("reasoning", event.index)
        if isinstance(event, Refusal):
            return event.text, ("refusal", event.index)
        if isinstance(event, ToolInputDelta):
            return event.partial_json, ("tool-input", event.index, event.tool_use_id)
        return None
