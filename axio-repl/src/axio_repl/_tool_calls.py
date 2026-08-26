"""Session-local tool-call identities and shared badge presentation."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from axio_repl._powerline import PowerlineBadge, PowerlineSegment
from axio_repl._terminal_sanitizer import sanitize_terminal_text
from axio_repl._theme import DEFAULT_THEME, PowerlineStyle, TerminalTheme, TextStyle


@dataclass(frozen=True, slots=True)
class ToolCallKey:
    """A provider call id scoped to the runtime turn that owns it."""

    agent_id: str
    run_id: str
    turn_id: str
    tool_use_id: str


@dataclass(frozen=True, slots=True)
class ToolCallDisplay:
    """Stable session-local presentation metadata for one tool call."""

    key: ToolCallKey
    ordinal: int
    name: str
    name_identity: bytes

    @property
    def marker(self) -> str:
        return f"#{self.ordinal:03d}"

    def name_matches(self, value: str) -> bool:
        return self.name_identity == _tool_name_identity(value)


@dataclass(frozen=True, slots=True)
class ToolResultDisplay:
    """Correlation status for one result without mutating its runtime event."""

    call: ToolCallDisplay
    event_name: str
    orphan: bool
    name_mismatch: bool


@dataclass(frozen=True, slots=True)
class ToolExecutionTiming:
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float | None = None


class ToolBadgeKind(StrEnum):
    CALL = "call"
    SUCCESS = "success"
    ERROR = "error"

    @property
    def glyph(self) -> str:
        return {
            ToolBadgeKind.CALL: "▶",
            ToolBadgeKind.SUCCESS: "✓",
            ToolBadgeKind.ERROR: "✗",
        }[self]


class ToolCallRegistry:
    """Allocate monotonic ordinals and retain only currently active calls."""

    def __init__(self) -> None:
        self._next_ordinal = 1
        self._active: OrderedDict[ToolCallKey, ToolCallDisplay] = OrderedDict()
        self._deferred: OrderedDict[ToolCallKey, ToolCallDisplay] = OrderedDict()
        self._timings: dict[ToolCallKey, ToolExecutionTiming] = {}

    @property
    def active_count(self) -> int:
        return len(self._active) + len(self._deferred)

    @property
    def next_ordinal(self) -> int:
        return self._next_ordinal

    def start(self, key: ToolCallKey, name: str) -> ToolCallDisplay:
        existing = self._active.get(key)
        if existing is not None:
            return existing
        call = self._allocate(key, name)
        self._active[key] = call
        return call

    def result(self, key: ToolCallKey, event_name: str) -> ToolResultDisplay:
        call = self._active.get(key)
        if call is None:
            call = self._allocate(key, event_name)
            self._active[key] = call
            return ToolResultDisplay(
                call=call,
                event_name=call.name,
                orphan=True,
                name_mismatch=False,
            )
        return ToolResultDisplay(
            call=call,
            event_name=tool_display_name(event_name),
            orphan=False,
            name_mismatch=not call.name_matches(event_name),
        )

    def complete(self, key: ToolCallKey) -> None:
        self._active.pop(key, None)
        self._timings.pop(key, None)

    def record_result_timing(
        self,
        key: ToolCallKey,
        *,
        started_at: datetime | None,
        finished_at: datetime | None,
        duration_seconds: float | None,
    ) -> None:
        if (
            started_at is not None
            and finished_at is not None
            and duration_seconds is not None
            and (key in self._active or key in self._deferred)
        ):
            self._timings[key] = ToolExecutionTiming(started_at, finished_at, max(0.0, duration_seconds))

    def result_marker(self, call: ToolCallDisplay) -> str:
        timing = self._timings.get(call.key)
        if timing is None or timing.finished_at is None or timing.duration_seconds is None:
            return call.marker
        return format_tool_result_marker(
            call.marker,
            started_at=timing.started_at,
            finished_at=timing.finished_at,
            duration_seconds=timing.duration_seconds,
        )

    def defer(self, key: ToolCallKey) -> ToolCallDisplay | None:
        call = self._active.pop(key, None)
        if call is not None:
            self._deferred[key] = call
        return call

    def take_deferred(self, key: ToolCallKey, event_name: str) -> ToolResultDisplay | None:
        call = self._deferred.pop(key, None)
        if call is None:
            return None
        self._timings.pop(key, None)
        return ToolResultDisplay(
            call=call,
            event_name=tool_display_name(event_name),
            orphan=False,
            name_mismatch=not call.name_matches(event_name),
        )

    def discard_deferred(self) -> None:
        self._deferred.clear()
        self._timings = {key: timing for key, timing in self._timings.items() if key in self._active}

    def discard_turn(self, *, agent_id: str, run_id: str, turn_id: str) -> None:
        discarded = {
            key for key in self._active if key.agent_id == agent_id and key.run_id == run_id and key.turn_id == turn_id
        }
        self._active = OrderedDict(
            (key, call)
            for key, call in self._active.items()
            if not (key.agent_id == agent_id and key.run_id == run_id and key.turn_id == turn_id)
        )
        for key in discarded:
            self._timings.pop(key, None)

    def discard_agent(self, agent_id: str) -> None:
        discarded = {key for key in self._active if key.agent_id == agent_id}
        self._active = OrderedDict((key, call) for key, call in self._active.items() if key.agent_id != agent_id)
        for key in discarded:
            self._timings.pop(key, None)

    def _allocate(self, key: ToolCallKey, name: str) -> ToolCallDisplay:
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        return ToolCallDisplay(
            key=key,
            ordinal=ordinal,
            name=tool_display_name(name),
            name_identity=_tool_name_identity(name),
        )


def tool_badge(
    kind: ToolBadgeKind,
    name: str,
    marker: str,
    *,
    powerline: bool,
    theme: TerminalTheme = DEFAULT_THEME,
) -> str:
    """Render one reset-safe call or response badge."""

    safe_name = tool_display_name(name)
    label = f"{kind.glyph} {safe_name} {marker}"
    if powerline and theme.reset:
        badge_style = _powerline_style(kind, theme)
        return PowerlineBadge((PowerlineSegment(f" {label} ", badge_style.foreground, badge_style.background),)).ansi()
    text_style = _text_style(kind, theme)
    return f"{text_style.ansi}{label}{theme.reset}"


def format_tool_result_marker(
    marker: str,
    *,
    started_at: datetime | None,
    finished_at: datetime | None,
    duration_seconds: float | None,
) -> str:
    """Add execution bounds to a result marker when the dispatch measured them."""

    if started_at is None or finished_at is None or duration_seconds is None:
        return marker
    start = started_at.astimezone().strftime("%H:%M:%S")
    finish = finished_at.astimezone().strftime("%H:%M:%S")
    duration = f"{duration_seconds * 1000:.0f}ms" if duration_seconds < 1 else f"{duration_seconds:.2f}s"
    return f"{marker} · {start}→{finish} · {duration}"


def tool_display_name(value: object) -> str:
    """Return one bounded, valid-Unicode terminal-safe tool name."""

    clean_name = " ".join(sanitize_terminal_text(value).split()) or "tool"
    return _fit_utf8(_replace_surrogates(clean_name), 80)


def _text_style(kind: ToolBadgeKind, theme: TerminalTheme) -> TextStyle:
    if kind is ToolBadgeKind.CALL:
        return theme.tool
    if kind is ToolBadgeKind.SUCCESS:
        return theme.success
    return theme.error


def _powerline_style(kind: ToolBadgeKind, theme: TerminalTheme) -> PowerlineStyle:
    if kind is ToolBadgeKind.CALL:
        return theme.tool_badge
    if kind is ToolBadgeKind.SUCCESS:
        return theme.success_badge
    return theme.error_badge


def _fit_utf8(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    suffix = "…"
    prefix = encoded[: limit - len(suffix.encode("utf-8"))].decode("utf-8", errors="ignore")
    return prefix + suffix


def _replace_surrogates(value: str) -> str:
    return "".join("\ufffd" if 0xD800 <= ord(character) <= 0xDFFF else character for character in value)


def _tool_name_identity(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8", errors="surrogatepass"), digest_size=16).digest()
