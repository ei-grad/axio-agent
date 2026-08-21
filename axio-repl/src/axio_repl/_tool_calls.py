"""Session-local tool-call identities and shared badge presentation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
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

    @property
    def marker(self) -> str:
        return f"#{self.ordinal:03d}"


@dataclass(frozen=True, slots=True)
class ToolResultDisplay:
    """Correlation status for one result without mutating its runtime event."""

    call: ToolCallDisplay
    event_name: str
    orphan: bool

    @property
    def name_mismatch(self) -> bool:
        return not self.orphan and self.call.name != self.event_name


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
            return ToolResultDisplay(call=call, event_name=event_name, orphan=True)
        return ToolResultDisplay(call=call, event_name=event_name, orphan=False)

    def complete(self, key: ToolCallKey) -> None:
        self._active.pop(key, None)

    def defer(self, key: ToolCallKey) -> ToolCallDisplay | None:
        call = self._active.pop(key, None)
        if call is not None:
            self._deferred[key] = call
        return call

    def take_deferred(self, key: ToolCallKey, event_name: str) -> ToolResultDisplay | None:
        call = self._deferred.pop(key, None)
        if call is None:
            return None
        return ToolResultDisplay(call=call, event_name=event_name, orphan=False)

    def discard_deferred(self) -> None:
        self._deferred.clear()

    def discard_turn(self, *, agent_id: str, run_id: str, turn_id: str) -> None:
        self._active = OrderedDict(
            (key, call)
            for key, call in self._active.items()
            if not (key.agent_id == agent_id and key.run_id == run_id and key.turn_id == turn_id)
        )

    def discard_agent(self, agent_id: str) -> None:
        self._active = OrderedDict((key, call) for key, call in self._active.items() if key.agent_id != agent_id)

    def _allocate(self, key: ToolCallKey, name: str) -> ToolCallDisplay:
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        return ToolCallDisplay(key=key, ordinal=ordinal, name=name)


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
