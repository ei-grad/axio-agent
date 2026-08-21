"""Semantic terminal themes for the inline REPL."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

RESET: Final = "\033[0m"


@dataclass(frozen=True, slots=True)
class TerminalColor:
    """One colour expressed for ANSI SGR and prompt-toolkit."""

    ansi_foreground: int
    ansi_background: int
    prompt_toolkit: str


@dataclass(frozen=True, slots=True)
class TextStyle:
    """One semantic text style for ANSI and prompt-toolkit consumers."""

    ansi: str = ""
    prompt_toolkit: str = ""


@dataclass(frozen=True, slots=True)
class PowerlineStyle:
    """Foreground and fill colours for one Powerline segment role."""

    foreground: TerminalColor
    background: TerminalColor


@dataclass(frozen=True, slots=True)
class TerminalTheme:
    """Immutable semantic palette shared by every terminal rendering path."""

    name: str
    reset: str
    prompt: TextStyle
    panel: str
    emphasis: TextStyle
    command: TextStyle
    tool: TextStyle
    agent: TextStyle
    action: TextStyle
    reasoning: TextStyle
    stdout: TextStyle
    stderr: TextStyle
    error: TextStyle
    success: TextStyle
    warning: TextStyle
    prompt_badge: PowerlineStyle
    tool_badge: PowerlineStyle
    success_badge: PowerlineStyle
    error_badge: PowerlineStyle
    agent_badge: PowerlineStyle
    action_badge: PowerlineStyle


_BLACK = TerminalColor(30, 40, "ansiblack")
_GRAY = TerminalColor(37, 47, "ansigray")
_BRIGHT_BLACK = TerminalColor(90, 100, "ansibrightblack")
_WHITE = TerminalColor(97, 107, "ansiwhite")
_CYAN = TerminalColor(36, 46, "ansicyan")
_GREEN = TerminalColor(32, 42, "ansigreen")
_RED = TerminalColor(31, 41, "ansired")
_MAGENTA = TerminalColor(35, 45, "ansimagenta")
_YELLOW = TerminalColor(33, 43, "ansiyellow")

DEFAULT_THEME = TerminalTheme(
    name="default",
    reset=RESET,
    prompt=TextStyle("\033[1;30;107m", "bold fg:ansiblack bg:ansiwhite"),
    panel="noreverse bg:default fg:#808080",
    emphasis=TextStyle("\033[1m", "bold"),
    command=TextStyle("\033[1m", "bold"),
    tool=TextStyle("\033[1m\033[36m", "bold ansicyan"),
    agent=TextStyle("\033[2m", "dim"),
    action=TextStyle(),
    reasoning=TextStyle("\033[2m", "dim"),
    stdout=TextStyle("\033[2m", "dim"),
    stderr=TextStyle("\033[2;33m", "dim ansiyellow"),
    error=TextStyle("\033[31m", "ansired"),
    success=TextStyle("\033[32m", "ansigreen"),
    warning=TextStyle("\033[33m", "ansiyellow"),
    prompt_badge=PowerlineStyle(_BLACK, _WHITE),
    tool_badge=PowerlineStyle(_BLACK, _CYAN),
    success_badge=PowerlineStyle(_BLACK, _GREEN),
    error_badge=PowerlineStyle(_WHITE, _RED),
    agent_badge=PowerlineStyle(_WHITE, _MAGENTA),
    action_badge=PowerlineStyle(_BLACK, _YELLOW),
)

MONOCHROME_THEME = TerminalTheme(
    name="monochrome",
    reset=RESET,
    prompt=TextStyle("\033[1;97m", "bold ansiwhite"),
    panel="noreverse bg:default fg:ansiwhite",
    emphasis=TextStyle("\033[1;97m", "bold ansiwhite"),
    command=TextStyle("\033[1m", "bold"),
    tool=TextStyle("\033[1;97m", "bold ansiwhite"),
    agent=TextStyle("\033[1;97m", "bold ansiwhite"),
    action=TextStyle("\033[1;97m", "bold ansiwhite"),
    reasoning=TextStyle("\033[2;97m", "dim ansiwhite"),
    stdout=TextStyle("\033[37m", "ansigray"),
    stderr=TextStyle("\033[1;97m", "bold ansiwhite"),
    error=TextStyle("\033[1;7m", "bold reverse"),
    success=TextStyle("\033[1;97m", "bold ansiwhite"),
    warning=TextStyle("\033[4;97m", "underline ansiwhite"),
    prompt_badge=PowerlineStyle(_BLACK, _WHITE),
    tool_badge=PowerlineStyle(_BLACK, _WHITE),
    success_badge=PowerlineStyle(_BLACK, _WHITE),
    error_badge=PowerlineStyle(_BLACK, _WHITE),
    agent_badge=PowerlineStyle(_WHITE, _BRIGHT_BLACK),
    action_badge=PowerlineStyle(_BLACK, _GRAY),
)

NO_COLOR_THEME = TerminalTheme(
    name="no-color",
    reset="",
    prompt=TextStyle(),
    panel="",
    emphasis=TextStyle(),
    command=TextStyle(),
    tool=TextStyle(),
    agent=TextStyle(),
    action=TextStyle(),
    reasoning=TextStyle(),
    stdout=TextStyle(),
    stderr=TextStyle(),
    error=TextStyle(),
    success=TextStyle(),
    warning=TextStyle(),
    prompt_badge=DEFAULT_THEME.prompt_badge,
    tool_badge=DEFAULT_THEME.tool_badge,
    success_badge=DEFAULT_THEME.success_badge,
    error_badge=DEFAULT_THEME.error_badge,
    agent_badge=DEFAULT_THEME.agent_badge,
    action_badge=DEFAULT_THEME.action_badge,
)

_BUILTIN_THEMES = MappingProxyType(
    {
        DEFAULT_THEME.name: DEFAULT_THEME,
        MONOCHROME_THEME.name: MONOCHROME_THEME,
    }
)


def theme_names() -> tuple[str, ...]:
    """Return stable built-in names suitable for configuration help."""

    return tuple(_BUILTIN_THEMES)


def resolve_theme(name: str) -> TerminalTheme:
    """Resolve an exact built-in name or fail with the available set."""

    try:
        return _BUILTIN_THEMES[name]
    except KeyError as exc:
        available = ", ".join(theme_names())
        raise ValueError(f"unknown theme {name!r}; available themes: {available}") from exc


def resolve_terminal_presentation(
    theme: TerminalTheme,
    *,
    powerline: bool,
    one_shot: bool,
    stdout_is_tty: bool,
    no_color: bool,
) -> tuple[TerminalTheme, bool]:
    """Apply accessibility and output-medium constraints to a selected theme."""

    if no_color or (one_shot and not stdout_is_tty):
        return NO_COLOR_THEME, False
    return theme, powerline
