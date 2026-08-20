"""Powerline badges shared by prompt-toolkit and streamed terminal output."""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.formatted_text import FormattedText

from axio_repl._theme import DEFAULT_THEME, TerminalColor, TerminalTheme
from axio_repl._theme import RESET as RESET

POWERLINE_RIGHT = "\ue0b0"


@dataclass(frozen=True, slots=True)
class PowerlineSegment:
    """Text and colours for one filled segment."""

    text: str
    foreground: TerminalColor
    background: TerminalColor


@dataclass(frozen=True, slots=True)
class PowerlineBadge:
    """A filled badge with a straight left edge and hard right separators."""

    segments: tuple[PowerlineSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a Powerline badge requires at least one segment")

    def ansi(self) -> str:
        """Render a reset-safe ANSI badge on the terminal's default background."""

        parts: list[str] = []
        for index, segment in enumerate(self.segments):
            parts.append(
                f"\033[1;{segment.foreground.ansi_foreground};{segment.background.ansi_background}m{segment.text}"
            )
            next_background = self.segments[index + 1].background if index + 1 < len(self.segments) else None
            edge_background = next_background.ansi_background if next_background is not None else 49
            parts.append(f"\033[22;{segment.background.ansi_foreground};{edge_background}m{POWERLINE_RIGHT}")
        parts.append(RESET)
        return "".join(parts)

    def formatted_text(self, *, trailing: str = "") -> FormattedText:
        """Render prompt-toolkit fragments with the same fills and separators."""

        parts: list[tuple[str, str]] = []
        for index, segment in enumerate(self.segments):
            parts.append(
                (
                    f"bold fg:{segment.foreground.prompt_toolkit} bg:{segment.background.prompt_toolkit}",
                    segment.text,
                )
            )
            next_background = self.segments[index + 1].background if index + 1 < len(self.segments) else None
            edge_background = next_background.prompt_toolkit if next_background is not None else "default"
            edge_style = f"fg:{segment.background.prompt_toolkit} bg:{edge_background}"
            parts.append((edge_style, POWERLINE_RIGHT))
        if trailing:
            parts.append(("", trailing))
        return FormattedText(parts)


def prompt_badge(label: str, theme: TerminalTheme = DEFAULT_THEME) -> FormattedText:
    """Return the complete input badge followed by one editor-space."""

    style = theme.prompt_badge
    badge = PowerlineBadge((PowerlineSegment(f" {label} ", style.foreground, style.background),))
    return badge.formatted_text(trailing=" ")


def submitted_prompt_badge(label: str, theme: TerminalTheme = DEFAULT_THEME) -> str:
    """Return the reset-safe prompt badge used for accepted scrollback."""

    style = theme.prompt_badge
    return PowerlineBadge((PowerlineSegment(f" {label} ", style.foreground, style.background),)).ansi()


def tool_title(name: str, theme: TerminalTheme = DEFAULT_THEME) -> str:
    """Return a reset-safe Powerline badge for a live tool call."""

    style = theme.tool_badge
    return PowerlineBadge((PowerlineSegment(f" ▶ {name} ", style.foreground, style.background),)).ansi()


def agent_header(identity: str, theme: TerminalTheme = DEFAULT_THEME) -> str:
    """Return a reset-safe Powerline badge for a live agent turn."""

    style = theme.agent_badge
    return PowerlineBadge((PowerlineSegment(f" agent {identity} ", style.foreground, style.background),)).ansi()


def action_frame_header(identity: str, kind: str, theme: TerminalTheme = DEFAULT_THEME) -> str:
    """Return the reset-safe two-segment badge of a background action."""

    agent_style = theme.agent_badge
    action_style = theme.action_badge
    return PowerlineBadge(
        (
            PowerlineSegment(f" agent {identity} ", agent_style.foreground, agent_style.background),
            PowerlineSegment(f" {kind} ", action_style.foreground, action_style.background),
        )
    ).ansi()


def action_frame_footer(identity: str, theme: TerminalTheme = DEFAULT_THEME) -> str:
    """Return the reset-safe closing badge of a background action."""

    style = theme.agent_badge
    return PowerlineBadge((PowerlineSegment(f" /agent {identity} ", style.foreground, style.background),)).ansi()
