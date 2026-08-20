"""Powerline badges shared by prompt-toolkit and streamed terminal output."""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.formatted_text import FormattedText

RESET = "\033[0m"
POWERLINE_RIGHT = "\ue0b0"
POWERLINE_LEFT = "\ue0b2"


@dataclass(frozen=True, slots=True)
class PowerlineColor:
    """One colour expressed for ANSI SGR and prompt-toolkit."""

    ansi_foreground: int
    ansi_background: int
    prompt_toolkit: str


@dataclass(frozen=True, slots=True)
class PowerlineSegment:
    """Text and colours for one filled segment."""

    text: str
    foreground: PowerlineColor
    background: PowerlineColor


@dataclass(frozen=True, slots=True)
class PowerlineBadge:
    """A filled badge with hard separators at both outer edges."""

    segments: tuple[PowerlineSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a Powerline badge requires at least one segment")

    def ansi(self) -> str:
        """Render a reset-safe ANSI badge on the terminal's default background."""

        first = self.segments[0]
        parts = [f"\033[22;{first.background.ansi_foreground};49m{POWERLINE_LEFT}"]
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

        first = self.segments[0]
        parts: list[tuple[str, str]] = [(f"fg:{first.background.prompt_toolkit} bg:default", POWERLINE_LEFT)]
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


_DARK = PowerlineColor(ansi_foreground=30, ansi_background=40, prompt_toolkit="ansiblack")
_WHITE = PowerlineColor(ansi_foreground=97, ansi_background=107, prompt_toolkit="ansiwhite")
_CYAN = PowerlineColor(ansi_foreground=36, ansi_background=46, prompt_toolkit="ansicyan")
_MAGENTA = PowerlineColor(ansi_foreground=35, ansi_background=45, prompt_toolkit="ansimagenta")
_YELLOW = PowerlineColor(ansi_foreground=33, ansi_background=43, prompt_toolkit="ansiyellow")


def prompt_badge() -> FormattedText:
    """Return the complete input badge followed by one editor-space."""

    badge = PowerlineBadge((PowerlineSegment(" axio-repl ", _DARK, _CYAN),))
    return badge.formatted_text(trailing=" ")


def tool_title(name: str) -> str:
    """Return a reset-safe Powerline badge for a live tool call."""

    return PowerlineBadge((PowerlineSegment(f" ▶ {name} ", _DARK, _CYAN),)).ansi()


def agent_header(identity: str) -> str:
    """Return a reset-safe Powerline badge for a live agent turn."""

    return PowerlineBadge((PowerlineSegment(f" agent {identity} ", _WHITE, _MAGENTA),)).ansi()


def action_frame_header(identity: str, kind: str) -> str:
    """Return the reset-safe two-segment badge of a background action."""

    return PowerlineBadge(
        (
            PowerlineSegment(f" agent {identity} ", _WHITE, _MAGENTA),
            PowerlineSegment(f" {kind} ", _DARK, _YELLOW),
        )
    ).ansi()


def action_frame_footer(identity: str) -> str:
    """Return the reset-safe closing badge of a background action."""

    return PowerlineBadge((PowerlineSegment(f" /agent {identity} ", _WHITE, _MAGENTA),)).ansi()
