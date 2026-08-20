from __future__ import annotations

import pytest

from axio_repl._theme import (
    DEFAULT_THEME,
    MONOCHROME_THEME,
    NO_COLOR_THEME,
    resolve_terminal_presentation,
    resolve_theme,
    theme_names,
)


def test_builtin_theme_registry_is_stable_and_exact() -> None:
    assert theme_names() == ("default", "monochrome")
    assert resolve_theme("default") is DEFAULT_THEME
    assert resolve_theme("monochrome") is MONOCHROME_THEME


def test_unknown_theme_fails_with_available_names() -> None:
    with pytest.raises(ValueError, match="unknown theme 'solarized'.*default, monochrome"):
        resolve_theme("solarized")


def test_default_and_monochrome_are_semantically_distinct() -> None:
    assert DEFAULT_THEME.prompt.ansi == "\033[1;30;107m"
    assert DEFAULT_THEME.prompt.prompt_toolkit == "bold fg:ansiblack bg:ansiwhite"
    assert DEFAULT_THEME.stderr.ansi == "\033[2;33m"
    assert DEFAULT_THEME.error.ansi == "\033[31m"
    assert DEFAULT_THEME.prompt_badge.foreground.prompt_toolkit == "ansiblack"
    assert DEFAULT_THEME.prompt_badge.background.prompt_toolkit == "ansiwhite"
    assert DEFAULT_THEME.tool_badge.foreground.prompt_toolkit == "ansiblack"
    assert DEFAULT_THEME.tool_badge.background.prompt_toolkit == "ansicyan"
    assert DEFAULT_THEME.agent_badge.background.prompt_toolkit == "ansimagenta"
    assert DEFAULT_THEME.action_badge.background.prompt_toolkit == "ansiyellow"
    assert MONOCHROME_THEME.prompt.ansi == "\033[1;97m"
    assert MONOCHROME_THEME.prompt.prompt_toolkit == "bold ansiwhite"
    assert MONOCHROME_THEME.prompt_badge.foreground.prompt_toolkit == "ansiblack"
    assert MONOCHROME_THEME.stderr.ansi == "\033[1;97m"
    assert MONOCHROME_THEME.error.ansi == "\033[1;7m"
    assert MONOCHROME_THEME.prompt_badge.background.prompt_toolkit == "ansiwhite"


@pytest.mark.parametrize("one_shot", [False, True])
def test_no_color_overrides_explicit_theme_and_powerline(one_shot: bool) -> None:
    theme, powerline = resolve_terminal_presentation(
        MONOCHROME_THEME,
        powerline=True,
        one_shot=one_shot,
        stdout_is_tty=True,
        no_color=True,
    )

    assert theme is NO_COLOR_THEME
    assert theme.reset == ""
    assert all(
        style.ansi == ""
        for style in (
            theme.prompt,
            theme.emphasis,
            theme.command,
            theme.tool,
            theme.agent,
            theme.action,
            theme.reasoning,
            theme.stdout,
            theme.stderr,
            theme.error,
            theme.success,
            theme.warning,
        )
    )
    assert powerline is False


def test_only_non_tty_one_shot_implicitly_disables_colour() -> None:
    one_shot_theme, one_shot_powerline = resolve_terminal_presentation(
        DEFAULT_THEME,
        powerline=True,
        one_shot=True,
        stdout_is_tty=False,
        no_color=False,
    )
    interactive_theme, interactive_powerline = resolve_terminal_presentation(
        DEFAULT_THEME,
        powerline=True,
        one_shot=False,
        stdout_is_tty=False,
        no_color=False,
    )

    assert (one_shot_theme, one_shot_powerline) == (NO_COLOR_THEME, False)
    assert (interactive_theme, interactive_powerline) == (DEFAULT_THEME, True)
