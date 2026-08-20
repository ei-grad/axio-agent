from __future__ import annotations

import pytest

from axio_repl._theme import DEFAULT_THEME, MONOCHROME_THEME, resolve_theme, theme_names


def test_builtin_theme_registry_is_stable_and_exact() -> None:
    assert theme_names() == ("default", "monochrome")
    assert resolve_theme("default") is DEFAULT_THEME
    assert resolve_theme("monochrome") is MONOCHROME_THEME


def test_unknown_theme_fails_with_available_names() -> None:
    with pytest.raises(ValueError, match="unknown theme 'solarized'.*default, monochrome"):
        resolve_theme("solarized")


def test_default_and_monochrome_are_semantically_distinct() -> None:
    assert DEFAULT_THEME.prompt.ansi == "\033[1;97m"
    assert DEFAULT_THEME.stderr.ansi == "\033[2;33m"
    assert DEFAULT_THEME.error.ansi == "\033[31m"
    assert MONOCHROME_THEME.stderr.ansi == "\033[1;97m"
    assert MONOCHROME_THEME.error.ansi == "\033[1;7m"
    assert MONOCHROME_THEME.prompt_badge.background.prompt_toolkit == "ansiwhite"
