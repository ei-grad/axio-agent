from __future__ import annotations

from axio_repl._theme import DEFAULT_THEME, NO_COLOR_THEME
from axio_repl._tool_result_display import is_write_file_result, parse_patch_result

_COMPACT = "+1 -1\n@@ -1,3 +1,3 @@ Service.run\n class Service:\n-    return 1\n+    return 2\n"


def test_compact_patch_result_has_exact_plain_and_owned_ansi_lines() -> None:
    display = parse_patch_result(_COMPACT)

    assert display is not None
    assert display.plain_text() == _COMPACT.rstrip("\n")
    assert display.styled_text(DEFAULT_THEME) == (
        f"{DEFAULT_THEME.stdout.ansi}+1 -1{DEFAULT_THEME.reset}\n"
        f"{DEFAULT_THEME.tool.ansi}@@ -1,3 +1,3 @@ Service.run{DEFAULT_THEME.reset}\n"
        f"{DEFAULT_THEME.reasoning.ansi} class Service:{DEFAULT_THEME.reset}\n"
        f"{DEFAULT_THEME.error.ansi}-    return 1{DEFAULT_THEME.reset}\n"
        f"{DEFAULT_THEME.success.ansi}+    return 2{DEFAULT_THEME.reset}"
    )
    assert DEFAULT_THEME.error.ansi not in display.styled_text(DEFAULT_THEME).splitlines()[0]
    assert display.styled_text(NO_COLOR_THEME) == display.plain_text()
    assert "\x1b[" not in display.styled_text(NO_COLOR_THEME)


def test_legacy_patch_result_is_compacted_and_has_at_most_one_fallback_path() -> None:
    legacy = (
        "Wrote 20 bytes to src/app.py\n"
        "Changed src/app.py:\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    normal = parse_patch_result(legacy)
    orphan = parse_patch_result(legacy, include_legacy_path=True)

    assert normal is not None and orphan is not None
    assert normal.plain_text() == "+1 -1\n@@ -1 +1 @@\n-old\n+new"
    assert "src/app.py" not in normal.plain_text()
    assert orphan.plain_text().count("src/app.py") == 1
    assert "---" not in orphan.plain_text()
    assert "+++" not in orphan.plain_text()


def test_parser_fails_open_on_malformed_unbounded_or_inconsistent_text() -> None:
    assert parse_patch_result("+1 -0\nnot a hunk\n+value") is None
    assert parse_patch_result("+9 -1\n@@ -1 +1 @@\n-old\n+new") is None
    assert parse_patch_result("+1 -0\n@@ -1 +1 @@ injected\x1b[2J\n+new") is None
    assert parse_patch_result("+1 -0\n@@ -1 +1 @@\n+" + "x" * (33 * 1024)) is None


def test_write_result_recognizer_is_tool_handler_specific() -> None:
    assert is_write_file_result("Wrote 5 bytes to app.py")
    assert is_write_file_result(
        "Wrote 4 bytes to app.py\nChanged app.py:\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"
    )
    assert not is_write_file_result("wrote 5 bytes to app.py")
    assert not is_write_file_result("Wrote several bytes to app.py")
    assert not is_write_file_result("prefix Wrote 5 bytes to app.py")
    assert not is_write_file_result("Wrote 5 bytes to app.py\nunrelated")
