from __future__ import annotations

import pytest
from axio.diff import describe_patch

from axio_repl._theme import DEFAULT_THEME, NO_COLOR_THEME
from axio_repl._tool_result_display import is_write_file_result, parse_patch_result

_COMPACT = "+1 -1\n@@ -1,2 +1,2 @@ Service.run\n class Service:\n-    return 1\n+    return 2\n"


def test_compact_patch_result_has_exact_plain_and_owned_ansi_lines() -> None:
    display = parse_patch_result(_COMPACT)

    assert display is not None
    assert display.plain_text() == _COMPACT.rstrip("\n")
    assert display.styled_text(DEFAULT_THEME) == (
        f"{DEFAULT_THEME.stdout.ansi}+1 -1{DEFAULT_THEME.reset}\n"
        f"{DEFAULT_THEME.tool.ansi}@@ -1,2 +1,2 @@ Service.run{DEFAULT_THEME.reset}\n"
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


def test_parser_is_total_for_huge_counts_and_invalid_unicode() -> None:
    huge = "9" * 5000

    assert parse_patch_result(f"+{huge} -0\n@@ -0,0 +1 @@\n+new") is None
    assert parse_patch_result(f"+1 -0\n@@ -0,0 +1,{huge} @@\n+new") is None
    assert parse_patch_result("+1 -0\n@@ -0,0 +1 @@\n+bad\ud800") is None


def test_parser_validates_each_complete_hunk_count() -> None:
    valid = "+2 -2\n@@ -1,2 +1,2 @@ one\n same\n-old\n+new\n@@ -10 +10 @@ two\n-before\n+after\n"
    bad_first = valid.replace("@@ -1,2 +1,2 @@", "@@ -1,3 +1,2 @@")
    bad_second = valid.replace("@@ -10 +10 @@", "@@ -10,2 +10 @@")

    assert parse_patch_result(valid) is not None
    assert parse_patch_result(bad_first) is None
    assert parse_patch_result(bad_second) is None
    assert parse_patch_result(valid.replace("@@ -1,2 +1,2 @@", "@@ -0,2 +1,2 @@")) is None


def test_parser_accepts_only_conventional_no_newline_metadata() -> None:
    valid = "+1 -1\n@@ -1 +1 @@\n-old\n\\ No newline at end of file\n+new\n"

    assert parse_patch_result(valid) is not None
    assert parse_patch_result(valid.replace("No newline at end of file", "unexpected metadata")) is None
    assert parse_patch_result(valid.replace("-old\n\\", "\\\n-old\n\\")) is None


def test_only_truncated_final_hunk_is_exempt_from_exact_counts() -> None:
    valid = "+20 -20\n@@ -1 +1 @@ one\n-old\n+new\n@@ -20,19 +20,19 @@ two\n-partial\n...[diff truncated]\n"
    bad_complete_hunk = valid.replace("@@ -1 +1 @@", "@@ -1,2 +1 @@")

    assert parse_patch_result(valid) is not None
    assert parse_patch_result(bad_complete_hunk) is None
    assert parse_patch_result(valid.replace("...[diff truncated]\n", "")) is None
    assert parse_patch_result(valid + "+after-marker\n") is None


@pytest.mark.parametrize(
    "content",
    [
        "+2 -0\n@@ -0,0 +1 @@\n+one\n+two\n...[diff truncated]\n",
        "+0 -2\n@@ -1 +0,0 @@\n-one\n-two\n...[diff truncated]\n",
        "+1 -0\n@@ -1 +1 @@\n context\n+extra\n...[diff truncated]\n",
        "+1 -1\n@@ -1 +1 @@\n-old\n+new\n context\n...[diff truncated]\n",
        ("+3 -2\n@@ -1 +1 @@ one\n-old\n+new\n@@ -10 +10 @@ two\n-before\n+after\n+overflow\n...[diff truncated]\n"),
    ],
)
def test_truncation_never_exempts_old_or_new_overcount(content: str) -> None:
    assert parse_patch_result(content) is None


def test_truncated_final_hunk_may_be_under_count_but_not_over_count() -> None:
    content = "+20 -20\n@@ -1 +1 @@ one\n-old\n+new\n@@ -20,19 +20,19 @@ two\n-partial\n...[diff truncated]\n"

    assert parse_patch_result(content) is not None


def test_parser_accepts_the_actual_bounded_patch_result() -> None:
    before = "line0\n"
    after = before + "".join(f"new{index}\n" for index in range(5000))
    result = describe_patch("large.txt", before, after)

    display = parse_patch_result(result)

    assert display is not None
    assert display.plain_text().endswith("...[diff truncated]")


def test_write_result_recognizer_is_tool_handler_specific() -> None:
    assert is_write_file_result("Wrote 5 bytes to app.py")
    assert is_write_file_result(
        "Wrote 4 bytes to app.py\nChanged app.py:\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"
    )
    assert not is_write_file_result("wrote 5 bytes to app.py")
    assert not is_write_file_result("Wrote several bytes to app.py")
    assert not is_write_file_result("prefix Wrote 5 bytes to app.py")
    assert not is_write_file_result("Wrote 5 bytes to app.py\nunrelated")
