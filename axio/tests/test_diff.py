"""Tests for axio.diff - the result messages shared by file-mutating tools."""

from __future__ import annotations

import pytest

from axio.diff import CONTEXT_LINES, MAX_DIFF_LINES, decode_patch_content, describe_patch, describe_write, render_diff


def _numbered(count: int) -> str:
    return "".join(f"line{i}\n" for i in range(count))


@pytest.mark.parametrize(
    ("framed", "decoded"),
    [
        ("│zero", "zero"),
        ("│    spaces\n│\ttext", "    spaces\n\ttext"),
        ("│empty follows\n│\n│last", "empty follows\n\nlast"),
        ("│with newline\n", "with newline\n"),
        ("│", "\n"),
        ("││literal sentinel", "│literal sentinel"),
    ],
)
def test_decode_patch_content_strips_one_sentinel_from_every_framed_line(framed: str, decoded: str) -> None:
    assert "".join(decode_patch_content(framed)) == decoded
    if framed == "│":
        assert decode_patch_content(framed) == ["\n"]


@pytest.mark.parametrize("legacy", ["", "zero", "    spaces\n\ttext", "zero\n"])
def test_decode_patch_content_keeps_entirely_unframed_legacy_content_literal(legacy: str) -> None:
    assert "".join(decode_patch_content(legacy)) == legacy


@pytest.mark.parametrize("mixed", ["│framed\nlegacy", "legacy\n│framed", "│framed\n\n│framed"])
def test_decode_patch_content_rejects_mixed_framing(mixed: str) -> None:
    with pytest.raises(ValueError, match="every content line"):
        decode_patch_content(mixed)


@pytest.mark.parametrize("numbered", ["L1│source", "L0042│    source\n"])
def test_decode_patch_content_rejects_read_file_line_number_metadata(numbered: str) -> None:
    with pytest.raises(ValueError, match="remove the L<number> prefix"):
        decode_patch_content(numbered)


def test_change_is_shown_with_surrounding_context() -> None:
    """The point of the diff is anchoring: an edit must arrive with the
    unchanged lines around it, and the hunk header must agree with them."""
    before = _numbered(20)
    after = before.replace("line10\n", "line10 CHANGED\n")

    diff = render_diff("/workspace/f.txt", before, after)

    assert "--- a/workspace/f.txt" in diff
    assert "+++ b/workspace/f.txt" in diff
    assert "-line10\n" in diff
    assert "+line10 CHANGED\n" in diff
    context = [line[1:] for line in diff.splitlines() if line.startswith(" ")]
    assert context == [f"line{i}" for i in (7, 8, 9, 11, 12, 13)]
    # 6 context lines plus the replaced one, at the same offset on both sides.
    assert f"@@ -8,{2 * CONTEXT_LINES + 1} +8,{2 * CONTEXT_LINES + 1} @@" in diff


def test_distant_changes_become_separate_hunks() -> None:
    """Untouched territory between two edits is collapsed, not reproduced."""
    before = _numbered(60)
    after = before.replace("line5\n", "line5 A\n").replace("line50\n", "line50 B\n")

    diff = render_diff("f.txt", before, after)

    assert diff.count("@@ -") == 2
    assert "+line5 A\n" in diff
    assert "+line50 B\n" in diff
    assert "line30" not in diff


def test_identical_snapshots_render_nothing() -> None:
    """An empty diff is the signal callers use to omit it from the message."""
    assert render_diff("f.txt", "same\n", "same\n") == ""


def test_missing_trailing_newline_keeps_lines_apart() -> None:
    """Without a forced terminator the last - and + lines run together."""
    diff = render_diff("f.txt", "old content", "new content")

    assert "-old content\n" in diff
    assert "+new content\n" in diff


def test_compact_patch_distinguishes_adding_and_removing_final_newline() -> None:
    added = describe_patch("f.txt", "same", "same\n")
    removed = describe_patch("f.txt", "same\n", "same")

    assert added == ("+1 -1\n@@ -1 +1 @@\n-same\n\\ No newline at end of file\n+same\n")
    assert removed == ("+1 -1\n@@ -1 +1 @@\n-same\n+same\n\\ No newline at end of file\n")


def test_compact_patch_marks_missing_newline_on_last_line_of_multiline_file() -> None:
    result = describe_patch("f.txt", "one\ntwo", "one\ntwo\n")

    assert result == ("+1 -1\n@@ -1,2 +1,2 @@\n one\n-two\n\\ No newline at end of file\n+two\n")


def test_compact_patch_handles_empty_and_single_line_newline_transitions() -> None:
    assert describe_patch("f.txt", "", "\n") == "+1 -0\n@@ -0,0 +1 @@\n+\n"
    assert describe_patch("f.txt", "\n", "") == "+0 -1\n@@ -1 +0,0 @@\n-\n"


def test_oversized_diff_is_truncated() -> None:
    """A large rewrite must stay a summary instead of flooding the context."""
    before = "line0\n"
    after = before + "".join(f"new{i}\n" for i in range(5000))

    diff = render_diff("big.txt", before, after)

    assert "...[diff truncated]" in diff
    assert len(diff.splitlines()) <= MAX_DIFF_LINES + 2


def test_describe_write_appends_a_diff_only_when_there_is_one() -> None:
    """No previous content and an unchanged rewrite both report size alone."""
    assert describe_write("f.txt", 5, None, "hello") == "Wrote 5 bytes to f.txt"
    assert describe_write("f.txt", 5, "hello", "hello") == "Wrote 5 bytes to f.txt"

    message = describe_write("f.txt", 5, "there", "hello")
    assert message.startswith("Wrote 5 bytes to f.txt\nChanged f.txt:\n")
    assert "+hello\n" in message


def test_describe_patch_is_a_path_free_compact_diff_fragment() -> None:
    result = describe_patch("src/app.py", "def run():\n    return 1\n", "def run():\n    return 2\n")

    assert result == ("+1 -1\n@@ -1,2 +1,2 @@ run\n def run():\n-    return 1\n+    return 2\n")
    assert "src/app.py" not in result
    assert "---" not in result
    assert "+++" not in result


def test_describe_patch_reports_an_exact_noop_without_a_path() -> None:
    assert describe_patch("secret/name.py", "same\n", "same\n") == "No changes"


def test_compact_patch_diff_is_bounded_after_adding_exact_stats() -> None:
    before = "def build():\n    return []\n"
    after = before + "".join(f"value_{index} = {index}\n" for index in range(5000))

    result = describe_patch("large.py", before, after)

    assert result.startswith("+5000 -0\n")
    assert "...[diff truncated]" in result
    assert len(result.splitlines()) <= MAX_DIFF_LINES + 2
