"""Tests for axio.diff - the result messages shared by file-mutating tools."""

from __future__ import annotations

from axio.diff import CONTEXT_LINES, MAX_DIFF_LINES, describe_write, render_diff


def _numbered(count: int) -> str:
    return "".join(f"line{i}\n" for i in range(count))


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
