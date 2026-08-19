"""Tests for axio.diff - the shared diff renderer used by file-mutating tools."""

from __future__ import annotations

from axio.diff import render_diff


def test_single_line_replacement() -> None:
    before = "line1\nline2\nline3\n"
    after = "line1\nREPLACED\nline3\n"
    diff = render_diff("f.txt", before, after)
    assert "a/f.txt" in diff
    assert "b/f.txt" in diff
    assert "-line2" in diff
    assert "+REPLACED" in diff


def test_insert_shows_added_lines() -> None:
    before = "a\nb\n"
    after = "a\nnew1\nnew2\nb\n"
    diff = render_diff("f.txt", before, after)
    assert "+new1" in diff
    assert "+new2" in diff


def test_missing_trailing_newline_is_readable() -> None:
    """A file with no final newline must not smear - and + onto one line."""
    before = "old content"
    after = "new content"
    diff = render_diff("f.txt", before, after)
    assert "-old content\n" in diff
    assert "+new content\n" in diff


def test_no_changes_is_not_a_dump() -> None:
    diff = render_diff("f.txt", "same\n", "same\n")
    assert "f.txt" in diff


def test_absolute_path_header_has_no_double_slash() -> None:
    diff = render_diff("/workspace/f.txt", "a\nb\n", "a\nc\n")
    assert "a/workspace/f.txt" in diff
    assert "a//workspace" not in diff


def test_huge_edit_is_truncated() -> None:
    before = "line0\n"
    after = before + "".join(f"new{i}\n" for i in range(5000))
    diff = render_diff("big.txt", before, after)
    assert diff.count("\n") <= 500
    assert "diff truncated" in diff


def test_multiple_hunks_both_present() -> None:
    before = "".join(f"line{i}\n" for i in range(50))
    after = before.replace("line3\n", "line3 X\n").replace("line40\n", "line40 Y\n")
    diff = render_diff("f.txt", before, after)
    assert "-line3" in diff
    assert "+line3 X" in diff
    assert "-line40" in diff
    assert "+line40 Y" in diff
