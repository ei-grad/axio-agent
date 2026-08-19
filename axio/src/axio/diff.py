"""Shared diff rendering for file-mutating tools.

``patch_file`` and overwriting ``write_file`` both change a file the agent
already saw. Returning only a byte/line count hides what actually changed and
forces the model to re-read the whole file to stay oriented. Every mutating
tool therefore renders a unified diff of the old content against the new one,
bounded so a large rewrite does not flood the context: unchanged territory is
collapsed to a few context lines around each hunk. The format is deliberately
the one ``git`` and ``diff -u`` produce, so models that already parse that
shape in ``git diff`` output need no new vocabulary.
"""

from __future__ import annotations

from difflib import unified_diff

_MAX_DIFF_CHARS = 8192
_MAX_DIFF_LINES = 400


def _normalize_line_terminal(s: str) -> str:
    """Force a trailing newline for diff rendering only.

    diff -u marks a missing final newline with ``\\ No newline at end of file``,
    which is accurate but noisy for tools that already report byte counts; giving
    every snapshot a terminal newline keeps the hunk readable without ever touching
    the real file contents.
    """
    return s if s.endswith("\n") else s + "\n"


def render_diff(path: str, before: str, after: str, *, diff_bytes: int = 0) -> str:
    """Render a bounded unified diff between two file snapshots.

    Line lengths stay reasonable even for generated files: hunks with no changes
    are dropped, and the total is truncated with an explicit marker when the edit
    exceeds the budget (in lines first, then in characters).
    """
    before_labels = (f"a/{path.lstrip('/')}", f"b/{path.lstrip('/')}")
    generated = unified_diff(
        _normalize_line_terminal(before).splitlines(keepends=True),
        _normalize_line_terminal(after).splitlines(keepends=True),
        *before_labels,
    )
    # Drop pure un-changed-only hunks (difflib still emits @@-less context).
    kept_lines = [line for line in generated if not (line.startswith(" ") and not _is_hunk_header(line))]

    if kept_lines:
        diff_text = "".join(_collapse_context(kept_lines))
        if diff_text.count("\n") > _MAX_DIFF_LINES:
            diff_text = "\n".join(diff_text.splitlines()[:_MAX_DIFF_LINES]) + "\n...[diff truncated]\n"
        if len(diff_text) > _MAX_DIFF_CHARS:
            diff_text = diff_text[:_MAX_DIFF_CHARS] + "\n...[diff truncated]\n"
        size_note = f" ({diff_bytes} bytes)" if diff_bytes else ""
        return f"Changed {path}{size_note}:\n{diff_text}"
    return f"Changed {path}{size_note} ({diff_bytes} bytes)" if diff_bytes else f"Changed {path}"


def _is_hunk_header(line: str) -> bool:
    return line.startswith("@@") or line.startswith(("---", "+++"))


def _collapse_context(lines: list[str]) -> list[str]:
    """Drop most unchanged context lines between hunks, keeping the closest four
    around each change so the model can anchor the edit."""
    result: list[str] = []
    pending_space: list[str] = []
    for line in lines:
        if line.startswith(" ") or not line:
            pending_space.append(line)
            continue
        # A change line (-, +, header): flush the held context, trimmed.
        gap = len(pending_space)
        if gap > 4:
            result.append("\n")  # blank separator marking removed context
            result.extend(pending_space[:2])
            result.append("...\n")
            result.extend(pending_space[-2:])
        else:
            result.extend(pending_space)
        result.append(line)
        pending_space = []
    return result
