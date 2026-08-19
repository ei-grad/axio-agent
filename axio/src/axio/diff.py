"""Result messages for file-mutating tools.

``patch_file`` and overwriting ``write_file`` both change a file the agent
already saw. Reporting only a byte count hides what actually changed and forces
the model to re-read the whole file to stay oriented, so both report a unified
diff of the edit instead. The format is the one ``git`` and ``diff -u`` produce,
so a model that already parses ``git diff`` output needs no new vocabulary.

Only UTF-8 text can be diffed. Callers signal "nothing to diff against" - a new
file, binary content, or a file too large to be worth re-reading - by passing
``before=None`` rather than by asking this module to guess an encoding.
"""

from __future__ import annotations

from difflib import unified_diff

CONTEXT_LINES = 3
MAX_DIFF_LINES = 400
MAX_DIFF_CHARS = 8192
MAX_DIFF_SOURCE_BYTES = 1 << 20

_TRUNCATION_MARKER = "...[diff truncated]\n"


def describe_write(path: str, written: int, before: str | None, after: str) -> str:
    """Compose the result message shared by every file-writing tool.

    ``before`` is the file's previous content, or None when there is nothing to
    diff against. Keeping the composition here is what keeps the local and
    docker implementations of the same tool from drifting into different shapes.
    """
    message = f"Wrote {written} bytes to {path}"
    if before is None:
        return message
    diff = render_diff(path, before, after)
    return f"{message}\n{diff}" if diff else message


def render_diff(path: str, before: str, after: str) -> str:
    """Render a bounded unified diff between two snapshots of ``path``.

    Returns an empty string when the snapshots are identical. ``difflib``
    collapses unchanged territory to ``CONTEXT_LINES`` around each hunk; a
    rewrite large enough to flood the model's context is cut off with an
    explicit marker so the tool result stays a summary rather than a dump.
    """
    label = path.lstrip("/")
    lines = list(
        unified_diff(
            _terminated(before),
            _terminated(after),
            f"a/{label}",
            f"b/{label}",
            n=CONTEXT_LINES,
        )
    )
    if not lines:
        return ""
    return f"Changed {path}:\n{_bounded(''.join(lines))}"


def _terminated(text: str) -> list[str]:
    r"""Split into lines, forcing a final newline for rendering only.

    ``diff -u`` marks a missing final newline with ``\ No newline at end of
    file``. The tools already report exact byte counts, so that note is noise -
    and without the newline the last ``-`` and ``+`` lines run together on one
    line. The real file content is never touched.
    """
    if text and not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def _bounded(diff_text: str) -> str:
    """Truncate an oversized diff, in lines first and then in characters."""
    lines = diff_text.splitlines(keepends=True)
    if len(lines) > MAX_DIFF_LINES:
        diff_text = "".join(lines[:MAX_DIFF_LINES]) + _TRUNCATION_MARKER
    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = diff_text[:MAX_DIFF_CHARS] + "\n" + _TRUNCATION_MARKER
    return diff_text
