"""Result messages for file-mutating tools.

``patch_file`` and overwriting ``write_file`` both change a file the agent
already saw. Reporting only a byte count hides what actually changed and forces
the model to re-read the whole file to stay oriented. ``write_file`` reports a
bounded unified diff; ``patch_file`` reports a path-free fragment with exact
stats, hunks, and best-effort function context because its tool input already
identifies the one edited path.

Only UTF-8 text can be diffed. Callers signal "nothing to diff against" - a new
file, binary content, or a file too large to be worth re-reading - by passing
``before=None`` rather than by asking this module to guess an encoding.
"""

from __future__ import annotations

import re
from difflib import unified_diff

from axio.symbol_context import line_contexts, sanitize_symbol

CONTEXT_LINES = 3
MAX_DIFF_LINES = 400
MAX_DIFF_CHARS = 8192
MAX_DIFF_SOURCE_BYTES = 1 << 20

_TRUNCATION_MARKER = "...[diff truncated]\n"
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@$")


class _AmbiguousContext:
    pass


_AMBIGUOUS = _AmbiguousContext()


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


def describe_patch(path: str, before: str, after: str) -> str:
    """Render the compact, path-free result returned by ``patch_file``."""
    before_lines = _terminated(before)
    after_lines = _terminated(after)
    lines = list(unified_diff(before_lines, after_lines, n=CONTEXT_LINES))
    if not lines:
        return "No changes"
    body = lines[2:]
    additions = sum(line.startswith("+") for line in body)
    deletions = sum(line.startswith("-") for line in body)
    contextual = _with_hunk_context(path, before_lines, after_lines, body)
    return _bounded(f"+{additions} -{deletions}\n{''.join(contextual)}")


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


def _with_hunk_context(
    path: str,
    before: list[str],
    after: list[str],
    diff_lines: list[str],
) -> list[str]:
    old_contexts = line_contexts(path, before)
    new_contexts = line_contexts(path, after)
    rendered: list[str] = []
    index = 0
    while index < len(diff_lines):
        header = diff_lines[index]
        match = _HUNK_HEADER.fullmatch(header.rstrip("\n"))
        if match is None:
            rendered.append(header)
            index += 1
            continue
        end = index + 1
        while end < len(diff_lines) and not diff_lines[end].startswith("@@ "):
            end += 1
        symbol = _hunk_symbol(match, diff_lines[index + 1 : end], old_contexts, new_contexts)
        suffix = sanitize_symbol(symbol) if isinstance(symbol, str) else None
        rendered.append(f"{header.rstrip()} {suffix}\n" if suffix else header)
        rendered.extend(diff_lines[index + 1 : end])
        index = end
    return rendered


def _hunk_symbol(
    header: re.Match[str],
    lines: list[str],
    old_contexts: tuple[str | None, ...],
    new_contexts: tuple[str | None, ...],
) -> str | _AmbiguousContext | None:
    old_line = int(header.group(1))
    new_line = int(header.group(3))
    old_symbols: list[str | None] = []
    new_symbols: list[str | None] = []
    for line in lines:
        if line.startswith("+"):
            new_symbols.append(_at(new_contexts, new_line))
            new_line += 1
        elif line.startswith("-"):
            old_symbols.append(_at(old_contexts, old_line))
            old_line += 1
        elif line.startswith(" "):
            old_line += 1
            new_line += 1
    preferred = _consistent_context(new_symbols) if new_symbols else _consistent_context(old_symbols)
    if preferred is _AMBIGUOUS or preferred is not None:
        return preferred
    return _consistent_context(old_symbols)


def _at(contexts: tuple[str | None, ...], line_number: int) -> str | None:
    if 0 < line_number < len(contexts):
        return contexts[line_number]
    return None


def _consistent_context(symbols: list[str | None]) -> str | _AmbiguousContext | None:
    distinct = {symbol for symbol in symbols if symbol is not None}
    if len(distinct) > 1:
        return _AMBIGUOUS
    return next(iter(distinct), None)
