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
PATCH_FROM_LINE_DESCRIPTION = (
    "1-indexed start of the edit. For replacement, this is the first old physical line removed, not merely the first "
    "line whose logic changes. For insertion (to_line = from_line - 1), new content is inserted before this line; "
    "line_count + 1 appends."
)
PATCH_TO_LINE_DESCRIPTION = (
    "For replacement, the 1-indexed inclusive last old physical line removed. For insertion, set to from_line - 1, "
    "which makes the selected old range empty."
)
PATCH_CONTENT_DESCRIPTION = (
    "Complete text inserted in place of the selected old range. For replacement, it may contain a different number "
    "of lines. Do not include unchanged source immediately before or after the range as context. If replacement "
    "content retains an old source line, include that line in the selected range. For insertion, the selected range "
    "is empty and content may intentionally duplicate adjacent source. Content is applied literally: preserve exact "
    "leading and trailing whitespace, tabs, empty lines, and newlines. Do not copy the L<number>│ metadata prefix "
    "from numbered read_file output."
)

_TRUNCATION_MARKER = "...[diff truncated]\n"
_NO_NEWLINE_MARKER = "\\ No newline at end of file\n"
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
    before_lines = _source_lines(before)
    after_lines = _source_lines(after)
    lines = _unified_diff(before_lines, after_lines)
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
    lines = _unified_diff(
        _source_lines(before),
        _source_lines(after),
        fromfile=f"a/{label}",
        tofile=f"b/{label}",
    )
    if not lines:
        return ""
    return f"Changed {path}:\n{_bounded(''.join(lines))}"


def _source_lines(text: str) -> list[str]:
    """Split source without erasing whether its final line has a newline."""
    return text.splitlines(keepends=True)


def _unified_diff(
    before: list[str],
    after: list[str],
    *,
    fromfile: str = "",
    tofile: str = "",
) -> list[str]:
    """Render records with conventional missing-final-newline metadata."""
    rendered: list[str] = []
    for line in unified_diff(before, after, fromfile, tofile, n=CONTEXT_LINES):
        if line.endswith("\n"):
            rendered.append(line)
            continue
        rendered.append(f"{line}\n")
        if line.startswith((" ", "+", "-")):
            rendered.append(_NO_NEWLINE_MARKER)
    return rendered


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
            if line[1:].strip():
                new_symbols.append(_at(new_contexts, new_line))
            new_line += 1
        elif line.startswith("-"):
            if line[1:].strip():
                old_symbols.append(_at(old_contexts, old_line))
            old_line += 1
        elif line.startswith(" "):
            old_line += 1
            new_line += 1
    new_context = _consistent_context(new_symbols)
    old_context = _consistent_context(old_symbols)
    if new_context is _AMBIGUOUS or old_context is _AMBIGUOUS:
        return _AMBIGUOUS
    if new_symbols and old_symbols and (new_context is None) != (old_context is None):
        return _AMBIGUOUS
    return new_context if new_symbols else old_context


def _at(contexts: tuple[str | None, ...], line_number: int) -> str | None:
    if 0 < line_number < len(contexts):
        return contexts[line_number]
    return None


def _consistent_context(symbols: list[str | None]) -> str | _AmbiguousContext | None:
    distinct = set(symbols)
    if len(distinct) > 1:
        return _AMBIGUOUS
    return next(iter(distinct), None)
