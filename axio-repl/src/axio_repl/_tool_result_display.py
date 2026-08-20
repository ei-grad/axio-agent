"""Bounded display-only parsing for file-tool results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from axio_repl._theme import TerminalTheme

_MAX_RESULT_BYTES = 32 * 1024
_MAX_HEADER_CHARS = 512
_COUNT = r"[0-9]{1,10}"
_SUMMARY = re.compile(rf"^\+({_COUNT}) -({_COUNT})$")
_HUNK = re.compile(
    rf"^@@ -(?P<old_start>{_COUNT})(?:,(?P<old_count>{_COUNT}))? "
    rf"\+(?P<new_start>{_COUNT})(?:,(?P<new_count>{_COUNT}))? @@"
    r"(?: [A-Za-z_$~][A-Za-z0-9_$~:.<>]*)?$"
)
_WRITE = re.compile(rf"^Wrote {_COUNT} bytes to (.+)$")
_TRUNCATED = "...[diff truncated]"
_NO_NEWLINE = "\\ No newline at end of file"


class DiffLineKind(StrEnum):
    SUMMARY = "summary"
    HUNK = "hunk"
    ADDITION = "addition"
    DELETION = "deletion"
    CONTEXT = "context"
    METADATA = "metadata"


@dataclass(frozen=True, slots=True)
class DiffDisplayLine:
    text: str
    kind: DiffLineKind


@dataclass(frozen=True, slots=True)
class PatchResultDisplay:
    lines: tuple[DiffDisplayLine, ...]
    fallback_path: str | None = None

    def plain_text(self) -> str:
        prefix = (f"path: {self.fallback_path}",) if self.fallback_path is not None else ()
        return "\n".join((*prefix, *(line.text for line in self.lines)))

    def line_kinds(self) -> tuple[DiffLineKind, ...]:
        prefix = (DiffLineKind.METADATA,) if self.fallback_path is not None else ()
        return (*prefix, *(line.kind for line in self.lines))

    def styled_text(self, theme: TerminalTheme) -> str:
        return style_classified_text(self.plain_text(), self.line_kinds(), theme)


@dataclass(slots=True)
class _HunkState:
    expected_old: int
    expected_new: int
    observed_old: int = 0
    observed_new: int = 0
    saw_change: bool = False
    metadata_allowed: bool = False

    def is_complete(self) -> bool:
        return self.saw_change and self.observed_old == self.expected_old and self.observed_new == self.expected_new


def parse_patch_result(content: str, *, include_legacy_path: bool = False) -> PatchResultDisplay | None:
    """Parse a compact patch diff, failing open with ``None`` on any mismatch."""
    try:
        return _parse_patch_result(content, include_legacy_path=include_legacy_path)
    except (UnicodeError, ValueError, OverflowError):
        return None


def _parse_patch_result(content: str, *, include_legacy_path: bool) -> PatchResultDisplay | None:
    if not content or "\r" in content or len(content.encode("utf-8")) > _MAX_RESULT_BYTES:
        return None
    raw_lines = content.splitlines()
    if not raw_lines:
        return None
    fallback_path: str | None = None
    legacy = _legacy_body(raw_lines)
    if legacy is not None:
        path, raw_lines = legacy
        fallback_path = path if include_legacy_path else None

    summary: tuple[int, int] | None = None
    summary_match = _SUMMARY.fullmatch(raw_lines[0])
    if summary_match is not None:
        summary = (int(summary_match.group(1)), int(summary_match.group(2)))
        raw_lines = raw_lines[1:]
    if not raw_lines:
        return None

    parsed: list[DiffDisplayLine] = []
    additions = 0
    deletions = 0
    hunk: _HunkState | None = None
    truncated = False
    for index, line in enumerate(raw_lines):
        if line == _TRUNCATED:
            if index != len(raw_lines) - 1 or hunk is None or summary is None:
                return None
            parsed.append(DiffDisplayLine(line, DiffLineKind.METADATA))
            truncated = True
            continue
        if line.startswith("@@"):
            match = _HUNK.fullmatch(line)
            if len(line) > _MAX_HEADER_CHARS or match is None:
                return None
            if hunk is not None and not hunk.is_complete():
                return None
            old_start = int(match.group("old_start"))
            new_start = int(match.group("new_start"))
            old_count = int(match.group("old_count") or "1")
            new_count = int(match.group("new_count") or "1")
            if (old_start == 0 and old_count > 0) or (new_start == 0 and new_count > 0):
                return None
            hunk = _HunkState(expected_old=old_count, expected_new=new_count)
            parsed.append(DiffDisplayLine(line, DiffLineKind.HUNK))
            continue
        if hunk is None:
            return None
        if line.startswith("+"):
            additions += 1
            hunk.observed_new += 1
            hunk.saw_change = True
            hunk.metadata_allowed = True
            kind = DiffLineKind.ADDITION
        elif line.startswith("-"):
            deletions += 1
            hunk.observed_old += 1
            hunk.saw_change = True
            hunk.metadata_allowed = True
            kind = DiffLineKind.DELETION
        elif line.startswith(" "):
            hunk.observed_old += 1
            hunk.observed_new += 1
            hunk.metadata_allowed = True
            kind = DiffLineKind.CONTEXT
        elif line == _NO_NEWLINE and hunk.metadata_allowed:
            hunk.metadata_allowed = False
            kind = DiffLineKind.METADATA
        else:
            return None
        parsed.append(DiffDisplayLine(line, kind))
    if hunk is None or (not truncated and not hunk.is_complete()):
        return None
    if summary is not None and not truncated and summary != (additions, deletions):
        return None
    if summary is None:
        summary = (additions, deletions)
    lines = (DiffDisplayLine(f"+{summary[0]} -{summary[1]}", DiffLineKind.SUMMARY), *parsed)
    return PatchResultDisplay(lines=lines, fallback_path=fallback_path)


def is_write_file_result(content: str) -> bool:
    """Recognize only the exact successful result shapes produced by write_file."""
    try:
        within_bound = len(content.encode("utf-8")) <= _MAX_RESULT_BYTES
    except UnicodeError:
        return False
    if not content or "\r" in content or not within_bound:
        return False
    lines = content.splitlines()
    if len(lines) == 1:
        return _WRITE.fullmatch(lines[0]) is not None
    return _legacy_body(lines) is not None and parse_patch_result(content) is not None


def style_classified_text(
    text: str,
    line_kinds: Sequence[DiffLineKind],
    theme: TerminalTheme,
) -> str:
    """Apply owned semantic styles to already-sanitized physical lines."""
    rendered: list[str] = []
    for index, line in enumerate(text.split("\n")):
        if index >= len(line_kinds):
            rendered.append(line)
            continue
        kind = line_kinds[index]
        style = {
            DiffLineKind.SUMMARY: theme.stdout.ansi,
            DiffLineKind.HUNK: theme.tool.ansi,
            DiffLineKind.ADDITION: theme.success.ansi,
            DiffLineKind.DELETION: theme.error.ansi,
            DiffLineKind.CONTEXT: theme.reasoning.ansi,
            DiffLineKind.METADATA: theme.stdout.ansi,
        }[kind]
        rendered.append(_styled_line(style, line, theme.reset))
    return "\n".join(rendered)


def _legacy_body(lines: list[str]) -> tuple[str, list[str]] | None:
    if len(lines) < 5:
        return None
    write_match = _WRITE.fullmatch(lines[0])
    if write_match is None:
        return None
    path = write_match.group(1)
    label = path.lstrip("/")
    if not path or lines[1] != f"Changed {path}:" or lines[2] != f"--- a/{label}" or lines[3] != f"+++ b/{label}":
        return None
    return path, lines[4:]


def _styled_line(style: str, text: str, reset: str) -> str:
    if not text or not style:
        return text
    return f"{style}{text}{reset}"
