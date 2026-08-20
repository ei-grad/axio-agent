"""Bounded display-only parsing for file-tool results."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from axio_repl._theme import TerminalTheme

_MAX_RESULT_BYTES = 32 * 1024
_MAX_HEADER_CHARS = 512
_SUMMARY = re.compile(r"^\+(\d+) -(\d+)$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(?: [A-Za-z_$~][A-Za-z0-9_$~:.<>]*)?$")
_WRITE = re.compile(r"^Wrote \d+ bytes to (.+)$")
_TRUNCATED = "...[diff truncated]"


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

    def styled_text(self, theme: TerminalTheme) -> str:
        rendered: list[str] = []
        if self.fallback_path is not None:
            rendered.append(_styled_line(theme.stdout.ansi, f"path: {self.fallback_path}", theme.reset))
        for line in self.lines:
            style = {
                DiffLineKind.SUMMARY: theme.stdout.ansi,
                DiffLineKind.HUNK: theme.tool.ansi,
                DiffLineKind.ADDITION: theme.success.ansi,
                DiffLineKind.DELETION: theme.error.ansi,
                DiffLineKind.CONTEXT: theme.reasoning.ansi,
                DiffLineKind.METADATA: theme.stdout.ansi,
            }[line.kind]
            rendered.append(_styled_line(style, line.text, theme.reset))
        return "\n".join(rendered)


def parse_patch_result(content: str, *, include_legacy_path: bool = False) -> PatchResultDisplay | None:
    """Parse a compact patch diff, failing open with ``None`` on any mismatch."""
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
    saw_hunk = False
    saw_change_in_hunk = False
    truncated = False
    for index, line in enumerate(raw_lines):
        if line == _TRUNCATED:
            if index != len(raw_lines) - 1 or not saw_hunk:
                return None
            parsed.append(DiffDisplayLine(line, DiffLineKind.METADATA))
            truncated = True
            continue
        if line.startswith("@@"):
            if len(line) > _MAX_HEADER_CHARS or _HUNK.fullmatch(line) is None:
                return None
            if saw_hunk and not saw_change_in_hunk:
                return None
            parsed.append(DiffDisplayLine(line, DiffLineKind.HUNK))
            saw_hunk = True
            saw_change_in_hunk = False
            continue
        if not saw_hunk:
            return None
        if line.startswith("+"):
            additions += 1
            saw_change_in_hunk = True
            kind = DiffLineKind.ADDITION
        elif line.startswith("-"):
            deletions += 1
            saw_change_in_hunk = True
            kind = DiffLineKind.DELETION
        elif line.startswith(" "):
            kind = DiffLineKind.CONTEXT
        elif line.startswith("\\"):
            kind = DiffLineKind.METADATA
        else:
            return None
        parsed.append(DiffDisplayLine(line, kind))
    if not saw_hunk or (not saw_change_in_hunk and not truncated):
        return None
    if summary is not None and not truncated and summary != (additions, deletions):
        return None
    if summary is None:
        summary = (additions, deletions)
    lines = (DiffDisplayLine(f"+{summary[0]} -{summary[1]}", DiffLineKind.SUMMARY), *parsed)
    return PatchResultDisplay(lines=lines, fallback_path=fallback_path)


def is_write_file_result(content: str) -> bool:
    """Recognize only the exact successful result shapes produced by write_file."""
    if not content or "\r" in content or len(content.encode("utf-8")) > _MAX_RESULT_BYTES:
        return False
    lines = content.splitlines()
    if len(lines) == 1:
        return _WRITE.fullmatch(lines[0]) is not None
    return _legacy_body(lines) is not None and parse_patch_result(content) is not None


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
    if not style:
        return text
    return f"{style}{text}{reset}"
