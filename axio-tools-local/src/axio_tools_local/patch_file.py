from __future__ import annotations

import asyncio
import os
from pathlib import Path

from axio.diff import describe_patch
from axio.exceptions import HandlerError
from axio.field import StrictStr


async def patch_file(
    path: StrictStr,
    from_line: int,
    to_line: int,
    content: str,
    mode: int = 0o644,
) -> str:
    """Replace a range of lines in an existing UTF-8 text file. Lines are
    1-indexed: from_line and to_line are both inclusive (from_line=2, to_line=4
    replaces lines 2, 3, 4). To insert without deleting, set
    to_line = from_line - 1. Always read the file first with line_numbers=True
    to get correct line numbers. content is applied literally: include exact
    leading whitespace on the first and every following line, and do not copy
    read_file's ``L<number>│`` metadata prefix. Use this for surgical edits
    instead of rewriting the whole file with write_file. The result reports a
    compact diff fragment with function context when it can be inferred. Binary
    files cannot be patched."""

    def _blocking() -> str:
        resolved = Path(os.getcwd()) / path
        if not resolved.exists():
            raise HandlerError(f"No such file or directory: {path}")
        if not resolved.is_file():
            raise HandlerError(f"Not a file: {path}")

        try:
            with resolved.open("r") as f:
                before = f.read()

            lines = before.splitlines(keepends=True)
            content_lines = content.splitlines(keepends=True)
            if content_lines and not content_lines[-1].endswith("\n") and to_line < len(lines):
                content_lines[-1] += "\n"

            after = "".join(lines[: from_line - 1] + content_lines + lines[to_line:])
            with resolved.open("w") as f:
                f.write(after)
            os.chmod(resolved, mode)
        except UnicodeDecodeError as exc:
            raise HandlerError(f"File is not valid UTF-8: {path}") from exc
        except OSError as exc:
            raise HandlerError(f"{exc.strerror or exc}: {path}") from exc
        return describe_patch(path, before, after)

    return await asyncio.to_thread(_blocking)
