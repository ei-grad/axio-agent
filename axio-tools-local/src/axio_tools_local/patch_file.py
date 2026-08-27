from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Annotated

from axio.diff import (
    PATCH_CONTENT_DESCRIPTION,
    PATCH_FROM_LINE_DESCRIPTION,
    PATCH_TO_LINE_DESCRIPTION,
    describe_patch,
)
from axio.exceptions import HandlerError
from axio.field import Field, StrictStr


async def patch_file(
    path: StrictStr,
    from_line: Annotated[int, Field(description=PATCH_FROM_LINE_DESCRIPTION)],
    to_line: Annotated[int, Field(description=PATCH_TO_LINE_DESCRIPTION)],
    content: Annotated[str, Field(description=PATCH_CONTENT_DESCRIPTION)],
    mode: int = 0o644,
) -> str:
    """Replace a range of lines in an existing UTF-8 text file. Lines are
    1-indexed: from_line and to_line are both inclusive (from_line=2, to_line=4
    replaces lines 2, 3, 4). To insert without deleting, set
    to_line = from_line - 1; this selects an empty old range and inserts before
    from_line. For replacement, the range names every old physical line removed,
    including unchanged lines retained in content; it is not merely the lines
    whose logic changes. Always read the file first with line_numbers=True to get
    correct line numbers. content is applied literally: include exact
    leading whitespace on the first and every following line, and do not copy
    read_file's ``L<number>│`` metadata prefix. Use this for surgical edits
    instead of rewriting the whole file with write_file. The result reports a
    compact diff fragment with function context when it can be inferred. Binary
    files cannot be patched. Multiple sequential patches are allowed; use the
    diff returned by each call as the authoritative description of the new
    file state."""

    def _blocking() -> str:
        resolved = (Path(os.getcwd()) / path).resolve()
        if not resolved.exists():
            raise HandlerError(f"No such file or directory: {path}")
        if not resolved.is_file():
            raise HandlerError(f"Not a file: {path}")

        try:
            with resolved.open("r", encoding="utf-8", newline="") as f:
                before = f.read()

            lines = before.splitlines(keepends=True)
            line_count = len(lines)
            if not 1 <= from_line <= line_count + 1:
                raise HandlerError(
                    f"from_line={from_line} out of range; file has {line_count} lines (valid: 1..{line_count + 1})"
                )
            if not from_line - 1 <= to_line <= line_count:
                raise HandlerError(f"to_line={to_line} out of range (valid: {from_line - 1}..{line_count})")
            content_lines = content.splitlines(keepends=True)
            if content_lines and not content_lines[-1].endswith("\n") and to_line < len(lines):
                content_lines[-1] += "\n"

            after = "".join(lines[: from_line - 1] + content_lines + lines[to_line:])
            diff = describe_patch(path, before, after)
            encoded = after.encode("utf-8")
            descriptor, temporary = tempfile.mkstemp(dir=resolved.parent)
            try:
                with os.fdopen(descriptor, "wb") as f:
                    f.write(encoded)
                os.chmod(temporary, mode)
                os.replace(temporary, resolved)
            except OSError:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
                raise
        except UnicodeDecodeError as exc:
            raise HandlerError(f"File is not valid UTF-8: {path}") from exc
        except UnicodeEncodeError as exc:
            raise HandlerError("content is not valid UTF-8") from exc
        except OSError as exc:
            raise HandlerError(f"{exc.strerror or exc}: {path}") from exc
        return diff

    return await asyncio.to_thread(_blocking)
