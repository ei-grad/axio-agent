from __future__ import annotations

import asyncio
import os

from axio.diff import MAX_DIFF_SOURCE_BYTES, describe_write
from axio.exceptions import HandlerError
from axio.field import StrictStr


async def write_file(
    path: StrictStr,
    content: str,
    mode: int = 0o644,
) -> str:
    """Create or overwrite a file with UTF-8 text. Parent directories are
    created automatically. Use this for new files or full rewrites; for partial
    edits prefer patch_file instead. Overwriting an existing text file reports a
    unified diff of the change. Binary files can be replaced but not written
    with this tool, and their replacement reports no diff."""

    def _blocking() -> str:
        resolved = os.path.join(os.getcwd(), path)
        before = _previous_content(resolved)
        try:
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
            with open(resolved, "w") as f:
                f.write(content)
            os.chmod(resolved, mode=mode)
        except OSError as exc:
            raise HandlerError(f"{exc.strerror or exc}: {path}") from exc
        return describe_write(path, len(content.encode()), before, content)

    return await asyncio.to_thread(_blocking)


def _previous_content(resolved: str) -> str | None:
    """Return the text to diff the write against, or None when there is none.

    A missing file has no previous version, a binary one cannot be diffed, and
    re-reading a huge file costs more than the diff is worth. None of those may
    block the write itself, so every one of them degrades to "no diff" instead
    of raising.
    """
    try:
        if os.path.getsize(resolved) > MAX_DIFF_SOURCE_BYTES:
            return None
        with open(resolved, encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None
