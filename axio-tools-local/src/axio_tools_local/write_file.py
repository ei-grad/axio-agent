from __future__ import annotations

import asyncio
import os

from axio.diff import render_diff
from axio.exceptions import HandlerError
from axio.field import StrictStr


async def write_file(
    path: StrictStr,
    content: str,
    mode: int = 0o644,
) -> str:
    """Create or overwrite a file with the given content. Parent directories
    are created automatically. Use this for new files or full rewrites.
    For partial edits prefer patch_file instead."""

    def _blocking() -> str:
        resolved = os.path.join(os.getcwd(), path)
        existed = os.path.exists(resolved)
        before = ""
        if existed:
            try:
                with open(resolved) as f:
                    before = f.read()
            except UnicodeDecodeError as exc:
                raise HandlerError(f"File is not valid UTF-8: {path}") from exc
            except OSError as exc:
                raise HandlerError(f"{exc.strerror or exc}: {path}") from exc
        try:
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
            with open(resolved, "w") as f:
                f.write(content)
            os.chmod(resolved, mode=mode)
        except OSError as exc:
            raise HandlerError(f"{exc.strerror or exc}: {path}") from exc
        size_note = f"{len(content)} bytes"
        if existed:
            return f"Wrote {size_note} to {path}\n{render_diff(path, before, content)}"
        return f"Wrote {size_note} to {path}"

    return await asyncio.to_thread(_blocking)
