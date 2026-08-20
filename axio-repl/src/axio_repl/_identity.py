"""Stable effective-user identity shared by the UI and model runtime."""

from __future__ import annotations

import json
import os
import pwd
from collections.abc import Callable
from typing import Protocol

from axio_repl._multiplexer import normalize_agent_name

RUNTIME_METADATA_KEY = "axio_runtime_metadata"


class _PasswdRecord(Protocol):
    pw_name: str


def sanitize_effective_username(value: object, *, fallback: str) -> str:
    """Return one bounded terminal-safe identity or the numeric fallback."""

    return normalize_agent_name(str(value)) or fallback


def resolve_effective_username(
    *,
    geteuid: Callable[[], int] | None = None,
    getpwuid: Callable[[int], _PasswdRecord] | None = None,
) -> str:
    """Resolve the effective uid through NSS without trusting environment aliases."""

    uid = (geteuid or os.geteuid)()
    fallback = str(uid)
    try:
        record = (getpwuid or pwd.getpwuid)(uid)
    except (AttributeError, KeyError, OSError):
        return fallback
    return sanitize_effective_username(record.pw_name, fallback=fallback)


def append_runtime_identity_metadata(system: str, effective_username: str) -> str:
    """Append one compact stable JSON data record to a system prompt."""

    safe_username = sanitize_effective_username(effective_username, fallback="unknown")
    payload = json.dumps(
        {
            RUNTIME_METADATA_KEY: {
                "kind": "data",
                "effective_username": safe_username,
            }
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{system.rstrip()}\n\n{payload}"
