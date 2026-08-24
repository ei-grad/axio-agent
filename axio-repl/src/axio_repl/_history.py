"""Project-scoped persistent prompt history."""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import os
import re
import stat
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

from prompt_toolkit.history import FileHistory

_LABEL_LIMIT = 48
_DIGEST_LENGTH = 24
_SAFE_LABEL = re.compile(r"[^a-zA-Z0-9._-]+")


def default_state_home(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the absolute XDG state directory, ignoring invalid relative overrides."""

    values = os.environ if environ is None else environ
    configured = values.get("XDG_STATE_HOME", "")
    if configured:
        candidate = Path(configured)
        if candidate.is_absolute():
            return candidate
    base = Path.home() if home is None else home
    return base.expanduser().resolve() / ".local" / "state"


def canonical_project_root(project_root: Path) -> Path:
    """Resolve one stable absolute identity for a project directory."""

    return project_root.expanduser().resolve()


def project_history_path(project_root: Path, *, state_home: Path | None = None) -> Path:
    """Map a canonical project root to a bounded, inspectable history filename."""

    canonical_root = canonical_project_root(project_root)
    root_bytes = os.fsencode(canonical_root)
    digest = hashlib.sha256(root_bytes).hexdigest()[:_DIGEST_LENGTH]
    label = _project_label(canonical_root)
    storage_root = default_state_home() if state_home is None else state_home.expanduser().resolve()
    return storage_root / "axio" / "history" / f"{label}-{digest}.history"


def legacy_history_path(*, home: Path | None = None) -> Path:
    """Return the former global history location without reading or modifying it."""

    base = Path.home() if home is None else home
    return base.expanduser().resolve() / ".axio_repl_history"


def _project_label(project_root: Path) -> str:
    raw_label = "root" if project_root.parent == project_root else project_root.name
    normalized = unicodedata.normalize("NFKD", raw_label).encode("ascii", errors="ignore").decode("ascii")
    label = _SAFE_LABEL.sub("-", normalized).strip("-._")
    return (label or "project")[:_LABEL_LIMIT]


class PrivateFileHistory(FileHistory):
    """A prompt-toolkit history file with private modes and serialized appends."""

    def __init__(self, filename: Path) -> None:
        if not filename.name:
            raise ValueError("prompt history path must name a file")
        self.path = filename
        super().__init__(str(filename))

    def load_history_strings(self) -> Iterable[str]:
        fd = _open_history(self.path, os.O_RDONLY, create=False)
        if fd is None:
            return ()
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            with os.fdopen(fd, "rb", closefd=False) as stream:
                strings: list[str] = []
                lines: list[str] = []

                def add() -> None:
                    if lines:
                        strings.append("".join(lines)[:-1])

                for line_bytes in stream:
                    line = line_bytes.decode("utf-8", errors="replace")
                    if line.startswith("+"):
                        lines.append(line[1:])
                    else:
                        add()
                        lines = []
                add()
            return reversed(strings)
        finally:
            os.close(fd)

    def store_string(self, string: str) -> None:
        timestamp = datetime.datetime.now().astimezone()
        lines = "".join(f"+{line}\n" for line in string.split("\n"))
        payload = f"\n# {timestamp}\n{lines}".encode("utf-8", errors="replace")
        fd = _open_history(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, create=True)
        assert fd is not None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(fd, remaining)
                if written == 0:
                    raise OSError("history append made no progress")
                remaining = remaining[written:]
        finally:
            os.close(fd)


def _open_history(filename: Path, flags: int, *, create: bool) -> int | None:
    parent_fd = _open_parent_directory(filename, create=create)
    if parent_fd is None:
        return None
    no_follow = _no_follow_flag()
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    try:
        try:
            fd = os.open(filename.name, flags | no_follow | close_on_exec, 0o600, dir_fd=parent_fd)
        except FileNotFoundError:
            if not create:
                return None
            raise
    finally:
        os.close(parent_fd)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"prompt history is not a regular file: {filename}")
        os.fchmod(fd, 0o600)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_parent_directory(filename: Path, *, create: bool) -> int | None:
    root, controlled_components = _storage_layout(filename)
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    try:
        current_fd = os.open(root, directory_flags)
    except FileNotFoundError:
        if not create:
            return None
        raise
    try:
        if not controlled_components:
            os.fchmod(current_fd, 0o700)
        for component in controlled_components:
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
            try:
                next_fd = os.open(component, directory_flags | _no_follow_flag(), dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    os.close(current_fd)
                    return None
                raise
            os.close(current_fd)
            current_fd = next_fd
            os.fchmod(current_fd, 0o700)
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _storage_layout(filename: Path) -> tuple[Path, tuple[str, ...]]:
    parent = filename.parent
    if parent.name == "history" and parent.parent.name == "axio":
        return parent.parent.parent, ("axio", "history")
    return parent, ()


def _no_follow_flag() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow, int):
        raise RuntimeError("prompt history requires O_NOFOLLOW support")
    return no_follow
