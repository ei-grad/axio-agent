"""Local executable and source provenance for ``axio-repl --version``."""

from __future__ import annotations

import shutil
import sys
import tomllib
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def installed_distribution_version() -> str:
    """Return the installed distribution version without importing providers."""

    try:
        return version("axio-repl")
    except PackageNotFoundError:
        return "unavailable"


def resolved_launcher(argv0: str) -> str:
    """Resolve an entry-point path while retaining a useful fallback."""

    if not argv0:
        return "unavailable"
    candidate = argv0
    if not Path(argv0).is_absolute() and len(Path(argv0).parts) == 1:
        located = shutil.which(argv0)
        if located is None:
            return argv0
        candidate = located
    try:
        return str(Path(candidate).expanduser().resolve(strict=False))
    except OSError:
        return candidate


def resolved_source(source: str | None) -> str:
    """Resolve a module source path without requiring that it still exists."""

    if not source:
        return "unavailable"
    try:
        return str(Path(source).expanduser().resolve(strict=False))
    except OSError:
        return source


def _git_directories(marker: Path) -> tuple[Path, Path] | None:
    if marker.is_dir():
        return marker, marker
    if not marker.is_file():
        return None
    try:
        marker_value = marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    prefix = "gitdir:"
    if not marker_value.lower().startswith(prefix):
        return None
    git_value = marker_value[len(prefix) :].strip()
    if not git_value:
        return None
    git_dir = Path(git_value)
    if not git_dir.is_absolute():
        git_dir = marker.parent / git_dir
    try:
        git_dir = git_dir.resolve(strict=False)
        common_file = git_dir / "commondir"
        if common_file.is_file():
            common_value = common_file.read_text(encoding="utf-8").strip()
            common_dir = Path(common_value)
            if not common_dir.is_absolute():
                common_dir = git_dir / common_dir
            common_dir = common_dir.resolve(strict=False)
        else:
            common_dir = git_dir
    except (OSError, UnicodeError):
        return None
    return git_dir, common_dir


def _read_git_ref(git_dir: Path, common_dir: Path, ref: str) -> str | None:
    for root in (git_dir, common_dir):
        try:
            value = (root / ref).read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            continue
        if value:
            return value
    try:
        packed_refs = (common_dir / "packed-refs").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        return None
    suffix = f" {ref}"
    for line in packed_refs:
        if line.startswith(("#", "^")) or not line.endswith(suffix):
            continue
        return line.split(" ", 1)[0]
    return None


def _nearest_git_marker(current: Path) -> Path | None:
    """Return the nearest lexical marker, including broken symlinks."""

    for candidate in (current, *current.parents):
        marker = candidate / ".git"
        try:
            marker.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            return marker
        return marker
    return None


def _is_axio_repl_checkout(source: Path, checkout_root: Path) -> bool:
    try:
        source.relative_to(checkout_root)
    except ValueError:
        return False
    current = source if source.is_dir() else source.parent
    while True:
        manifest = current / "pyproject.toml"
        try:
            data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            pass
        else:
            project = data.get("project")
            if isinstance(project, dict) and project.get("name") == "axio-repl":
                try:
                    relative_source = source.relative_to(current)
                except ValueError:
                    pass
                else:
                    source_parts = relative_source.parts
                    if source_parts[:2] == ("src", "axio_repl") or source_parts[:1] == ("axio_repl",):
                        return True
        if current == checkout_root:
            return False
        current = current.parent


def git_revision(module_source: str | None) -> str:
    """Read a short checkout revision without invoking Git or the network."""

    if not module_source:
        return "unavailable"
    try:
        source = Path(module_source).expanduser().resolve(strict=False)
    except OSError:
        return "unavailable"
    current = source if source.is_dir() else source.parent
    marker = _nearest_git_marker(current)
    if marker is None:
        return "unavailable"
    if not _is_axio_repl_checkout(source, marker.parent):
        return "unavailable"
    directories = _git_directories(marker)
    if directories is None:
        return "unavailable"
    git_dir, common_dir = directories
    try:
        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return "unavailable"
    revision = _read_git_ref(git_dir, common_dir, head.removeprefix("ref: ")) if head.startswith("ref: ") else head
    if revision is None or len(revision) < 7:
        return "unavailable"
    if any(character not in "0123456789abcdefABCDEF" for character in revision):
        return "unavailable"
    return revision[:12].lower()


def version_report(
    *,
    argv0: str | None = None,
    module_source: str | None = None,
    interpreter: str | None = None,
    distribution_version_provider: Callable[[], str] | None = None,
    git_revision_provider: Callable[[str | None], str] | None = None,
) -> str:
    """Build the stable multi-line provenance report."""

    source = resolved_source(module_source)
    distribution = (distribution_version_provider or installed_distribution_version)()
    revision = (git_revision_provider or git_revision)(None if source == "unavailable" else source)
    return "\n".join(
        (
            f"axio-repl {distribution}",
            f"launcher: {resolved_launcher(sys.argv[0] if argv0 is None else argv0)}",
            f"module: {source}",
            f"interpreter: {resolved_source(sys.executable if interpreter is None else interpreter)}",
            f"git revision: {revision}",
        )
    )
