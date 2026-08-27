"""DockerSandbox: async context manager for sandboxed Docker execution."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import io
import logging
import os
import posixpath
import shlex
import stat as stat_module
import tarfile
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Any, cast

import aiodocker
from aiodocker.exceptions import DockerError
from axio._asyncio import shield_until_done
from axio.diff import (
    MAX_DIFF_SOURCE_BYTES,
    PATCH_CONTENT_DESCRIPTION,
    PATCH_FROM_LINE_DESCRIPTION,
    PATCH_TO_LINE_DESCRIPTION,
    describe_patch,
    describe_write,
)
from axio.exceptions import HandlerError
from axio.field import Field
from axio.schema import build_tool_schema
from axio.tool import CONTEXT, Tool

logger = logging.getLogger(__name__)

_SHELL_PREFERENCE = ("bash", "sh", "zsh", "dash")
_DEFAULT_CONTAINER_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_SHELL_PROBE_TIMEOUT = 2.0


class ImageNotAvailableError(RuntimeError):
    """Raised when a local-only sandbox image is absent from the daemon."""


@dataclass(frozen=True, slots=True)
class _ShellExecutable:
    name: str
    path: str


def _shell_choices_text(available: tuple[_ShellExecutable, ...]) -> str:
    if not available:
        return "No supported shells were found in the container PATH."
    names = ", ".join(item.name for item in available)
    return f"Available shells: {names}. Omit shell to use {available[0].name}."


def _select_shell(requested: str | None, available: tuple[_ShellExecutable, ...]) -> _ShellExecutable:
    if not available:
        tried = ", ".join(_SHELL_PREFERENCE)
        raise HandlerError(f"No supported shell found in the container PATH (tried: {tried})")
    if requested is None:
        return available[0]
    for item in available:
        if requested == item.name:
            return item
    choices = ", ".join(item.name for item in available)
    raise HandlerError(f"Unknown shell {requested!r}; available shells: {choices}")


def parse_memory(s: str) -> int:
    """Parse human-readable memory string to bytes: "256m" → 268435456."""
    units = {"k": 1024, "m": 1024**2, "g": 1024**3}
    s = s.lower().strip()
    if s[-1] in units:
        return int(s[:-1]) * units[s[-1]]
    return int(s)


def parse_cpus(s: str) -> int:
    """Parse CPU string to NanoCPUs: "1.0" → 1_000_000_000."""
    return int(float(s) * 1_000_000_000)


def _resolve_path(workdir: str, path: str) -> str:
    """Resolve a possibly-relative path against the container workdir."""
    if os.path.isabs(path):
        return path
    return os.path.join(workdir, path)


def parse_device(s: str) -> dict[str, str]:
    """Parse a device string into a Docker device mapping dict.

    Accepted formats (mirrors ``docker run --device``):
    - ``/dev/sda`` → host=/dev/sda, container=/dev/sda, perms=rwm
    - ``/dev/sda:/dev/xvda`` → host=/dev/sda, container=/dev/xvda, perms=rwm
    - ``/dev/sda:/dev/xvda:r`` → explicit permissions
    """
    parts = s.split(":")
    host = parts[0]
    container = parts[1] if len(parts) > 1 else host
    perms = parts[2] if len(parts) > 2 else "rwm"
    return {"PathOnHost": host, "PathInContainer": container, "CgroupPermissions": perms}


# ---------------------------------------------------------------------------
# Tool handlers - plain async functions, context via CONTEXT.get()
# ---------------------------------------------------------------------------


class _ShellControl(str):
    pass


type _ShellRecord = tuple[float, str, str]


def _format_shell_records(records: list[_ShellRecord]) -> str:
    segments: list[tuple[str, str]] = []
    for _, key, text in records:
        if (
            segments
            and segments[-1][0] == key
            and not isinstance(segments[-1][1], _ShellControl)
            and not isinstance(text, _ShellControl)
        ):
            previous_key, previous_text = segments[-1]
            segments[-1] = previous_key, previous_text + text
        else:
            segments.append((key, text))

    output = ""
    current_key: str | None = None
    for key, text in segments:
        if isinstance(text, _ShellControl):
            output += ("\n" if output else "") + text
            current_key = None
            continue
        if key != current_key:
            if output or key != "stdout":
                output += ("\n" if output else "") + f"[{key}]\n"
            current_key = key
        output += text
    if not output.strip():
        return "(no output)"
    return output.rstrip("\n")


async def _shell_stream(
    command: str,
    timeout: float = 5,
    cwd: str = ".",
    stdin: str | None = None,
    shell: str | None = None,
) -> AsyncGenerator[tuple[str, str], None]:
    sandbox: DockerSandbox = CONTEXT.get()
    resolved = _resolve_path(sandbox.workdir, cwd)
    cmd = f"cd {shlex.quote(resolved)} && {command}"
    async for chunk in sandbox.exec_stream(cmd, timeout=timeout, stdin=stdin, shell=shell):
        yield chunk


async def shell(
    command: str,
    timeout: float = 5,
    cwd: str = ".",
    stdin: str | None = None,
    shell: str | None = None,
) -> str:
    """Run a shell command and return combined stdout/stderr. Use for git,
    build tools, grep, tests, or any CLI operation. Non-zero exit codes
    are reported. Optionally pass stdin data for commands that read from
    standard input. Live and final output preserve Docker-observed
    stdout/stderr order; this does not imply a global syscall order across the
    two descriptors. Commands default to the first shell discovered in the
    container (bash is preferred); pass shell to select another advertised
    shell. Commands use non-login ``-c`` mode and do not load login profiles.
    Prefer short timeouts and avoid interactive commands."""
    records: list[_ShellRecord] = []
    async for key, text in _shell_stream(command, timeout, cwd, stdin, shell):
        records.append((0.0, key, text))
    return _format_shell_records(records)


shell.stream = _shell_stream  # type: ignore[attr-defined]
shell.format_stream_result = staticmethod(_format_shell_records)  # type: ignore[attr-defined]


async def write_file(path: str, content: str, mode: int = 0o644) -> str:
    """Create or overwrite a file with UTF-8 text. Parent directories are
    created automatically. Use this for new files or full rewrites; for partial
    edits prefer patch_file instead. Overwriting an existing text file reports a
    unified diff of the change. Binary files can be replaced but not written
    with this tool, and their replacement reports no diff."""
    sandbox: DockerSandbox = CONTEXT.get()
    resolved = _resolve_path(sandbox.workdir, path)
    before = await _previous_content(sandbox, resolved)
    await sandbox.write_file(resolved, content, mode=mode)
    return describe_write(path, len(content.encode()), before, content)


async def _previous_content(sandbox: DockerSandbox, resolved: str) -> str | None:
    """Return the text to diff the write against, or None when there is none.

    A missing file has no previous version, a binary one cannot be diffed, and
    pulling a huge file back out of the container costs more than the diff is
    worth. None of those may block the write itself, so every one of them
    degrades to "no diff" instead of raising. One archive fetch answers both
    "does it exist" and "what did it hold".
    """
    try:
        raw = await sandbox.read_file_bytes(resolved)
    except FileNotFoundError:
        return None
    if len(raw) > MAX_DIFF_SOURCE_BYTES:
        return None
    try:
        return raw.decode()
    except UnicodeDecodeError:
        return None


async def read_file(
    path: str,
    max_chars: int = 32768,
    binary_as_hex: bool = True,
    start_line: int | None = None,
    end_line: int | None = None,
    line_numbers: bool = False,
) -> str:
    """Read file contents. Returns text for text files, hex for binaries.
    Lines are 1-indexed: start_line=1 is the first line, end_line=3 includes
    line 3. Pass line_numbers=True to prefix each line as
    ``L<number>│<source>`` — required before calling patch_file. Everything
    after ``│`` is exact file content; the whole prefix is display metadata and
    must not be copied into patch_file content. Large files are truncated to
    max_chars. Always read the file before editing it with write_file or
    patch_file."""
    sandbox: DockerSandbox = CONTEXT.get()
    resolved = _resolve_path(sandbox.workdir, path)
    if max_chars < 0:
        raise HandlerError(f"max_chars must be >= 0: {resolved}")
    try:
        raw = await sandbox.read_file_bytes(resolved)
    except FileNotFoundError as exc:
        raise HandlerError(str(exc)) from exc
    try:
        text = raw.decode()
    except UnicodeDecodeError as exc:
        if binary_as_hex:
            return "Encoded binary data HEX: " + raw[:max_chars].hex()
        raise HandlerError(f"File is not valid UTF-8: {resolved}") from exc
    all_lines = text.splitlines(keepends=True)
    start = 0 if start_line is None else start_line - 1
    end = len(all_lines) if end_line is None else end_line
    selected = all_lines[start:end]
    if line_numbers:
        result = "".join(f"L{start + 1 + i}│{line}" for i, line in enumerate(selected))
    else:
        result = "".join(selected)
    if len(result) > max_chars:
        return result[:max_chars] + "\n...[truncated]"
    return result


async def list_files(path: str = ".") -> str:
    """List files and directories. Shows permissions, size, modification time,
    and name for each entry. Directories are listed first and marked with
    a trailing slash. Use this to explore the project structure before
    reading or editing files."""
    sandbox: DockerSandbox = CONTEXT.get()
    resolved = _resolve_path(sandbox.workdir, path)
    target = shlex.quote(resolved)
    command = f"""
target={target}
if [ ! -e "$target" ] && [ ! -L "$target" ]; then
    printf 'AXIO_LIST_ERROR\\0missing\\0'
    exit 0
fi
if [ ! -d "$target" ]; then
    printf 'AXIO_LIST_ERROR\\0not-directory\\0'
    exit 0
fi
if [ ! -r "$target" ] || [ ! -x "$target" ]; then
    printf 'AXIO_LIST_ERROR\\0permission-denied\\0'
    exit 0
fi
printf 'AXIO_LIST_V3\\0'
cd "$target" || exit 1
find . ! -path . -prune -exec sh -c '
    printf "AXIO_LIST_BATCH\\0%s\\0" "$#"
    for entry in "$@"; do
        printf "%s\\0" "$entry"
    done
    stat -c "%f %s %Y" -- "$@"
' sh {{}} +
"""
    stdout: list[str] = []
    stderr: list[str] = []
    controls: list[str] = []
    async for key, text in sandbox.exec_stream(command):
        if isinstance(text, _ShellControl):
            controls.append(text)
        elif key == "stdout":
            stdout.append(text)
        else:
            stderr.append(text)

    output = "".join(stdout)
    if stderr or controls:
        details = []
        stderr_text = "".join(stderr).strip()
        if stderr_text:
            details.append(stderr_text)
        details.extend(controls)
        detail = "; ".join(details) or "command failed"
        raise HandlerError(f"Failed to list directory {resolved}: {detail}")

    fields = output.split("\0", maxsplit=2)
    if fields[:2] == ["AXIO_LIST_ERROR", "missing"]:
        raise HandlerError(f"No such file or directory: {resolved}")
    if fields[:2] == ["AXIO_LIST_ERROR", "not-directory"]:
        raise HandlerError(f"Not a directory: {resolved}")
    if fields[:2] == ["AXIO_LIST_ERROR", "permission-denied"]:
        raise HandlerError(f"Permission denied: {resolved}")
    if not output.startswith("AXIO_LIST_V3\0"):
        raise HandlerError(f"Failed to list directory {resolved}: {output}")

    entries: list[tuple[bool, str, int, float, str]] = []
    try:
        payload = output.removeprefix("AXIO_LIST_V3\0")
        while payload:
            batch_header, count_text, payload = payload.split("\0", maxsplit=2)
            if batch_header != "AXIO_LIST_BATCH":
                raise ValueError("invalid batch header")
            entry_count = int(count_text)
            if entry_count <= 0:
                raise ValueError("invalid batch entry count")
            name_fields = payload.split("\0", maxsplit=entry_count)
            if len(name_fields) != entry_count + 1:
                raise ValueError("missing entry names")
            names = name_fields[:entry_count]
            payload = name_fields[-1]
            for full_name in names:
                metadata, separator, payload = payload.partition("\n")
                if not separator:
                    raise ValueError("missing entry metadata")
                mode_text, size_text, mtime_text = metadata.split()
                mode = int(mode_text, 16)
                size = int(size_text)
                mtime = float(mtime_text)
                name = os.path.basename(full_name)
                entries.append((stat_module.S_ISDIR(mode), stat_module.filemode(mode), size, mtime, name))
    except (ValueError, OverflowError) as exc:
        raise HandlerError(f"Failed to parse directory listing for {resolved}") from exc

    entries.sort(key=lambda entry: (not entry[0], entry[4]))
    if not entries:
        return "(empty directory)"
    return "\n".join(
        f"{mode} {size:>8} {datetime.fromtimestamp(mtime).strftime('%b %d %H:%M')} {name}{'/' if is_dir else ''}"
        for is_dir, mode, size, mtime, name in entries
    )


async def run_python(code: str, cwd: str = ".", timeout: float = 5, stdin: str | None = None) -> str:
    """Run a Python code snippet in a subprocess and return stdout/stderr.
    The code is written to a temp file and executed with the current
    interpreter. Use for calculations, data processing, or testing
    small scripts. Optionally pass stdin data. Non-zero exit codes
    and tracebacks are returned as-is."""
    sandbox: DockerSandbox = CONTEXT.get()
    resolved = _resolve_path(sandbox.workdir, cwd)
    tmp = f"/tmp/.axio_{uuid.uuid4().hex}.py"
    await sandbox.write_file(tmp, code)
    cmd = f"cd {shlex.quote(resolved)} && python3 {tmp}; _rc=$?; rm -f {tmp}; exit $_rc"
    return await sandbox.exec(cmd, timeout=timeout, stdin=stdin)


async def patch_file(
    path: str,
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
    files cannot be patched."""
    sandbox: DockerSandbox = CONTEXT.get()
    resolved = _resolve_path(sandbox.workdir, path)
    try:
        raw = await sandbox.read_file_bytes(resolved)
    except FileNotFoundError as exc:
        raise HandlerError(str(exc)) from exc
    try:
        before = raw.decode()
        lines = before.splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise HandlerError(f"File is not valid UTF-8: {resolved}") from exc
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
    await sandbox.write_file(resolved, after, mode=mode)
    return describe_patch(path, before, after)


# ---------------------------------------------------------------------------
# DockerSandbox
# ---------------------------------------------------------------------------


class DockerSandbox:
    """Async context manager that provides a sandboxed Docker container with axio tools."""

    def __init__(
        self,
        url: str = "unix:///var/run/docker.sock",
        *,
        image: str = "python:latest",
        memory: str = "256m",
        cpus: str = "1.0",
        network: bool | str = False,
        workdir: str = "/workspace",
        volumes: dict[str, str] | None = None,
        read_only_volumes: dict[str, str] | None = None,
        named_volumes: dict[str, str] | None = None,
        volumes_remove: bool = False,
        env: dict[str, str] | None = None,
        user: str = "",
        group_add: list[str] | None = None,
        name: str = "",
        remove: bool = True,
        read_only: bool = False,
        shm_size: str = "",
        cap_add: list[str] | None = None,
        cap_drop: list[str] | None = None,
        privileged: bool = False,
        ulimits: dict[str, int | tuple[int, int]] | None = None,
        tmpfs: dict[str, str] | None = None,
        ports: dict[int, int] | None = None,
        platform: str = "",
        extra_hosts: dict[str, str] | None = None,
        devices: list[str] | None = None,
        dns: list[str] | None = None,
        require_internal_network: bool = False,
        pull_missing: bool = True,
    ) -> None:
        """Create a DockerSandbox.

        Args:
            url: Docker daemon URL (unix socket or TCP).
            image: Container image to use.
            memory: Memory limit, e.g. "256m", "1g".
            cpus: CPU limit, e.g. "1.0".
            network: Network mode. ``False`` disables networking entirely
                (``NetworkMode: none``). ``True`` uses the Docker default.
                A string sets ``NetworkMode`` explicitly, e.g. ``"host"``,
                ``"bridge"``, or a named network like ``"my-project_default"``.
            workdir: Working directory inside the container.
            volumes: Mapping of {container_path: host_path} bind mounts.
            read_only_volumes: Mapping of {container_path: host_path} read-only bind mounts.
            named_volumes: Mapping of {container_path: volume_name} named Docker volumes.
                Docker creates the volume automatically if it does not exist. Named volumes
                persist across container restarts and can be shared between sandbox sessions.
            volumes_remove: Remove named volumes on exit. Has no effect when attaching to
                an existing container (``name=`` reuse) or when ``named_volumes`` is empty.
            env: Environment variables passed to all commands, e.g. {"PYTHONPATH": "/app"}.
            user: User to run as inside the container, e.g. "1000" or "nobody".
            group_add: Supplementary group names or numeric IDs for the container process.
            name: Container name. If a container with this name already exists and
                is running, the sandbox attaches to it instead of creating a new one
                and will not remove it on exit. If no container exists, a new one is
                created (and removed on exit if ``remove=True``).
            remove: Remove the container on exit (default: True). Has no effect when
                attaching to an existing container.
            read_only: Mount the container's root filesystem as read-only.
            shm_size: Size of ``/dev/shm``, e.g. ``"64m"``, ``"1g"``.
            cap_add: Linux capabilities to add, e.g. ``["NET_ADMIN", "SYS_PTRACE"]``.
            cap_drop: Linux capabilities to drop, e.g. ``["ALL"]``.
            privileged: Give extended privileges to the container (implies full
                capability set and device access). Use with care.
            ulimits: Resource limits as ``{name: value}`` or ``{name: (soft, hard)}``.
                A single integer sets soft == hard. Examples: ``{"nofile": 1024}``,
                ``{"nofile": (1024, 65536), "nproc": 512}``.
            tmpfs: Tmpfs mounts as ``{path: options}``, e.g.
                ``{"/tmp": "size=128m,mode=1777"}``. An empty string for options
                uses Docker defaults.
            ports: Port bindings as ``{container_port: host_port}``, e.g.
                ``{8080: 8080}``. Only meaningful when ``network`` is not ``False``.
            platform: Platform string for the container image, e.g.
                ``"linux/amd64"`` or ``"linux/arm64"``.
            extra_hosts: Additional ``/etc/hosts`` entries as ``{hostname: ip}``,
                e.g. ``{"host.docker.internal": "host-gateway"}``.
            devices: Host devices to expose inside the container. Each entry
                follows the ``docker run --device`` format:
                ``"/dev/sda"`` (maps to same path, permissions ``rwm``),
                ``"/dev/sda:/dev/xvda"`` (custom container path),
                ``"/dev/sda:/dev/xvda:r"`` (read-only).
            dns: DNS servers to use inside the container, e.g.
                ``["8.8.8.8", "1.1.1.1"]``.
            require_internal_network: When true, ``network`` must name a Docker network
                configured with ``Internal=true``. Container creation fails closed if the
                network is missing or externally routed.
            pull_missing: Pull ``image`` when absent locally. When false, fail without
                contacting a registry.
        """
        if require_internal_network and not isinstance(network, str):
            raise ValueError("require_internal_network needs a named Docker network")
        if require_internal_network and name:
            raise ValueError("require_internal_network cannot be combined with named-container reuse")
        self.url = url
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.network: bool | str = network
        self.workdir = workdir
        self.volumes: dict[str, str] = volumes or {}
        self.read_only_volumes: dict[str, str] = read_only_volumes or {}
        self.named_volumes: dict[str, str] = named_volumes or {}
        self.volumes_remove = volumes_remove
        self.env: dict[str, str] = env or {}
        self.user = user
        self.group_add: list[str] = group_add or []
        self.name = name
        self.remove = remove
        self.read_only = read_only
        self.shm_size = shm_size
        self.cap_add: list[str] = cap_add or []
        self.cap_drop: list[str] = cap_drop or []
        self.privileged = privileged
        self.ulimits: dict[str, int | tuple[int, int]] = ulimits or {}
        self.tmpfs: dict[str, str] = tmpfs or {}
        self.ports: dict[int, int] = ports or {}
        self.platform = platform
        self.extra_hosts: dict[str, str] = extra_hosts or {}
        self.devices: list[str] = devices or []
        self.dns: list[str] = dns or []
        self.require_internal_network = require_internal_network
        self.pull_missing = pull_missing
        self._verified_network_id: str | None = None
        self._client: aiodocker.Docker | None = None
        self._container: aiodocker.containers.DockerContainer | None = None
        self._attached: bool = False  # True when we reused an existing container
        self._shells: tuple[_ShellExecutable, ...] | None = None

    async def __aenter__(self) -> DockerSandbox:
        self._client = aiodocker.Docker(url=self.url)
        try:
            await self._client.system.info()
        except Exception as exc:
            await self._client.close()
            self._client = None
            raise RuntimeError(f"Docker daemon not available at {self.url!r}: {exc}") from exc

        if self.require_internal_network:
            assert isinstance(self.network, str)
            try:
                docker_network = await self._client.networks.get(self.network)
                network_info = await docker_network.show()
            except Exception as exc:  # network lookup may fail through Docker, HTTP, or socket layers
                await self._client.close()
                self._client = None
                raise RuntimeError(f"Required internal Docker network {self.network!r} is unavailable: {exc}") from exc
            if not isinstance(network_info, dict) or network_info.get("Internal") is not True:
                await self._client.close()
                self._client = None
                raise RuntimeError(f"Docker network {self.network!r} is not internal; refusing sandbox egress")
            network_id = network_info.get("Id")
            if not isinstance(network_id, str) or not network_id.strip() or network_id != network_id.strip():
                await self._client.close()
                self._client = None
                raise RuntimeError(f"Docker network {self.network!r} returned no stable ID; refusing sandbox egress")
            self._verified_network_id = network_id

        if self.name:
            try:
                self._container = await self._client.containers.get(self.name)
                info = await self._container.show()
                if not info.get("State", {}).get("Running", False):
                    await self._container.start()
                self._attached = True
                logger.info("Attached to existing container (name=%s)", self.name)
            except aiodocker.exceptions.DockerError:
                self._attached = False

        if not self._attached:
            try:
                await self.ensure_image()
            except Exception:  # inspection and pull failures can originate in Docker, HTTP, or the socket
                self._verified_network_id = None
                await self._client.close()
                self._client = None
                raise
            binds = [f"{host}:{container}" for container, host in self.volumes.items()]
            binds += [f"{host}:{container}:ro" for container, host in self.read_only_volumes.items()]
            binds += [f"{vol}:{path}" for path, vol in self.named_volumes.items()]
            host_config: dict[str, Any] = {
                "Init": True,
                "Memory": parse_memory(self.memory),
                "NanoCPUs": parse_cpus(self.cpus),
                "Binds": binds,
            }
            if self.network is False:
                host_config["NetworkMode"] = "none"
            elif isinstance(self.network, str):
                host_config["NetworkMode"] = self._verified_network_id or self.network
            if self.read_only:
                host_config["ReadonlyRootfs"] = True
            if self.shm_size:
                host_config["ShmSize"] = parse_memory(self.shm_size)
            if self.cap_add:
                host_config["CapAdd"] = self.cap_add
            if self.cap_drop:
                host_config["CapDrop"] = self.cap_drop
            if self.privileged:
                host_config["Privileged"] = True
            if self.ulimits:
                host_config["Ulimits"] = [
                    {
                        "Name": limit_name,
                        "Soft": val if isinstance(val, int) else val[0],
                        "Hard": val if isinstance(val, int) else val[1],
                    }
                    for limit_name, val in self.ulimits.items()
                ]
            if self.tmpfs:
                host_config["Tmpfs"] = self.tmpfs
            if self.ports:
                host_config["PortBindings"] = {
                    f"{port}/tcp": [{"HostPort": str(host_port)}] for port, host_port in self.ports.items()
                }
            if self.extra_hosts:
                host_config["ExtraHosts"] = [f"{host}:{ip}" for host, ip in self.extra_hosts.items()]
            if self.devices:
                host_config["Devices"] = [parse_device(d) for d in self.devices]
            if self.dns:
                host_config["Dns"] = self.dns
            if self.group_add:
                host_config["GroupAdd"] = self.group_add

            config: dict[str, Any] = {
                "Image": self.image,
                "Cmd": ["sleep", "infinity"],
                "WorkingDir": self.workdir,
                "Env": [f"{k}={v}" for k, v in self.env.items()],
                "HostConfig": host_config,
            }
            if self.user:
                config["User"] = self.user
            if self.ports:
                config["ExposedPorts"] = {f"{port}/tcp": {} for port in self.ports}
            if self.platform:
                config["Platform"] = self.platform
            create_kwargs: dict[str, Any] = {"config": config}
            if self.name:
                create_kwargs["name"] = self.name
            try:
                self._container = await self._client.containers.create(**create_kwargs)
                await self._container.start()
            except Exception:  # create/start failures may originate in Docker, HTTP, or socket layers
                if self._container is not None:
                    with contextlib.suppress(Exception):
                        await self._container.delete(force=True)
                self._container = None
                self._verified_network_id = None
                await self._client.close()
                self._client = None
                raise
            logger.info("Started sandbox container (image=%s)", self.image)

        try:
            self._shells = await self._discover_shells()
        except BaseException:
            # Discovery is part of startup: no failure may leave its unpublished container running.
            try:
                await shield_until_done(self.__aexit__())
            except BaseException:  # cleanup failure must not replace the original startup failure
                logger.exception("Failed to clean up sandbox after startup failure")
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        was_attached = self._attached
        if self._container is not None:
            if self.remove and not was_attached:
                with contextlib.suppress(Exception):
                    await self._container.delete(force=True)
                logger.info("Removed sandbox container")
            else:
                logger.info("Kept sandbox container (attached=%r, remove=%r)", was_attached, self.remove)
            self._container = None
            self._attached = False
            self._verified_network_id = None
        if self._client is not None:
            if self.named_volumes and self.volumes_remove and not was_attached:
                for vol_name in self.named_volumes.values():
                    with contextlib.suppress(Exception):
                        vol = await self._client.volumes.get(vol_name)
                        await vol.delete()
                logger.info("Removed %d named volume(s)", len(self.named_volumes))
            await self._client.close()
            self._client = None
        self._shells = None

    @property
    def tools(self) -> list[Tool[Any]]:
        """Return fresh axio tools bound to this running sandbox."""
        if self._container is None or self._shells is None:
            raise RuntimeError("DockerSandbox must be used as an async context manager")
        return [
            self._make_shell_tool(),
            Tool(name="write_file", handler=write_file, context=self),
            Tool(name="read_file", handler=read_file, context=self),
            Tool(name="list_files", handler=list_files, context=self),
            Tool(name="run_python", handler=run_python, context=self),
            Tool(name="patch_file", handler=patch_file, context=self),
        ]

    @property
    def client(self) -> aiodocker.Docker | None:
        """Return the Docker client while the sandbox context is active."""
        return self._client

    @property
    def container(self) -> aiodocker.containers.DockerContainer | None:
        """Return the Docker container while the sandbox context is active."""
        return self._container

    @property
    def attached(self) -> bool:
        """Whether the sandbox reused a named container."""
        return self._attached

    @property
    def container_id(self) -> str:
        """Return the ID of the running container. Only valid inside `async with`."""
        if self._container is None:
            raise RuntimeError("DockerSandbox must be used as an async context manager")
        return str(self._container.id)

    @property
    def available_shells(self) -> tuple[str, ...]:
        """Shell names cached for the running container, in default preference order."""
        if self._shells is None:
            raise RuntimeError("DockerSandbox must be used as an async context manager")
        return tuple(item.name for item in self._shells)

    async def ensure_running(self) -> None:
        """Start the container when it exists but is not currently running."""
        if self._container is None:
            return
        info = await self._container.show()
        if not info.get("State", {}).get("Running", False):
            await self._container.start()

    def _make_shell_tool(self) -> Tool[Any]:
        assert self._shells is not None
        schema = build_tool_schema(shell)
        shell_schema = cast(dict[str, Any], schema["properties"]["shell"])
        shell_schema["description"] = f"Shell name to execute. {_shell_choices_text(self._shells)}"
        if self._shells:
            string_schema = cast(dict[str, Any], shell_schema["anyOf"][0])
            string_schema["enum"] = [item.name for item in self._shells]
        description = f"{shell.__doc__ or ''}\n{_shell_choices_text(self._shells)}"
        return Tool(
            name="shell",
            handler=shell,
            context=self,
            description=description,
            schema=MappingProxyType(schema),
        )

    async def _probe_shell(self, executable: str) -> bool:
        assert self._container is not None
        stream: Any | None = None
        try:
            exec_obj = await self._container.exec(
                cmd=[executable, "-c", "exit 0"],
                stdout=True,
                stderr=True,
                tty=False,
            )
            stream = exec_obj.start(detach=False)
            while await asyncio.wait_for(stream.read_out(), timeout=_SHELL_PROBE_TIMEOUT) is not None:
                pass
            info = await exec_obj.inspect()
            exit_code = info.get("ExitCode")
            return isinstance(exit_code, int) and exit_code == 0
        finally:
            if stream is not None:
                await stream.close()

    async def _discover_shells(self) -> tuple[_ShellExecutable, ...]:
        assert self._container is not None
        info = await self._container.show()
        config = info.get("Config") if isinstance(info, dict) else None
        env = config.get("Env") if isinstance(config, dict) else None
        search_path: str | None = None
        if isinstance(env, list):
            for item in env:
                if isinstance(item, str) and item.startswith("PATH="):
                    search_path = item.removeprefix("PATH=")
        if search_path is None:
            search_path = _DEFAULT_CONTAINER_PATH

        directories = search_path.split(":")
        available: list[_ShellExecutable] = []
        for name in _SHELL_PREFERENCE:
            for directory in directories:
                base = directory or self.workdir
                if not posixpath.isabs(base):
                    base = posixpath.join(self.workdir, base)
                candidate = posixpath.normpath(posixpath.join(base, name))
                if await self._probe_shell(candidate):
                    available.append(_ShellExecutable(name=name, path=candidate))
                    break
        return tuple(available)

    async def ensure_image(self) -> None:
        """Pull the image if it is not present locally."""
        assert self._client is not None
        try:
            await self._client.images.inspect(self.image)
            logger.debug("Image already present: %s", self.image)
        except aiodocker.exceptions.DockerError as exc:
            if exc.status != 404:
                raise
            if not self.pull_missing:
                raise ImageNotAvailableError(f"Docker image {self.image!r} is not available locally") from exc
            logger.info("Pulling image %s ...", self.image)
            await self._client.images.pull(self.image)
            logger.info("Image pulled: %s", self.image)

    async def _prepare_exec_command(self, command: str, stdin: str | None) -> str:
        assert self._container is not None
        if stdin is not None:
            stdin_path = f"/tmp/.axio_stdin_{uuid.uuid4().hex}"
            await self.write_file(stdin_path, stdin)
            # Wrap in a subshell so the redirect applies to the whole command,
            # not just the last simple command when the caller's command already
            # uses semicolons (e.g. RunPython's "; exit $_rc" suffix).
            return f"( {command} ) < {stdin_path}; _rc=$?; rm -f {stdin_path}; exit $_rc"
        return command

    async def exec_stream(
        self,
        command: str,
        timeout: float = 30,
        stdin: str | None = None,
        shell: str | None = None,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Yield tagged chunks in Docker's observed multiplex-frame order."""
        assert self._container is not None
        command = await self._prepare_exec_command(command, stdin)
        if self._shells is None:
            raise RuntimeError("DockerSandbox must be used as an async context manager")
        executable = _select_shell(shell, self._shells)

        exec_obj = await self._container.exec(
            cmd=[executable.path, "-c", command],
            stdout=True,
            stderr=True,
            tty=False,
        )
        stream = exec_obj.start(detach=False)
        decoders: dict[int, codecs.IncrementalDecoder] = {
            1: codecs.getincrementaldecoder("utf-8")(errors="replace"),
            2: codecs.getincrementaldecoder("utf-8")(errors="replace"),
        }
        timed_out = False
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                try:
                    msg = await asyncio.wait_for(stream.read_out(), timeout=max(0.0, remaining))
                except TimeoutError:
                    timed_out = True
                    break
                if msg is None:
                    break
                stream_id = 1 if msg.stream == 1 else 2
                text = decoders[stream_id].decode(msg.data)
                if text:
                    yield ("stdout" if stream_id == 1 else "stderr"), text
        finally:
            await stream.close()

        for stream_id, decoder in decoders.items():
            tail = decoder.decode(b"", final=True)
            if tail:
                yield ("stdout" if stream_id == 1 else "stderr"), tail

        if timed_out:
            yield "stderr", _ShellControl(f"[timeout after {timeout}s]")
            return

        info = await exec_obj.inspect()
        exit_code: int = info["ExitCode"]
        if exit_code != 0:
            yield "stderr", _ShellControl(f"[exit code: {exit_code}]")

    async def exec(
        self,
        command: str,
        timeout: float = 30,
        stdin: str | None = None,
        shell: str | None = None,
    ) -> str:
        """Execute a shell command inside the container and return its output."""
        records: list[_ShellRecord] = []
        async for key, text in self.exec_stream(command, timeout=timeout, stdin=stdin, shell=shell):
            records.append((0.0, key, text))
        return _format_shell_records(records)

    async def write_bytes(self, path: str, data: bytes, mode: int = 0o644) -> str:
        """Write raw bytes to a file inside the container."""
        assert self._container is not None
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:") as tar:
            info = tarfile.TarInfo(name=os.path.basename(path))
            info.size = len(data)
            info.mode = mode
            user_parts = self.user.split(":", maxsplit=1)
            if len(user_parts) == 2 and all(part.isdecimal() for part in user_parts):
                info.uid = int(user_parts[0])
                info.gid = int(user_parts[1])
            tar.addfile(info, io.BytesIO(data))
        parent = os.path.dirname(path) or "/"
        await self.exec(f"mkdir -p {shlex.quote(parent)}")
        await self._container.put_archive(
            path=parent,
            data=buf.getvalue(),
        )
        return f"Wrote {len(data)} bytes to {path}"

    async def write_file(self, path: str, content: str, mode: int = 0o644) -> str:
        """Write UTF-8 text to a file inside the container."""
        return await self.write_bytes(path, content.encode(), mode=mode)

    async def get_archive(self, path: str) -> tarfile.TarFile:
        """Fetch a path from the container as a TarFile object."""
        assert self._container is not None
        try:
            return cast(tarfile.TarFile, await self._container.get_archive(path=path))
        except DockerError as exc:
            # A missing path is an ordinary outcome for an agent exploring a
            # tree, so report it as such: the engine's phrasing carries a
            # container id the caller can do nothing with.
            if exc.status == 404:
                raise FileNotFoundError(f"No such file or directory: {path}") from exc
            raise

    async def read_file_bytes(self, path: str) -> bytes:
        """Read a file from inside the container and return raw bytes."""
        tar = await self.get_archive(path)
        member = tar.next()
        if member is None:
            return b""
        f = tar.extractfile(member)
        return f.read() if f else b""
