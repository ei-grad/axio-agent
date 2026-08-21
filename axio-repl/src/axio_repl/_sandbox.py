"""Optional Docker isolation for the tools handed to the model.

``axio-tools-docker`` already implements every filesystem and execution tool
against a container; this module only decides when to use it and swaps the tools
by name. Coordination tools (spawn/stop/interrupt/peers) are left alone — they
touch no files. Spawned subagents inherit the parent's tools, so substituting
once covers the whole tree.

The container runs with networking disabled unless an operator selects a named,
internal Docker network. The model is contacted by axio-repl on the host, never
from inside the sandbox.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shlex
import shutil
import tempfile
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit

from axio import Tool

from axio_repl import _search

logger = logging.getLogger(__name__)

# Replaced by the container-backed implementations of the same name.
SANDBOXED_TOOL_NAMES = frozenset({"read_file", "write_file", "patch_file", "list_files", "shell"})

# Offered only inside the sandbox: running arbitrary Python on the host is
# exactly what this is meant to avoid.
SANDBOX_ONLY_TOOL_NAMES = frozenset({"run_python"})

DEFAULT_SANDBOX_IMAGE = "axio-agent-sandbox:standard"
SANDBOX_HOME = "/tmp/axio-home"
DATASETS_DIR = "/datasets"
EGRESS_CA_PATH = "/etc/axio/egress-ca.pem"
CARGO_HOME = f"{SANDBOX_HOME}/.cargo"
CARGO_CONFIG_PATH = f"{CARGO_HOME}/config.toml"
DEFAULT_SANDBOX_MEMORY = "256m"
DEFAULT_SANDBOX_CPUS = "1.0"
HOST_IDENTITY_ESCAPE_HATCH = "Pass `--sandbox none` to explicitly run tools on the host."

_MEMORY_MULTIPLIERS = {"k": 1024, "m": 1024**2, "g": 1024**3}

_ROUTED_DOCKER_NETWORKS = frozenset({"bridge", "default", "host", "none"})


def _parse_memory(value: str) -> int:
    normalized = value.lower().strip()
    suffix = normalized[-1:] if normalized[-1:] in _MEMORY_MULTIPLIERS else ""
    amount = normalized[: -len(suffix)] if suffix else normalized
    try:
        parsed = int(amount)
    except ValueError as exc:
        raise ValueError("--sandbox-memory must be a positive integer with an optional k, m, or g suffix") from exc
    if parsed <= 0:
        raise ValueError("--sandbox-memory must be greater than zero")
    return parsed * _MEMORY_MULTIPLIERS.get(suffix, 1)


def _parse_http_endpoint(option: str, value: str) -> SplitResult:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{option} must be a valid HTTP or HTTPS URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{option} must be an HTTP or HTTPS URL without credentials, query, or fragment")
    return parsed


@dataclass(frozen=True, slots=True)
class SandboxOptions:
    """Operator-controlled Docker resources and restricted network endpoints."""

    network: str | None = None
    memory: str = DEFAULT_SANDBOX_MEMORY
    cpus: str = DEFAULT_SANDBOX_CPUS
    proxy: str | None = None
    no_proxy: str | None = None
    pypi_index: str | None = None
    npm_registry: str | None = None
    cargo_index: str | None = None
    go_proxy: str | None = None
    go_sumdb: str | None = None
    datasets: Path | None = None
    ca_certificate: Path | None = None

    def __post_init__(self) -> None:
        _parse_memory(self.memory)
        try:
            cpus = float(self.cpus)
        except ValueError as exc:
            raise ValueError("--sandbox-cpus must be a positive number") from exc
        if not math.isfinite(cpus) or cpus <= 0:
            raise ValueError("--sandbox-cpus must be greater than zero")
        if self.network is not None and (not self.network or self.network != self.network.strip()):
            raise ValueError("--sandbox-network must be a non-empty Docker network name without surrounding spaces")
        if self.network in _ROUTED_DOCKER_NETWORKS or (self.network or "").startswith("container:"):
            raise ValueError("--sandbox-network must name a user-defined internal Docker network")
        endpoints = (
            self.proxy,
            self.no_proxy,
            self.pypi_index,
            self.npm_registry,
            self.cargo_index,
            self.go_proxy,
            self.go_sumdb,
            self.ca_certificate,
        )
        if any(endpoints) and self.network is None:
            raise ValueError("sandbox proxy, registry, and CA settings require --sandbox-network")
        if self.datasets is not None and not self.datasets.is_dir():
            raise ValueError(f"sandbox dataset directory does not exist or is not a directory: {self.datasets}")
        if self.ca_certificate is not None and not self.ca_certificate.is_file():
            raise ValueError(f"sandbox CA certificate does not exist or is not a file: {self.ca_certificate}")
        if self.pypi_index:
            _parse_http_endpoint("--sandbox-pypi-index", self.pypi_index)
        if self.cargo_index:
            cargo_endpoint = self.cargo_index.removeprefix("sparse+")
            _parse_http_endpoint("--sandbox-cargo-index", cargo_endpoint)
            if self.cargo_index.startswith("sparse+") and not self.cargo_index.endswith("/"):
                raise ValueError("--sandbox-cargo-index sparse registry URLs must end with a slash")

    @property
    def requires_docker(self) -> bool:
        return (
            self.network is not None
            or self.datasets is not None
            or self.ca_certificate is not None
            or _parse_memory(self.memory) != _parse_memory(DEFAULT_SANDBOX_MEMORY)
            or float(self.cpus) != float(DEFAULT_SANDBOX_CPUS)
        )

    def environment(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if self.proxy:
            env.update(
                {
                    "HTTP_PROXY": self.proxy,
                    "HTTPS_PROXY": self.proxy,
                    "http_proxy": self.proxy,
                    "https_proxy": self.proxy,
                }
            )
        if self.no_proxy:
            env.update({"NO_PROXY": self.no_proxy, "no_proxy": self.no_proxy})
        if self.pypi_index:
            env.update({"UV_DEFAULT_INDEX": self.pypi_index, "PIP_INDEX_URL": self.pypi_index})
            parsed_index = _parse_http_endpoint("--sandbox-pypi-index", self.pypi_index)
            if parsed_index.scheme == "http":
                env.update({"UV_INSECURE_HOST": parsed_index.netloc, "PIP_TRUSTED_HOST": parsed_index.netloc})
        if self.npm_registry:
            env["NPM_CONFIG_REGISTRY"] = self.npm_registry
        if self.cargo_index:
            env["CARGO_HOME"] = CARGO_HOME
        if self.go_proxy:
            env["GOPROXY"] = self.go_proxy
        if self.go_sumdb:
            env["GOSUMDB"] = self.go_sumdb
        if self.ca_certificate:
            env.update(
                {
                    "SSL_CERT_FILE": EGRESS_CA_PATH,
                    "REQUESTS_CA_BUNDLE": EGRESS_CA_PATH,
                    "CURL_CA_BUNDLE": EGRESS_CA_PATH,
                    "GIT_SSL_CAINFO": EGRESS_CA_PATH,
                    "NODE_EXTRA_CA_CERTS": EGRESS_CA_PATH,
                    "CARGO_HTTP_CAINFO": EGRESS_CA_PATH,
                }
            )
        return env

    def cargo_config(self) -> str | None:
        if not self.cargo_index:
            return None
        registry = json.dumps(self.cargo_index)
        return f'[source.crates-io]\nreplace-with = "axio-mirror"\n\n[source.axio-mirror]\nregistry = {registry}\n'

    def read_only_volumes(self) -> dict[str, str]:
        volumes: dict[str, str] = {}
        if self.datasets:
            volumes[DATASETS_DIR] = str(self.datasets)
        if self.ca_certificate:
            volumes[EGRESS_CA_PATH] = str(self.ca_certificate)
        return volumes


@dataclass(frozen=True, slots=True)
class HostIdentity:
    """POSIX identity projected into a REPL-created sandbox."""

    uid: int
    gid: int
    supplementary_gids: tuple[int, ...]


def _current_host_identity() -> HostIdentity:
    if os.name != "posix" or not all(hasattr(os, name) for name in ("getuid", "getgid", "getgroups")):
        raise RuntimeError(
            f"Docker sandbox host-user projection requires a POSIX client. {HOST_IDENTITY_ESCAPE_HATCH}"
        )
    if not Path("/etc/passwd").is_file() or not Path("/etc/group").is_file():
        raise RuntimeError(
            f"Docker sandbox host-user projection requires /etc/passwd and /etc/group. {HOST_IDENTITY_ESCAPE_HATCH}"
        )

    uid = os.getuid()
    gid = os.getgid()
    supplementary_gids = tuple(dict.fromkeys(group_id for group_id in os.getgroups() if group_id != gid))
    return HostIdentity(uid=uid, gid=gid, supplementary_gids=supplementary_gids)


async def _verify_runtime_identity(sandbox: Any, workspace: Path, identity: HostIdentity) -> None:
    supplementary_checks = "\n".join(
        (
            f'case " $(id -G) " in *" {group_id} "*) ;; '
            f'*) fail "supplementary group {group_id} was not preserved" ;; esac'
        )
        for group_id in identity.supplementary_gids
    )
    script = f"""
fail() {{ printf 'AXIO_RUNTIME_ERROR: %s\\n' "$1"; exit 1; }}
[ "$(id -u)" = {identity.uid} ] || fail 'container UID differs from the invoking user'
[ "$(id -g)" = {identity.gid} ] || fail 'container primary GID differs from the invoking user'
id -un >/dev/null 2>&1 || fail 'current UID is not resolvable through the mounted /etc/passwd'
id -gn >/dev/null 2>&1 || fail 'current GID is not resolvable through the mounted /etc/group'
command -v getent >/dev/null 2>&1 || fail 'the sandbox image does not provide getent'
getent passwd {identity.uid} >/dev/null || fail 'current UID is absent from the mounted /etc/passwd'
getent group {identity.gid} >/dev/null || fail 'current GID is absent from the mounted /etc/group'
[ "$(pwd -P)" = {shlex.quote(str(workspace))} ] || fail 'container workdir differs from the host project path'
[ "$HOME" = {shlex.quote(SANDBOX_HOME)} ] || fail 'sandbox HOME is not isolated'
[ -w "$HOME" ] || fail 'sandbox HOME is not writable by the invoking user'
[ -w . ] || fail 'project directory is not writable by the invoking user'
{supplementary_checks}
awk '$5 == "/etc/passwd" && $6 ~ /(^|,)ro(,|$)/ {{ found=1 }} END {{ exit !found }}' /proc/self/mountinfo \
    || fail '/etc/passwd is not a read-only mount'
awk '$5 == "/etc/group" && $6 ~ /(^|,)ro(,|$)/ {{ found=1 }} END {{ exit !found }}' /proc/self/mountinfo \
    || fail '/etc/group is not a read-only mount'
printf 'AXIO_RUNTIME_OK\\n'
"""
    result = await sandbox.exec(script, timeout=15)
    if result != "AXIO_RUNTIME_OK":
        raise RuntimeError(f"Docker sandbox identity verification failed: {result}. {HOST_IDENTITY_ESCAPE_HATCH}")


# ast-grep installs its binary as `ast-grep`. `sg` is shadow-utils' setgid
# helper on Linux and must never be invoked in its place.
AST_GREP = "ast-grep"

_AST_GREP_DOC = """Structural search: match a code *pattern* by syntax rather than text.
    Patterns are written as ordinary code with `$VAR` metavariables, e.g.
    `$A == $A` or `def $NAME($$$ARGS)`. Prefer this over search_files when
    looking for a shape of code rather than a literal string."""

# What an agent reaches for first, and what a slim image is least likely to have.
PROBED_COMMANDS = (
    "python3",
    "python-data",
    "pip",
    "uv",
    "node",
    "npm",
    "go",
    "rustc",
    "cargo",
    "java",
    "git",
    "gh",
    "glab",
    "kaggle",
    "hf",
    "make",
    "gcc",
    "curl",
    "wget",
    "patch",
    "rg",
    AST_GREP,
    "jq",
    "xxd",
    "pdftotext",
    "qpdf",
    "tesseract",
    "grep",
    "sed",
    "awk",
    "find",
    "tar",
    "diff",
)


def docker_available() -> bool:
    """True when the bindings are installed and a daemon looks reachable."""
    try:
        import aiodocker  # noqa: F401
    except ImportError:
        return False
    return Path("/var/run/docker.sock").exists()


def ast_grep_available() -> bool:
    return shutil.which(AST_GREP) is not None


def _ast_grep_argv(pattern: str, path: str, lang: str | None) -> list[str]:
    argv = [AST_GREP, "run", "--pattern", pattern, "--heading", "never"]
    if lang:
        argv += ["--lang", lang]
    argv.append(path)
    return argv


def _truncate(text: str, max_results: int) -> str:
    lines = text.splitlines()
    if not lines:
        return "No matches"
    if len(lines) <= max_results:
        return text
    return "\n".join(lines[:max_results]) + f"\n[truncated at {max_results} lines]"


def _make_host_ast_grep() -> Any:
    async def ast_grep(pattern: str, path: str = ".", lang: str | None = None, max_results: int = 100) -> str:
        proc = await asyncio.create_subprocess_exec(
            *_ast_grep_argv(pattern, path, lang),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return _truncate(out.decode("utf-8", "replace").strip(), max_results)

    ast_grep.__doc__ = _AST_GREP_DOC
    return ast_grep


def _make_sandbox_overrides(sandbox: Any) -> list[Tool[Any]]:
    """search_files and ast_grep re-expressed against the container.

    axio-tools-docker has no counterpart for either, and left alone they would
    keep reading the host filesystem — the one thing the sandbox exists to stop.
    """

    async def search_files(query: str, path: str = ".", regex: bool = False, max_results: int = 100) -> str:
        script = "/tmp/.axio_search.py"
        await sandbox.write_file(script, Path(_search.__file__).read_text(encoding="utf-8"))
        params = json.dumps({"query": query, "path": path, "regex": regex, "max_results": max_results})
        out: str = await sandbox.exec(f"python3 {script}", stdin=params)
        return out

    async def ast_grep(pattern: str, path: str = ".", lang: str | None = None, max_results: int = 100) -> str:
        cmd = shlex.join(_ast_grep_argv(pattern, path, lang))
        return _truncate(await sandbox.exec(cmd), max_results)

    search_files.__doc__ = _search.search.__doc__
    ast_grep.__doc__ = _AST_GREP_DOC
    return [Tool(name="search_files", handler=search_files), Tool(name="ast_grep", handler=ast_grep)]


async def describe_environment(sandbox: Any, image: str, networking: bool) -> str:
    """What the image has and what it lacks, asked of the image itself.

    Explicit image overrides may contain a very different toolset from the
    standard image. Probe rather than assume so the system prompt describes the
    container the caller actually selected.
    """
    probe = "; ".join(f"command -v {name} >/dev/null 2>&1 && echo {name}" for name in PROBED_COMMANDS)
    try:
        output = await sandbox.exec(f"{probe}; exit 0", timeout=15)
    except Exception:  # a sandbox that cannot answer must not stop the session
        logger.warning("Could not probe the sandbox image", exc_info=True)
        return ""

    found = {line.strip() for line in output.splitlines()}
    present = [name for name in PROBED_COMMANDS if name in found]
    missing = [name for name in PROBED_COMMANDS if name not in found]
    if not present:
        return ""

    network_state = "restricted to an internal Docker network" if networking else "off"
    lines = [f"Sandbox: docker — image {image}, networking {network_state}.", ""]
    lines.append(f"Available: {', '.join(present)}")
    if missing:
        lines.append(f"Not installed: {', '.join(missing)}")
        if not networking:
            lines.append(
                "Networking is off, so none of the missing ones can be added: uv add/sync, npm install, "
                "cargo add, apt-get and git clone all fail to resolve a host. Solve the task with what is "
                "listed, and say so plainly when it cannot be done rather than retrying installs."
            )
        if AST_GREP in missing:
            lines.append("The ast_grep tool needs the ast-grep binary, which this image lacks — use search_files.")
    return "\n".join(lines)


async def build_tools(
    stack: AsyncExitStack,
    tools: list[Tool[Any]],
    mode: str,
    image: str,
    workspace: Path,
    options: SandboxOptions | None = None,
) -> tuple[list[Tool[Any]], str, Path, str]:
    """Return the toolset, a one-line description, the root the tools see, and
    what the sandbox environment offers.

    REPL-created containers preserve the project's absolute host path so tool
    output and system-prompt paths stay valid on both sides of the bind mount.
    """
    options = options or SandboxOptions()
    if mode == "none" or (mode == "auto" and not docker_available()):
        if options.requires_docker:
            raise RuntimeError("restricted sandbox settings require Docker, but sandbox execution is unavailable")
        result = [t for t in tools if t.name not in SANDBOX_ONLY_TOOL_NAMES]
        if ast_grep_available():
            result.append(Tool(name="ast_grep", handler=_make_host_ast_grep()))
        return result, "host — tools run directly on this machine", workspace, ""

    from axio_tools_docker.sandbox import DockerSandbox, ImageNotAvailableError

    try:
        workspace = workspace.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Docker sandbox project directory is unavailable: {workspace}") from exc
    if not workspace.is_dir():
        raise RuntimeError(f"Docker sandbox project path is not a directory: {workspace}")
    if workspace == Path("/"):
        raise RuntimeError("Docker sandbox refuses to mount the host filesystem root as a project")
    if ":" in str(workspace):
        raise RuntimeError(
            f"Docker sandbox project path contains ':' and cannot be encoded as a bind mount: {workspace}"
        )
    if str(workspace) == SANDBOX_HOME:
        raise RuntimeError(f"Docker sandbox project path conflicts with its isolated HOME: {workspace}")

    identity = _current_host_identity()
    network: bool | str = options.network if options.network is not None else False
    read_only_volumes = options.read_only_volumes()
    read_only_volumes.update({"/etc/passwd": "/etc/passwd", "/etc/group": "/etc/group"})
    home_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="axio-sandbox-home-")))
    (home_dir / ".cargo").mkdir()
    cargo_config = options.cargo_config()
    if cargo_config:
        cargo_config_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="axio-cargo-config-")))
        cargo_config_file = cargo_config_dir / "config.toml"
        cargo_config_file.write_text(cargo_config, encoding="utf-8")
        read_only_volumes[CARGO_CONFIG_PATH] = str(cargo_config_file)
    environment = options.environment()
    environment["HOME"] = SANDBOX_HOME
    try:
        sandbox = await stack.enter_async_context(
            DockerSandbox(
                image=image,
                memory=options.memory,
                cpus=options.cpus,
                volumes={str(workspace): str(workspace), SANDBOX_HOME: str(home_dir)},
                read_only_volumes=read_only_volumes,
                workdir=str(workspace),
                network=network,
                env=environment,
                user=f"{identity.uid}:{identity.gid}",
                group_add=[str(group_id) for group_id in identity.supplementary_gids],
                require_internal_network=options.network is not None,
                pull_missing=image != DEFAULT_SANDBOX_IMAGE,
            )
        )
    except ImageNotAvailableError as exc:
        if image == DEFAULT_SANDBOX_IMAGE:
            raise RuntimeError(
                f"Default sandbox image {DEFAULT_SANDBOX_IMAGE!r} is not built; run `make sandbox-image`"
            ) from exc
        raise
    await _verify_runtime_identity(sandbox, workspace, identity)
    available = {t.name: t for t in sandbox.tools}
    overrides = {t.name: t for t in _make_sandbox_overrides(sandbox)}

    merged: list[Tool[Any]] = []
    for tool in tools:
        if tool.name in overrides:
            merged.append(overrides.pop(tool.name))
        else:
            merged.append(available.get(tool.name, tool))
    merged.extend(overrides.values())
    for name in SANDBOX_ONLY_TOOL_NAMES & available.keys():
        merged.append(available[name])
    networking = options.network is not None
    note = await describe_environment(sandbox, image, networking=networking)
    network_description = f"internal network {options.network}" if options.network else "no network"
    return (
        merged,
        f"docker — {image}, {network_description}, project at {workspace}",
        workspace,
        note,
    )
