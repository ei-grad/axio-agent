"""Declarative global defaults and named agent bundles for axio-repl."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from axio.effort import EFFORT_LEVELS

SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 1024 * 1024
MAX_INSTRUCTIONS_BYTES = 1024 * 1024
AGENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,127})\Z")
ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

_SETTINGS_KEYS = frozenset({"transport", "model", "runtime", "sandbox", "tools"})
_TRANSPORT_KEYS = frozenset({"name", "base_url", "api_key_env"})
_RUNTIME_KEYS = frozenset(
    {
        "temperature",
        "effort",
        "max_tokens",
        "max_iterations",
        "debug",
        "agent_actions",
        "powerline",
        "session_log",
        "session_log_dir",
    }
)
_SANDBOX_KEYS = frozenset(
    {
        "backend",
        "image",
        "network",
        "memory",
        "cpus",
        "proxy",
        "no_proxy",
        "registries",
        "datasets",
        "ca_certificate",
    }
)
_REGISTRY_KEYS = frozenset({"pypi", "npm", "cargo", "go", "go_sumdb"})
_EFFORT_VALUES = frozenset({"default", *EFFORT_LEVELS})
_CLI_OPTION_DESTINATIONS = {
    "--transport": "transport",
    "--transport-base-url": "transport_base_url",
    "--transport-api-key-env": "transport_api_key_env",
    "--model": "model",
    "--temperature": "temperature",
    "--effort": "effort",
    "--max-tokens": "max_tokens",
    "--max-iterations": "max_iterations",
    "--debug": "debug",
    "--no-debug": "debug",
    "--agent-actions": "agent_actions",
    "--powerline": "powerline",
    "--no-powerline": "powerline",
    "--session-log": "no_session_log",
    "--no-session-log": "no_session_log",
    "--session-log-dir": "session_log_dir",
    "--sandbox": "sandbox",
    "--sandbox-image": "sandbox_image",
    "--sandbox-network": "sandbox_network",
    "--sandbox-memory": "sandbox_memory",
    "--sandbox-cpus": "sandbox_cpus",
    "--sandbox-proxy": "sandbox_proxy",
    "--sandbox-no-proxy": "sandbox_no_proxy",
    "--sandbox-pypi-index": "sandbox_pypi_index",
    "--sandbox-npm-registry": "sandbox_npm_registry",
    "--sandbox-cargo-index": "sandbox_cargo_index",
    "--sandbox-go-proxy": "sandbox_go_proxy",
    "--sandbox-go-sumdb": "sandbox_go_sumdb",
    "--sandbox-datasets": "sandbox_datasets",
    "--sandbox-ca-cert": "sandbox_ca_cert",
    "--tools": "tools",
}


class AgentConfigError(ValueError):
    """A configuration file or environment override is invalid."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    output: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in output
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class TransportSettings:
    name: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None

    def overlay(self, other: TransportSettings) -> TransportSettings:
        if other.name is not None and other.name != self.name:
            return other
        return TransportSettings(
            name=other.name if other.name is not None else self.name,
            base_url=other.base_url if other.base_url is not None else self.base_url,
            api_key_env=other.api_key_env if other.api_key_env is not None else self.api_key_env,
        )


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    temperature: float | None = None
    effort: str | None = None
    max_tokens: int | None = None
    max_iterations: int | None = None
    debug: bool | None = None
    agent_actions: str | None = None
    powerline: bool | None = None
    session_log: bool | None = None
    session_log_dir: Path | None = None

    def overlay(self, other: RuntimeSettings) -> RuntimeSettings:
        return RuntimeSettings(
            temperature=other.temperature if other.temperature is not None else self.temperature,
            effort=other.effort if other.effort is not None else self.effort,
            max_tokens=other.max_tokens if other.max_tokens is not None else self.max_tokens,
            max_iterations=other.max_iterations if other.max_iterations is not None else self.max_iterations,
            debug=other.debug if other.debug is not None else self.debug,
            agent_actions=other.agent_actions if other.agent_actions is not None else self.agent_actions,
            powerline=other.powerline if other.powerline is not None else self.powerline,
            session_log=other.session_log if other.session_log is not None else self.session_log,
            session_log_dir=(other.session_log_dir if other.session_log_dir is not None else self.session_log_dir),
        )


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    backend: str | None = None
    image: str | None = None
    network: str | None = None
    memory: str | None = None
    cpus: str | None = None
    proxy: str | None = None
    no_proxy: str | None = None
    pypi_index: str | None = None
    npm_registry: str | None = None
    cargo_index: str | None = None
    go_proxy: str | None = None
    go_sumdb: str | None = None
    datasets: Path | None = None
    ca_certificate: Path | None = None

    def overlay(self, other: SandboxSettings) -> SandboxSettings:
        return SandboxSettings(
            backend=other.backend if other.backend is not None else self.backend,
            image=other.image if other.image is not None else self.image,
            network=other.network if other.network is not None else self.network,
            memory=other.memory if other.memory is not None else self.memory,
            cpus=other.cpus if other.cpus is not None else self.cpus,
            proxy=other.proxy if other.proxy is not None else self.proxy,
            no_proxy=other.no_proxy if other.no_proxy is not None else self.no_proxy,
            pypi_index=other.pypi_index if other.pypi_index is not None else self.pypi_index,
            npm_registry=(other.npm_registry if other.npm_registry is not None else self.npm_registry),
            cargo_index=other.cargo_index if other.cargo_index is not None else self.cargo_index,
            go_proxy=other.go_proxy if other.go_proxy is not None else self.go_proxy,
            go_sumdb=other.go_sumdb if other.go_sumdb is not None else self.go_sumdb,
            datasets=other.datasets if other.datasets is not None else self.datasets,
            ca_certificate=(other.ca_certificate if other.ca_certificate is not None else self.ca_certificate),
        )


@dataclass(frozen=True, slots=True)
class ProfileSettings:
    transport: TransportSettings = TransportSettings()
    model: str | None = None
    runtime: RuntimeSettings = RuntimeSettings()
    sandbox: SandboxSettings = SandboxSettings()
    tools: tuple[str, ...] | None = None

    def overlay(self, other: ProfileSettings) -> ProfileSettings:
        return ProfileSettings(
            transport=self.transport.overlay(other.transport),
            model=other.model if other.model is not None else self.model,
            runtime=self.runtime.overlay(other.runtime),
            sandbox=self.sandbox.overlay(other.sandbox),
            tools=other.tools if other.tools is not None else self.tools,
        )


@dataclass(frozen=True, slots=True)
class ResolvedAgentProfile:
    config_dir: Path
    name: str | None
    description: str | None
    instructions: tuple[Path, ...]
    settings: ProfileSettings
    sources: tuple[Path, ...]

    def instructions_text(self) -> str:
        parts: list[str] = []
        remaining = MAX_INSTRUCTIONS_BYTES
        for instruction_file in self.instructions:
            try:
                with instruction_file.open("rb") as stream:
                    raw = stream.read(remaining + 1)
            except (OSError, UnicodeError) as exc:
                raise AgentConfigError(f"cannot read instructions file {instruction_file}: {exc}") from exc
            if len(raw) > remaining:
                raise AgentConfigError(f"agent instructions exceed the {MAX_INSTRUCTIONS_BYTES}-byte limit")
            try:
                parts.append(raw.decode("utf-8").strip())
            except UnicodeError as exc:
                raise AgentConfigError(f"cannot read instructions file {instruction_file}: {exc}") from exc
            remaining -= len(raw)
        return "\n\n".join(part for part in parts if part)


def list_agent_names(config_dir: Path) -> tuple[str, ...]:
    """List valid direct-child bundles without requiring a registry file."""

    agents_dir = config_dir.expanduser().resolve() / "agents"
    try:
        children = tuple(agents_dir.iterdir())
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise AgentConfigError(f"cannot list agent bundles in {agents_dir}: {exc}") from exc
    return tuple(
        sorted(
            child.name
            for child in children
            if child.is_dir() and AGENT_NAME_PATTERN.fullmatch(child.name) and (child / "agent.yaml").is_file()
        )
    )


def explicit_cli_destinations(argv: list[str]) -> frozenset[str]:
    """Return config-backed argparse destinations explicitly present in argv."""

    destinations: set[str] = set()
    for token in argv:
        if token == "--":
            break
        option = token.partition("=")[0]
        destination = _CLI_OPTION_DESTINATIONS.get(option)
        if destination is not None:
            destinations.add(destination)
    return frozenset(destinations)


def apply_profile_to_args(args: Any, profile: ResolvedAgentProfile, explicit: frozenset[str]) -> None:
    """Apply file and environment layers without replacing explicit CLI flags."""

    settings = profile.settings
    values: dict[str, object | None] = {
        "transport": settings.transport.name,
        "transport_base_url": settings.transport.base_url,
        "transport_api_key_env": settings.transport.api_key_env,
        "model": settings.model,
        "temperature": settings.runtime.temperature,
        "effort": settings.runtime.effort,
        "max_tokens": settings.runtime.max_tokens,
        "max_iterations": settings.runtime.max_iterations,
        "debug": settings.runtime.debug,
        "agent_actions": settings.runtime.agent_actions,
        "powerline": settings.runtime.powerline,
        "session_log_dir": settings.runtime.session_log_dir,
        "sandbox": settings.sandbox.backend,
        "sandbox_image": settings.sandbox.image,
        "sandbox_network": settings.sandbox.network,
        "sandbox_memory": settings.sandbox.memory,
        "sandbox_cpus": settings.sandbox.cpus,
        "sandbox_proxy": settings.sandbox.proxy,
        "sandbox_no_proxy": settings.sandbox.no_proxy,
        "sandbox_pypi_index": settings.sandbox.pypi_index,
        "sandbox_npm_registry": settings.sandbox.npm_registry,
        "sandbox_cargo_index": settings.sandbox.cargo_index,
        "sandbox_go_proxy": settings.sandbox.go_proxy,
        "sandbox_go_sumdb": settings.sandbox.go_sumdb,
        "sandbox_datasets": settings.sandbox.datasets,
        "sandbox_ca_cert": settings.sandbox.ca_certificate,
        "tools": settings.tools,
    }
    for destination, value in values.items():
        if value is not None and destination not in explicit:
            setattr(args, destination, value)
    if "transport" in explicit and args.transport != settings.transport.name:
        if "transport_base_url" not in explicit:
            args.transport_base_url = None
        if "transport_api_key_env" not in explicit:
            args.transport_api_key_env = None
    if settings.runtime.session_log is not None and "no_session_log" not in explicit:
        args.no_session_log = not settings.runtime.session_log


def default_config_dir(
    environ: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    configured = values.get("AXIO_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg = values.get("XDG_CONFIG_HOME")
    if xdg:
        xdg_path = Path(xdg).expanduser()
        if xdg_path.is_absolute():
            return xdg_path.resolve() / "axio"
    home_dir = Path.home() if home is None else home
    return home_dir.expanduser().resolve() / ".config" / "axio"


def resolve_agent_name(cli_name: str | None, environ: Mapping[str, str] | None = None) -> str | None:
    values = os.environ if environ is None else environ
    name = cli_name if cli_name is not None else values.get("AXIO_REPL_AGENT")
    if name is None:
        return None
    if not AGENT_NAME_PATTERN.fullmatch(name):
        raise AgentConfigError(
            "agent name must start with an ASCII letter or digit and contain only letters, digits, '.', '_', or '-'"
        )
    return name


def resolve_api_key(api_key_env: str | None, environ: Mapping[str, str] | None = None) -> str | None:
    """Resolve a transport secret without retaining it in the loaded profile."""

    if api_key_env is None:
        return None
    if not ENV_NAME_PATTERN.fullmatch(api_key_env):
        raise AgentConfigError("transport API key reference is not a valid environment variable name")
    values = os.environ if environ is None else environ
    secret = values.get(api_key_env)
    if not secret:
        raise AgentConfigError(f"environment variable {api_key_env} referenced by transport.api_key_env is not set")
    return secret


def load_agent_profile(
    config_dir: Path,
    agent_name: str | None,
    environ: Mapping[str, str] | None = None,
    *,
    cwd: Path | None = None,
) -> ResolvedAgentProfile:
    values = os.environ if environ is None else environ
    root = config_dir.expanduser().resolve()
    settings = ProfileSettings()
    sources: list[Path] = []

    global_file = root / "config.yaml"
    if global_file.exists():
        data = _load_yaml_mapping(global_file)
        _check_keys(data, {"version", "defaults"}, global_file, "root")
        _require_version(data, global_file)
        defaults = _mapping(data.get("defaults"), global_file, "defaults")
        _check_keys(defaults, _SETTINGS_KEYS, global_file, "defaults")
        settings = settings.overlay(_parse_settings(defaults, global_file, global_file.parent))
        sources.append(global_file)

    description: str | None = None
    instructions: tuple[Path, ...] = ()
    if agent_name is not None:
        if not AGENT_NAME_PATTERN.fullmatch(agent_name):
            raise AgentConfigError(f"invalid agent name {agent_name!r}")
        bundle_dir = root / "agents" / agent_name
        manifest_file = bundle_dir / "agent.yaml"
        if not manifest_file.is_file():
            raise AgentConfigError(f"agent {agent_name!r} does not exist: expected {manifest_file}")
        data = _load_yaml_mapping(manifest_file)
        _check_keys(
            data,
            {"version", "description", "instructions", *_SETTINGS_KEYS},
            manifest_file,
            "root",
        )
        _require_version(data, manifest_file)
        description = _optional_string(data, "description", manifest_file, "description")
        instructions = _parse_instruction_paths(data.get("instructions"), manifest_file, bundle_dir)
        settings = settings.overlay(_parse_settings(data, manifest_file, bundle_dir))
        sources.append(manifest_file)

    env_base = Path.cwd() if cwd is None else cwd
    settings = settings.overlay(_environment_settings(values, env_base.resolve()))
    return ResolvedAgentProfile(
        config_dir=root,
        name=agent_name,
        description=description,
        instructions=instructions,
        settings=settings,
        sources=tuple(sources),
    )


def _load_yaml_mapping(config_file: Path) -> dict[str, object]:
    try:
        stat = config_file.stat()
        if stat.st_size > MAX_CONFIG_BYTES:
            raise AgentConfigError(f"configuration file is larger than {MAX_CONFIG_BYTES} bytes: {config_file}")
        raw = config_file.read_text(encoding="utf-8")
        loaded: Any = yaml.load(raw, Loader=_UniqueKeyLoader)
    except AgentConfigError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise AgentConfigError(f"cannot load configuration file {config_file}: {exc}") from exc
    return _mapping(loaded, config_file, "root")


def _mapping(value: object, source: Path, location: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AgentConfigError(f"{source}: {location} must be a mapping")
    output: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise AgentConfigError(f"{source}: {location} keys must be strings")
        output[key] = item
    return output


def _check_keys(data: Mapping[str, object], allowed: set[str] | frozenset[str], source: Path, location: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise AgentConfigError(f"{source}: unknown {location} field(s): {', '.join(unknown)}")


def _require_version(data: Mapping[str, object], source: Path) -> None:
    version = data.get("version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise AgentConfigError(f"{source}: version must be the integer {SCHEMA_VERSION}")


def _parse_settings(data: Mapping[str, object], source: Path, base_dir: Path) -> ProfileSettings:
    settings_data = {key: value for key, value in data.items() if key in _SETTINGS_KEYS}
    _check_keys(settings_data, _SETTINGS_KEYS, source, "settings")
    transport = _parse_transport(_mapping(settings_data.get("transport"), source, "transport"), source)
    runtime = _parse_runtime(_mapping(settings_data.get("runtime"), source, "runtime"), source, base_dir)
    sandbox = _parse_sandbox(_mapping(settings_data.get("sandbox"), source, "sandbox"), source, base_dir)
    model = _optional_string(settings_data, "model", source, "model")
    tools = _optional_string_list(settings_data, "tools", source, "tools")
    if tools is not None and len(set(tools)) != len(tools):
        raise AgentConfigError(f"{source}: tools must not contain duplicate names")
    return ProfileSettings(transport=transport, model=model, runtime=runtime, sandbox=sandbox, tools=tools)


def _parse_transport(data: Mapping[str, object], source: Path) -> TransportSettings:
    _check_keys(data, _TRANSPORT_KEYS, source, "transport")
    name = _optional_string(data, "name", source, "transport.name")
    base_url = _optional_string(data, "base_url", source, "transport.base_url")
    api_key_env = _optional_string(data, "api_key_env", source, "transport.api_key_env")
    if base_url is not None:
        _validate_http_url(base_url, source, "transport.base_url")
    if api_key_env is not None and not ENV_NAME_PATTERN.fullmatch(api_key_env):
        raise AgentConfigError(f"{source}: transport.api_key_env is not a valid environment variable name")
    return TransportSettings(name=name, base_url=base_url, api_key_env=api_key_env)


def _parse_runtime(data: Mapping[str, object], source: Path, base_dir: Path) -> RuntimeSettings:
    _check_keys(data, _RUNTIME_KEYS, source, "runtime")
    temperature = _optional_float(data, "temperature", source, "runtime.temperature")
    if temperature is not None and not 0 <= temperature <= 2:
        raise AgentConfigError(f"{source}: runtime.temperature must be between 0 and 2")
    max_tokens = _optional_positive_int(data, "max_tokens", source, "runtime.max_tokens")
    max_iterations = _optional_positive_int(data, "max_iterations", source, "runtime.max_iterations")
    effort = _optional_string(data, "effort", source, "runtime.effort")
    if effort is not None and effort not in _EFFORT_VALUES:
        raise AgentConfigError(f"{source}: runtime.effort must be one of: {', '.join(sorted(_EFFORT_VALUES))}")
    agent_actions = _optional_string(data, "agent_actions", source, "runtime.agent_actions")
    if agent_actions is not None and agent_actions not in {"off", "on"}:
        raise AgentConfigError(f"{source}: runtime.agent_actions must be 'off' or 'on'")
    return RuntimeSettings(
        temperature=temperature,
        effort=effort,
        max_tokens=max_tokens,
        max_iterations=max_iterations,
        debug=_optional_bool(data, "debug", source, "runtime.debug"),
        agent_actions=agent_actions,
        powerline=_optional_bool(data, "powerline", source, "runtime.powerline"),
        session_log=_optional_bool(data, "session_log", source, "runtime.session_log"),
        session_log_dir=_optional_path(data, "session_log_dir", source, "runtime.session_log_dir", base_dir),
    )


def _parse_sandbox(data: Mapping[str, object], source: Path, base_dir: Path) -> SandboxSettings:
    _check_keys(data, _SANDBOX_KEYS, source, "sandbox")
    registries = _mapping(data.get("registries"), source, "sandbox.registries")
    _check_keys(registries, _REGISTRY_KEYS, source, "sandbox.registries")
    backend = _optional_string(data, "backend", source, "sandbox.backend")
    if backend is not None and backend not in {"auto", "docker", "none"}:
        raise AgentConfigError(f"{source}: sandbox.backend must be 'auto', 'docker', or 'none'")
    return SandboxSettings(
        backend=backend,
        image=_optional_string(data, "image", source, "sandbox.image"),
        network=_optional_string(data, "network", source, "sandbox.network"),
        memory=_optional_string(data, "memory", source, "sandbox.memory"),
        cpus=_optional_string(data, "cpus", source, "sandbox.cpus"),
        proxy=_optional_string(data, "proxy", source, "sandbox.proxy"),
        no_proxy=_optional_string(data, "no_proxy", source, "sandbox.no_proxy"),
        pypi_index=_optional_string(registries, "pypi", source, "sandbox.registries.pypi"),
        npm_registry=_optional_string(registries, "npm", source, "sandbox.registries.npm"),
        cargo_index=_optional_string(registries, "cargo", source, "sandbox.registries.cargo"),
        go_proxy=_optional_string(registries, "go", source, "sandbox.registries.go"),
        go_sumdb=_optional_string(registries, "go_sumdb", source, "sandbox.registries.go_sumdb"),
        datasets=_optional_path(data, "datasets", source, "sandbox.datasets", base_dir),
        ca_certificate=_optional_path(data, "ca_certificate", source, "sandbox.ca_certificate", base_dir),
    )


def _parse_instruction_paths(value: object, source: Path, bundle_dir: Path) -> tuple[Path, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise AgentConfigError(f"{source}: instructions must be a list of non-empty relative file paths")
    output: list[Path] = []
    for item in value:
        candidate = Path(item)
        if candidate.is_absolute():
            raise AgentConfigError(f"{source}: instructions path must be relative to the agent bundle: {item}")
        resolved = (bundle_dir / candidate).resolve()
        try:
            resolved.relative_to(bundle_dir.resolve())
        except ValueError as exc:
            raise AgentConfigError(f"{source}: instructions path escapes the agent bundle: {item}") from exc
        if not resolved.is_file():
            raise AgentConfigError(f"{source}: instructions file does not exist: {item}")
        output.append(resolved)
    if len(set(output)) != len(output):
        raise AgentConfigError(f"{source}: instructions must not contain duplicate paths")
    return tuple(output)


def _optional_string(data: Mapping[str, object], key: str, source: Path, location: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, str) or not value or value != value.strip():
        raise AgentConfigError(f"{source}: {location} must be a non-empty string without surrounding whitespace")
    return value


def _optional_string_list(data: Mapping[str, object], key: str, source: Path, location: str) -> tuple[str, ...] | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item or item != item.strip() for item in value
    ):
        raise AgentConfigError(f"{source}: {location} must be a list of non-empty strings")
    return tuple(value)


def _optional_bool(data: Mapping[str, object], key: str, source: Path, location: str) -> bool | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if type(value) is not bool:
        raise AgentConfigError(f"{source}: {location} must be a boolean")
    return value


def _optional_float(data: Mapping[str, object], key: str, source: Path, location: str) -> float | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentConfigError(f"{source}: {location} must be a number")
    return float(value)


def _optional_positive_int(data: Mapping[str, object], key: str, source: Path, location: str) -> int | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if type(value) is not int or value <= 0:
        raise AgentConfigError(f"{source}: {location} must be a positive integer")
    return value


def _optional_path(data: Mapping[str, object], key: str, source: Path, location: str, base_dir: Path) -> Path | None:
    value = _optional_string(data, key, source, location)
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    resolved = (base_dir / candidate).resolve()
    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError as exc:
        raise AgentConfigError(f"{source}: relative {location} escapes its configuration directory") from exc
    return resolved


def _validate_http_url(value: str, source: Path, location: str) -> None:
    if any(char.isspace() for char in value):
        raise AgentConfigError(f"{source}: {location} must not contain whitespace")
    parsed = urlsplit(value)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise AgentConfigError(f"{source}: {location} must contain a valid port") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise AgentConfigError(f"{source}: {location} must be an HTTP(S) URL with a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise AgentConfigError(f"{source}: {location} must not contain credentials")
    if parsed.query or parsed.fragment:
        raise AgentConfigError(f"{source}: {location} must not contain a query or fragment")


def _environment_settings(values: Mapping[str, str], cwd: Path) -> ProfileSettings:
    def text(name: str) -> str | None:
        value = values.get(name)
        if value is None:
            return None
        if not value or value != value.strip():
            raise AgentConfigError(f"environment variable {name} must be non-empty without surrounding whitespace")
        return value

    def positive_int(name: str) -> int | None:
        value = text(name)
        if value is None:
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise AgentConfigError(f"environment variable {name} must be a positive integer") from exc
        if parsed <= 0:
            raise AgentConfigError(f"environment variable {name} must be a positive integer")
        return parsed

    def number(name: str) -> float | None:
        value = text(name)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise AgentConfigError(f"environment variable {name} must be a number") from exc

    def boolean(name: str) -> bool | None:
        value = text(name)
        if value is None:
            return None
        normalized = value.lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise AgentConfigError(f"environment variable {name} must be a boolean")

    def external_path(name: str) -> Path | None:
        value = text(name)
        if value is None:
            return None
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else cwd / candidate).resolve()

    transport = TransportSettings(
        name=text("AXIO_REPL_TRANSPORT"),
        base_url=text("AXIO_REPL_TRANSPORT_BASE_URL"),
        api_key_env=text("AXIO_REPL_TRANSPORT_API_KEY_ENV"),
    )
    if transport.base_url is not None:
        _validate_http_url(transport.base_url, Path("<environment>"), "AXIO_REPL_TRANSPORT_BASE_URL")
    if transport.api_key_env is not None and not ENV_NAME_PATTERN.fullmatch(transport.api_key_env):
        raise AgentConfigError("AXIO_REPL_TRANSPORT_API_KEY_ENV is not a valid environment variable name")

    temperature = number("AXIO_REPL_TEMPERATURE")
    if temperature is not None and not 0 <= temperature <= 2:
        raise AgentConfigError("AXIO_REPL_TEMPERATURE must be between 0 and 2")
    agent_actions = text("AXIO_REPL_AGENT_ACTIONS")
    if agent_actions is not None and agent_actions not in {"off", "on"}:
        raise AgentConfigError("AXIO_REPL_AGENT_ACTIONS must be 'off' or 'on'")
    runtime = RuntimeSettings(
        temperature=temperature,
        effort=text("AXIO_REPL_EFFORT"),
        max_tokens=positive_int("AXIO_REPL_MAX_TOKENS"),
        max_iterations=positive_int("AXIO_REPL_MAX_ITERATIONS"),
        debug=boolean("AXIO_REPL_DEBUG"),
        agent_actions=agent_actions,
        powerline=boolean("AXIO_REPL_POWERLINE"),
        session_log=boolean("AXIO_REPL_SESSION_LOG"),
        session_log_dir=external_path("AXIO_REPL_SESSION_LOG_DIR"),
    )
    if runtime.effort is not None and runtime.effort not in _EFFORT_VALUES:
        raise AgentConfigError(f"AXIO_REPL_EFFORT must be one of: {', '.join(sorted(_EFFORT_VALUES))}")
    backend = text("AXIO_REPL_SANDBOX")
    if backend is not None and backend not in {"auto", "docker", "none"}:
        raise AgentConfigError("AXIO_REPL_SANDBOX must be 'auto', 'docker', or 'none'")
    sandbox = SandboxSettings(
        backend=backend,
        image=text("AXIO_REPL_SANDBOX_IMAGE"),
        network=text("AXIO_REPL_SANDBOX_NETWORK"),
        memory=text("AXIO_REPL_SANDBOX_MEMORY"),
        cpus=text("AXIO_REPL_SANDBOX_CPUS"),
        proxy=text("AXIO_REPL_SANDBOX_PROXY"),
        no_proxy=text("AXIO_REPL_SANDBOX_NO_PROXY"),
        pypi_index=text("AXIO_REPL_SANDBOX_PYPI_INDEX"),
        npm_registry=text("AXIO_REPL_SANDBOX_NPM_REGISTRY"),
        cargo_index=text("AXIO_REPL_SANDBOX_CARGO_INDEX"),
        go_proxy=text("AXIO_REPL_SANDBOX_GO_PROXY"),
        go_sumdb=text("AXIO_REPL_SANDBOX_GO_SUMDB"),
        datasets=external_path("AXIO_REPL_SANDBOX_DATASETS"),
        ca_certificate=external_path("AXIO_REPL_SANDBOX_CA_CERT"),
    )
    tools_value = text("AXIO_REPL_TOOLS")
    tools = None if tools_value is None else tuple(part.strip() for part in tools_value.split(","))
    if tools is not None and any(not tool for tool in tools):
        raise AgentConfigError("AXIO_REPL_TOOLS must be 'all', 'none', or a comma-separated list of tool names")
    if tools is not None and len(set(tools)) != len(tools):
        raise AgentConfigError("AXIO_REPL_TOOLS must not contain duplicate names")
    return ProfileSettings(
        transport=transport,
        model=text("AXIO_REPL_MODEL"),
        runtime=runtime,
        sandbox=sandbox,
        tools=tools,
    )
