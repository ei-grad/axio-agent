from __future__ import annotations

from pathlib import Path

import pytest

from axio_repl._agent_config import (
    MAX_MODEL_CONTEXT_BYTES,
    AgentConfigError,
    apply_profile_to_args,
    default_config_dir,
    explicit_cli_destinations,
    list_agent_names,
    load_agent_profile,
    resolve_agent_name,
    resolve_api_key,
)


def write_config(config_file: Path, content: str) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(content, encoding="utf-8")


def test_default_config_dir_uses_xdg_and_ignores_relative_xdg(tmp_path: Path) -> None:
    assert default_config_dir({"XDG_CONFIG_HOME": str(tmp_path)}, home=Path("/home/test")) == tmp_path / "axio"
    assert default_config_dir({"XDG_CONFIG_HOME": "relative"}, home=tmp_path) == tmp_path / ".config" / "axio"


def test_default_config_dir_explicit_override_wins(tmp_path: Path) -> None:
    custom = tmp_path / "custom"

    assert default_config_dir({"AXIO_CONFIG_DIR": str(custom), "XDG_CONFIG_HOME": "/ignored"}) == custom


def test_load_profile_merges_global_agent_and_environment(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    write_config(
        config_dir / "config.yaml",
        """\
version: 1
defaults:
  transport:
    name: openai
    base_url: https://global.example/v1
  model: global-model
  runtime:
    max_iterations: 25
    debug: false
  sandbox:
    backend: auto
    memory: 1g
    registries:
      npm: https://npm.global.example/
  tools: [read_file, shell]
""",
    )
    bundle = config_dir / "agents" / "local"
    write_config(bundle / "prompts" / "system.md", "Use the local model.\n")
    write_config(bundle / "certs" / "ca.pem", "certificate\n")
    write_config(
        bundle / "agent.yaml",
        """\
version: 1
description: Local coding agent
instructions:
  - prompts/system.md
transport:
  name: llama-cpp
  base_url: http://127.0.0.1:18080/v1
model: local-model
runtime:
  max_tokens: 4096
sandbox:
  backend: docker
  network: axio-agent-egress
  registries:
    pypi: http://devpi:3141/root/pypi/+simple/
  ca_certificate: certs/ca.pem
""",
    )

    profile = load_agent_profile(
        config_dir,
        "local",
        {
            "AXIO_REPL_MODEL": "env-model",
            "AXIO_REPL_DEBUG": "true",
            "AXIO_REPL_SANDBOX_MEMORY": "4g",
        },
        cwd=tmp_path,
    )

    assert profile.name == "local"
    assert profile.description == "Local coding agent"
    assert profile.instructions_text() == "Use the local model."
    assert profile.settings.transport.name == "llama-cpp"
    assert profile.settings.transport.base_url == "http://127.0.0.1:18080/v1"
    assert profile.settings.model == "env-model"
    assert profile.settings.runtime.max_iterations == 25
    assert profile.settings.runtime.max_tokens == 4096
    assert profile.settings.runtime.debug is True
    assert profile.settings.sandbox.backend == "docker"
    assert profile.settings.sandbox.network == "axio-agent-egress"
    assert profile.settings.sandbox.memory == "4g"
    assert profile.settings.sandbox.npm_registry == "https://npm.global.example/"
    assert profile.settings.sandbox.pypi_index == "http://devpi:3141/root/pypi/+simple/"
    assert profile.settings.sandbox.ca_certificate == (bundle / "certs" / "ca.pem").resolve()
    assert profile.settings.tools == ("read_file", "shell")
    assert profile.sources == (config_dir / "config.yaml", bundle / "agent.yaml")


def test_powerline_is_resolved_from_yaml_environment_and_cli(tmp_path: Path) -> None:
    from argparse import Namespace

    write_config(
        tmp_path / "config.yaml",
        "version: 1\ndefaults:\n  runtime:\n    powerline: true\n",
    )

    profile = load_agent_profile(tmp_path, None, {})
    assert profile.settings.runtime.powerline is True
    assert load_agent_profile(tmp_path, None, {"AXIO_REPL_POWERLINE": "false"}).settings.runtime.powerline is False

    args = Namespace(powerline=None, no_session_log=False)
    apply_profile_to_args(args, profile, explicit_cli_destinations([]))
    assert args.powerline is True

    args = Namespace(powerline=False, no_session_log=False)
    apply_profile_to_args(args, profile, explicit_cli_destinations(["--no-powerline"]))
    assert args.powerline is False

    disabled_profile = load_agent_profile(tmp_path, None, {"AXIO_REPL_POWERLINE": "false"})
    args = Namespace(powerline=None, no_session_log=False)
    apply_profile_to_args(args, disabled_profile, explicit_cli_destinations([]))
    assert args.powerline is False

    args = Namespace(powerline=True, no_session_log=False)
    apply_profile_to_args(args, disabled_profile, explicit_cli_destinations(["--powerline"]))
    assert args.powerline is True


def test_session_replay_is_opt_in_and_obeys_config_environment_and_cli_precedence(tmp_path: Path) -> None:
    from argparse import Namespace

    write_config(
        tmp_path / "config.yaml",
        "version: 1\ndefaults:\n  runtime:\n    session_replay: true\n",
    )
    profile = load_agent_profile(tmp_path, None, {})
    assert profile.settings.runtime.session_replay is True
    environment = load_agent_profile(tmp_path, None, {"AXIO_REPL_SESSION_REPLAY": "false"})
    assert environment.settings.runtime.session_replay is False

    args = Namespace(session_replay=False, no_session_log=False)
    apply_profile_to_args(args, profile, explicit_cli_destinations([]))
    assert args.session_replay is True

    args = Namespace(session_replay=False, no_session_log=False)
    apply_profile_to_args(args, profile, explicit_cli_destinations(["--no-session-replay"]))
    assert args.session_replay is False


def test_session_replay_config_environment_and_cli_surface_is_documented() -> None:
    repository = Path(__file__).resolve().parents[2]
    guide = (repository / "docs" / "guides" / "axio-repl.md").read_text(encoding="utf-8")

    assert "session_replay: false" in guide
    assert "AXIO_REPL_SESSION_REPLAY" in guide
    assert "`--session-replay`" in guide


def test_theme_uses_global_agent_environment_and_explicit_cli_precedence(tmp_path: Path) -> None:
    from argparse import Namespace

    write_config(
        tmp_path / "config.yaml",
        "version: 1\ndefaults:\n  runtime:\n    theme: default\n",
    )
    write_config(
        tmp_path / "agents" / "local" / "agent.yaml",
        "version: 1\nruntime:\n  theme: monochrome\n",
    )

    agent_profile = load_agent_profile(tmp_path, "local", {})
    assert agent_profile.settings.runtime.theme == "monochrome"

    environment_profile = load_agent_profile(tmp_path, "local", {"AXIO_REPL_THEME": "default"})
    assert environment_profile.settings.runtime.theme == "default"

    args = Namespace(theme="monochrome", no_session_log=False)
    apply_profile_to_args(args, environment_profile, explicit_cli_destinations(["--theme=monochrome"]))
    assert args.theme == "monochrome"
    assert explicit_cli_destinations(["--theme=monochrome"]) == frozenset({"theme"})


@pytest.mark.parametrize(
    "content",
    (
        "version: 1\ndefaults:\n  runtime:\n    theme: unknown\n",
        "version: 1\ndefaults:\n  runtime:\n    theme: 7\n",
    ),
)
def test_theme_rejects_invalid_yaml_values(tmp_path: Path, content: str) -> None:
    write_config(tmp_path / "config.yaml", content)

    with pytest.raises(AgentConfigError, match="runtime.theme"):
        load_agent_profile(tmp_path, None, {})


def test_theme_rejects_invalid_environment_value(tmp_path: Path) -> None:
    with pytest.raises(AgentConfigError, match="AXIO_REPL_THEME must be one of"):
        load_agent_profile(tmp_path, None, {"AXIO_REPL_THEME": "unknown"})


def test_selected_agent_manifest_exposes_trusted_model_context(tmp_path: Path) -> None:
    write_config(
        tmp_path / "agents" / "local" / "agent.yaml",
        """\
version: 1
description: Catalog text only
model_context: |-
  Network access is routed through the configured policy proxy.
  Treat denied requests as policy outcomes.
""",
    )

    profile = load_agent_profile(tmp_path, "local", {})

    assert profile.description == "Catalog text only"
    assert profile.model_context == (
        "Network access is routed through the configured policy proxy.\nTreat denied requests as policy outcomes."
    )


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ([], "model_context must be a non-empty string"),
        ("bad\x1bcontext", "forbidden control character"),
        ("bad\x7fcontext", "forbidden control character"),
        ("x" * (MAX_MODEL_CONTEXT_BYTES + 1), "byte limit"),
    ),
)
def test_model_context_rejects_invalid_values(tmp_path: Path, value: object, message: str) -> None:
    import yaml

    write_config(
        tmp_path / "agents" / "local" / "agent.yaml",
        yaml.safe_dump({"version": 1, "model_context": value}, sort_keys=False),
    )

    with pytest.raises(AgentConfigError, match=message):
        load_agent_profile(tmp_path, "local", {})


def test_global_defaults_load_without_named_agent(tmp_path: Path) -> None:
    write_config(
        tmp_path / "config.yaml",
        """\
version: 1
defaults:
  sandbox:
    backend: none
""",
    )

    profile = load_agent_profile(tmp_path, None, {})

    assert profile.name is None
    assert profile.settings.sandbox.backend == "none"
    assert profile.sources == (tmp_path / "config.yaml",)


def test_global_defaults_reject_unknown_fields(tmp_path: Path) -> None:
    write_config(tmp_path / "config.yaml", "version: 1\ndefaults:\n  typo: true\n")

    with pytest.raises(AgentConfigError, match="unknown defaults field"):
        load_agent_profile(tmp_path, None, {})


def test_list_agent_names_uses_valid_direct_child_bundles(tmp_path: Path) -> None:
    write_config(tmp_path / "agents" / "zeta" / "agent.yaml", "version: 1\n")
    write_config(tmp_path / "agents" / "alpha" / "agent.yaml", "version: 1\n")
    write_config(tmp_path / "agents" / "missing-manifest" / "notes.md", "not an agent\n")

    assert list_agent_names(tmp_path) == ("alpha", "zeta")
    assert list_agent_names(tmp_path / "absent") == ()


def test_explicit_cli_destinations_accept_equals_and_stop_at_separator() -> None:
    assert explicit_cli_destinations(
        ["--model=cli", "--no-debug", "--session-log", "--session-replay", "--", "--sandbox", "docker"]
    ) == frozenset({"model", "debug", "no_session_log", "session_replay"})


def test_apply_profile_preserves_explicit_cli_values(tmp_path: Path) -> None:
    from argparse import Namespace

    write_config(
        tmp_path / "config.yaml",
        """\
version: 1
defaults:
  model: configured
  runtime:
    debug: true
    session_log: false
  sandbox:
    backend: docker
""",
    )
    profile = load_agent_profile(tmp_path, None, {})
    args = Namespace(model="cli", debug=False, no_session_log=False, sandbox="auto")

    apply_profile_to_args(args, profile, frozenset({"model", "debug", "no_session_log"}))

    assert args.model == "cli"
    assert args.debug is False
    assert args.no_session_log is False
    assert args.sandbox == "docker"


def test_changing_transport_name_drops_lower_layer_connection_settings(tmp_path: Path) -> None:
    write_config(
        tmp_path / "config.yaml",
        """\
version: 1
defaults:
  transport:
    name: openai
    base_url: https://openai.internal/v1
    api_key_env: OPENAI_INTERNAL_KEY
""",
    )

    profile = load_agent_profile(tmp_path, None, {"AXIO_REPL_TRANSPORT": "anthropic"})

    assert profile.settings.transport.name == "anthropic"
    assert profile.settings.transport.base_url is None
    assert profile.settings.transport.api_key_env is None


def test_explicit_transport_drops_profile_connection_settings(tmp_path: Path) -> None:
    from argparse import Namespace

    write_config(
        tmp_path / "config.yaml",
        """\
version: 1
defaults:
  transport:
    name: openai
    base_url: https://openai.internal/v1
    api_key_env: OPENAI_INTERNAL_KEY
""",
    )
    profile = load_agent_profile(tmp_path, None, {})
    args = Namespace(
        transport="anthropic",
        transport_base_url=None,
        transport_api_key_env=None,
        no_session_log=False,
    )

    apply_profile_to_args(args, profile, frozenset({"transport"}))

    assert args.transport == "anthropic"
    assert args.transport_base_url is None
    assert args.transport_api_key_env is None


def test_explicit_transport_connection_settings_are_kept(tmp_path: Path) -> None:
    from argparse import Namespace

    profile = load_agent_profile(tmp_path, None, {})
    args = Namespace(
        transport="anthropic",
        transport_base_url="https://anthropic.internal",
        transport_api_key_env="ANTHROPIC_INTERNAL_KEY",
        no_session_log=False,
    )

    apply_profile_to_args(
        args,
        profile,
        frozenset({"transport", "transport_base_url", "transport_api_key_env"}),
    )

    assert args.transport_base_url == "https://anthropic.internal"
    assert args.transport_api_key_env == "ANTHROPIC_INTERNAL_KEY"


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("../escape", "agent name must"),
        ("space name", "agent name must"),
        ("", "agent name must"),
    ],
)
def test_agent_name_rejects_path_and_whitespace(name: str, message: str) -> None:
    with pytest.raises(AgentConfigError, match=message):
        resolve_agent_name(name, {})


def test_agent_name_can_come_from_environment() -> None:
    assert resolve_agent_name(None, {"AXIO_REPL_AGENT": "local.dev"}) == "local.dev"
    assert resolve_agent_name("cli", {"AXIO_REPL_AGENT": "environment"}) == "cli"


def test_missing_agent_fails_with_expected_manifest_path(tmp_path: Path) -> None:
    with pytest.raises(AgentConfigError, match=r"expected .*agents/missing/agent.yaml"):
        load_agent_profile(tmp_path, "missing", {})


def test_unknown_and_duplicate_fields_fail_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "agents" / "bad" / "agent.yaml"
    write_config(manifest, "version: 1\nunknown: true\n")
    with pytest.raises(AgentConfigError, match="unknown root field"):
        load_agent_profile(tmp_path, "bad", {})

    write_config(manifest, "version: 1\nversion: 1\n")
    with pytest.raises(AgentConfigError, match="duplicate key"):
        load_agent_profile(tmp_path, "bad", {})


def test_version_is_required_and_must_be_supported(tmp_path: Path) -> None:
    manifest = tmp_path / "agents" / "bad" / "agent.yaml"
    write_config(manifest, "description: missing version\n")
    with pytest.raises(AgentConfigError, match="version must be the integer 1"):
        load_agent_profile(tmp_path, "bad", {})

    write_config(manifest, "version: 2\n")
    with pytest.raises(AgentConfigError, match="version must be the integer 1"):
        load_agent_profile(tmp_path, "bad", {})


def test_instruction_paths_cannot_escape_bundle(tmp_path: Path) -> None:
    manifest = tmp_path / "agents" / "bad" / "agent.yaml"
    write_config(tmp_path / "outside.md", "outside\n")
    write_config(manifest, "version: 1\ninstructions: [../../outside.md]\n")

    with pytest.raises(AgentConfigError, match="escapes the agent bundle"):
        load_agent_profile(tmp_path, "bad", {})


def test_agent_instructions_have_a_total_size_limit(tmp_path: Path) -> None:
    manifest = tmp_path / "agents" / "large" / "agent.yaml"
    write_config(manifest.parent / "instructions.md", "x" * (1024 * 1024 + 1))
    write_config(manifest, "version: 1\ninstructions: [instructions.md]\n")

    profile = load_agent_profile(tmp_path, "large", {})

    with pytest.raises(AgentConfigError, match="instructions exceed"):
        profile.instructions_text()


def test_secret_is_stored_as_environment_reference_not_value(tmp_path: Path) -> None:
    manifest = tmp_path / "agents" / "secure" / "agent.yaml"
    write_config(
        manifest,
        """\
version: 1
transport:
  name: openai
  api_key_env: PRIVATE_OPENAI_TOKEN
""",
    )

    profile = load_agent_profile(tmp_path, "secure", {"PRIVATE_OPENAI_TOKEN": "not-read-by-loader"})

    assert profile.settings.transport.api_key_env == "PRIVATE_OPENAI_TOKEN"
    assert "not-read-by-loader" not in repr(profile)


def test_transport_secret_is_resolved_only_when_requested() -> None:
    assert resolve_api_key("PRIVATE_TOKEN", {"PRIVATE_TOKEN": "secret"}) == "secret"
    with pytest.raises(AgentConfigError, match="PRIVATE_TOKEN.*is not set"):
        resolve_api_key("PRIVATE_TOKEN", {})
    with pytest.raises(AgentConfigError, match="not a valid environment variable"):
        resolve_api_key("INVALID-NAME", {})


def test_relative_external_paths_are_confined_to_source_directory(tmp_path: Path) -> None:
    manifest = tmp_path / "agents" / "bad" / "agent.yaml"
    write_config(
        manifest,
        """\
version: 1
sandbox:
  datasets: ../../outside
""",
    )

    with pytest.raises(AgentConfigError, match="escapes its configuration directory"):
        load_agent_profile(tmp_path, "bad", {})


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"AXIO_REPL_DEBUG": "maybe"}, "must be a boolean"),
        ({"AXIO_REPL_MAX_ITERATIONS": "0"}, "must be a positive integer"),
        ({"AXIO_REPL_TEMPERATURE": "3"}, "must be between 0 and 2"),
        ({"AXIO_REPL_TRANSPORT_API_KEY_ENV": "BAD-NAME"}, "valid environment variable"),
        ({"AXIO_REPL_TRANSPORT_BASE_URL": "http://localhost:bad/v1"}, "valid port"),
        ({"AXIO_REPL_TOOLS": "shell,,read_file"}, "comma-separated list"),
    ],
)
def test_invalid_environment_overrides_fail_closed(tmp_path: Path, environment: dict[str, str], message: str) -> None:
    with pytest.raises(AgentConfigError, match=message):
        load_agent_profile(tmp_path, None, environment)
