from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from axio.events import IterationEnd, StreamEvent, TextDelta
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.types import StopReason, Usage

from axio_repl import main


async def test_named_agent_configures_transport_sandbox_tools_and_instructions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import axio_repl

    config_dir = tmp_path / "config"
    bundle_dir = config_dir / "agents" / "local"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "instructions.md").write_text("Use the configured agent profile.\n", encoding="utf-8")
    (bundle_dir / "agent.yaml").write_text(
        """\
version: 1
instructions: [instructions.md]
transport:
  name: llama-cpp
  base_url: http://127.0.0.1:18080/v1
  api_key_env: LOCAL_LLM_TOKEN
model: configured/model
runtime:
  max_iterations: 17
sandbox:
  backend: docker
  image: local/sandbox:test
  network: agent-egress
  registries:
    pypi: http://devpi:3141/root/pypi/+simple/
tools: [read_file, shell]
""",
        encoding="utf-8",
    )
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "AGENTS.md").write_text("Use the project rules.\n", encoding="utf-8")

    constructor_kwargs: dict[str, object] = {}
    selected: dict[str, object] = {}
    observed: dict[str, object] = {}

    class ConfiguredTransport:
        name = "configured"

        def __init__(self, **kwargs: object) -> None:
            constructor_kwargs.update(kwargs)
            self.model = ModelSpec(
                id="configured/model",
                capabilities=frozenset({Capability.text, Capability.tool_use}),
            )
            self.models = ModelRegistry([self.model])
            self.temperature: float | None = None
            self.max_output_tokens: int | None = None
            self.debug = False

        async def fetch_models(self) -> None:
            pass

        async def stream(
            self,
            messages: list[Message],
            tools: list[Tool[Any]],
            system: str,
        ) -> AsyncIterator[StreamEvent]:
            del messages
            observed["tools"] = [tool.name for tool in tools]
            observed["system"] = system
            yield TextDelta(index=0, delta="done")
            yield IterationEnd(
                iteration=1,
                stop_reason=StopReason.end_turn,
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    def select_transport(name: str | None, credential_override: bool = False) -> tuple[type[ConfiguredTransport], str]:
        assert name == "llama-cpp"
        assert credential_override is True
        return ConfiguredTransport, ""

    async def build_tools(
        stack: object,
        tools: list[Tool[Any]],
        mode: str,
        image: str,
        workspace: Path,
        options: object,
    ) -> tuple[list[Tool[Any]], str, Path, str]:
        del stack
        selected.update(
            tools=[tool.name for tool in tools],
            mode=mode,
            image=image,
            workspace=workspace,
            options=options,
        )
        return tools, "configured sandbox", workspace, "sandbox note"

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    monkeypatch.setenv("LOCAL_LLM_TOKEN", "secret-value")
    monkeypatch.setattr(axio_repl, "_select_transport", select_transport)
    monkeypatch.setattr(axio_repl._sandbox, "build_tools", build_tools)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axio-repl",
            "run",
            "--config-dir",
            str(config_dir),
            "--agent",
            "local",
            "--no-session-log",
        ],
    )

    await main()

    assert constructor_kwargs["base_url"] == "http://127.0.0.1:18080/v1"
    assert constructor_kwargs["api_key"] == "secret-value"
    assert "session" in constructor_kwargs
    build_input_tools = cast(list[str], selected["tools"])
    assert "read_file" in build_input_tools
    assert "shell" in build_input_tools
    assert selected["mode"] == "docker"
    assert selected["image"] == "local/sandbox:test"
    assert selected["workspace"] == project_dir
    options = selected["options"]
    assert getattr(options, "network") == "agent-egress"
    assert getattr(options, "pypi_index") == "http://devpi:3141/root/pypi/+simple/"
    assert observed["tools"] == ["read_file", "shell"]
    system = str(observed["system"])
    assert "Use the configured agent profile." in system
    assert "Use the project rules." in system
    assert system.count("Agent profile instructions:") == 1
    assert system.count("AGENTS.md instructions:") == 1


async def test_list_agents_does_not_initialize_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import axio_repl

    config_dir = tmp_path / "config"
    for name in ("zeta", "alpha"):
        bundle_dir = config_dir / "agents" / name
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "agent.yaml").write_text("version: 1\n", encoding="utf-8")

    def unexpected_transport(name: str | None) -> None:
        raise AssertionError(f"transport initialized for {name}")

    monkeypatch.setattr(axio_repl, "_select_transport", unexpected_transport)
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--config-dir", str(config_dir), "--list-agents"])

    await main()

    assert capsys.readouterr().out == "alpha\nzeta\n"
