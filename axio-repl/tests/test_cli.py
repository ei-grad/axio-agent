from __future__ import annotations

import sys

import pytest
from axio.events import ToolInputDelta, ToolUseStart
from axio_tools_agents.runtime import ConfigurationChanged, RuntimeEvent

from axio_repl import ReplRenderer, _build_argument_parser, _handle_agent_actions, main
from axio_repl._multiplexer import ActionMultiplexer, DisplayMode


def test_agent_actions_default_to_off() -> None:
    args = _build_argument_parser().parse_args([])

    assert args.agent_actions == "off"


def test_agent_actions_can_be_enabled() -> None:
    args = _build_argument_parser().parse_args(["--agent-actions", "on", "inspect"])

    assert args.agent_actions == "on"
    assert args.prompt == "inspect"


def test_agent_actions_rejects_unknown_modes() -> None:
    with pytest.raises(SystemExit):
        _build_argument_parser().parse_args(["--agent-actions", "verbose"])


def test_sandbox_networking_defaults_to_fail_closed() -> None:
    args = _build_argument_parser().parse_args([])

    assert args.sandbox_network is None
    assert args.sandbox_proxy is None
    assert args.sandbox_memory == "256m"
    assert args.sandbox_cpus == "1.0"


def test_sandbox_restricted_network_flags_are_parsed() -> None:
    args = _build_argument_parser().parse_args(
        [
            "--sandbox-network",
            "agent-egress",
            "--sandbox-proxy",
            "http://mitmania:8080",
            "--sandbox-pypi-index",
            "http://nexus:8081/repository/pypi/simple",
            "--sandbox-datasets",
            "/srv/datasets",
        ]
    )

    assert args.sandbox_network == "agent-egress"
    assert args.sandbox_proxy == "http://mitmania:8080"
    assert args.sandbox_pypi_index.endswith("/simple")
    assert args.sandbox_datasets.as_posix() == "/srv/datasets"


async def test_sandbox_none_reports_restricted_options_as_cli_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--sandbox", "none", "--sandbox-memory", "1g"])

    with pytest.raises(SystemExit, match="2"):
        await main()

    assert "restricted sandbox settings require Docker" in capsys.readouterr().err


async def test_agent_actions_command_toggles_and_publishes_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = ReplRenderer()
    published: list[RuntimeEvent] = []

    async def publish(event: RuntimeEvent) -> None:
        published.append(event)

    assert await _handle_agent_actions(renderer, "on", publish)

    assert renderer.display_mode is DisplayMode.ALL_ACTIONS
    assert published == [ConfigurationChanged(name="agent_actions", value="on", source="interactive")]
    assert "Agent actions" in capsys.readouterr().out


async def test_agent_actions_command_reports_invalid_value_without_changing_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    assert not await _handle_agent_actions(renderer, "verbose")

    assert renderer.display_mode is DisplayMode.ACTIVE_ONLY
    assert "must be 'on' or 'off'" in capsys.readouterr().out


async def test_agent_actions_command_reports_discarded_incomplete_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    renderer = ReplRenderer(action_multiplexer=mux)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="incomplete"))
    retained = mux.retained_bytes

    assert await _handle_agent_actions(renderer, "off")

    output = capsys.readouterr().out
    assert "discarded 0 queued frame(s)" in output
    assert f"({retained} bytes)" in output
