from __future__ import annotations

import pytest
from axio_tools_agents.runtime import ConfigurationChanged, RuntimeEvent

from axio_repl import ReplRenderer, _build_argument_parser, _handle_agent_actions
from axio_repl._multiplexer import DisplayMode


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
