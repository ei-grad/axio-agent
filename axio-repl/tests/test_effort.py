from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from axio.effort import (
    EFFORT_LEVELS,
    EffortLevel,
    EffortMechanism,
    EffortRuntime,
    EffortState,
    PromptEffortAdapter,
    parse_effort,
)
from axio.models import Capability, ModelRegistry, ModelSpec

from axio_repl import (
    Command,
    _apply_cli_args,
    _apply_effort,
    _apply_model,
    _build_argument_parser,
    _clone_transport_for_spawn,
    _show_effort,
)


@dataclass
class _NativeTransport:
    model: ModelSpec
    models: ModelRegistry
    native_effort: str | None = None
    supported: dict[str, tuple[EffortLevel, ...]] = field(default_factory=dict)

    def configure_effort(self, requested: str | None) -> EffortState:
        level = parse_effort(requested)
        if Capability.reasoning not in self.model.capabilities:
            self.native_effort = None
            return PromptEffortAdapter().configure_effort(level)
        allowed = self.supported.get(self.model.id, EFFORT_LEVELS)
        if level is not None and level not in allowed:
            raise ValueError(f"Effort {level!r} is not supported by {self.model.id}")
        self.native_effort = level
        return EffortState(level, EffortMechanism.native_effort, provider_value=level, allowed=allowed)


_REASONING = ModelSpec(id="reasoning", capabilities=frozenset({Capability.text, Capability.reasoning}))
_PLAIN = ModelSpec(id="plain", capabilities=frozenset({Capability.text}))


def _runtime() -> EffortRuntime:
    return EffortRuntime(_NativeTransport(_REASONING, ModelRegistry([_REASONING, _PLAIN])))


def test_effort_show_set_default_and_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    effort = _runtime()

    _show_effort(effort)
    assert "Requested effort" in capsys.readouterr().out
    assert _apply_effort(effort, "high") is not None
    assert effort.state.requested == "high"
    assert "native-effort" in capsys.readouterr().out
    assert _apply_effort(effort, "default") is not None
    assert effort.state.requested is None
    capsys.readouterr()
    assert _apply_effort(effort, "impossible") is None
    assert effort.state.requested is None
    assert "Valid values" in capsys.readouterr().out


def test_cli_effort_uses_the_slash_command_semantics() -> None:
    args = _build_argument_parser().parse_args(["--effort", "xhigh"])
    effort = _runtime()
    commands = {"/effort": Command(lambda: None, lambda value: _apply_effort(effort, value))}

    _apply_cli_args(args, commands)

    assert effort.state.requested == "xhigh"
    assert args.effort == "xhigh"


def test_cli_invalid_effort_exits() -> None:
    args = _build_argument_parser().parse_args(["--effort", "impossible"])
    effort = _runtime()
    commands = {"/effort": Command(lambda: None, lambda value: _apply_effort(effort, value))}

    with pytest.raises(SystemExit) as exc_info:
        _apply_cli_args(args, commands)

    assert exc_info.value.code == 2


def test_parser_and_help_do_not_expose_thinking() -> None:
    parser = _build_argument_parser()

    assert "--effort" in parser.format_help()
    assert "--thinking" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--thinking", "high"])


def test_model_switch_reapplies_effort_and_reports_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    transport = _NativeTransport(_REASONING, ModelRegistry([_REASONING, _PLAIN]))
    effort = EffortRuntime(transport)
    effort.configure("high")
    agent: Any = SimpleNamespace(system="old", transport=transport)

    _apply_model(transport, agent, [], Path("/tmp/test-workspace"), "", "plain", effort=effort)

    assert effort.state.requested == "high"
    assert effort.state.mechanism is EffortMechanism.prompt_fallback
    assert "Effort guidance (high)" in agent.system
    assert "Effort reapplied" in capsys.readouterr().out


def test_model_switch_resets_effort_when_native_level_is_unavailable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    wide = ModelSpec(id="wide", capabilities=frozenset({Capability.text, Capability.reasoning}))
    narrow = ModelSpec(id="narrow", capabilities=frozenset({Capability.text, Capability.reasoning}))
    transport = _NativeTransport(
        wide,
        ModelRegistry([wide, narrow]),
        supported={"wide": ("low", "high"), "narrow": ("low",)},
    )
    effort = EffortRuntime(transport)
    effort.configure("high")
    agent: Any = SimpleNamespace(system="old", transport=transport)

    _apply_model(transport, agent, [], Path("/tmp/test-workspace"), "", "narrow", effort=effort)

    assert effort.state.requested is None
    assert effort.state.allowed == ("low",)
    assert "Effort guidance" not in agent.system
    output = capsys.readouterr().out
    assert "reset to" in output
    assert "not supported" in output


def test_effort_show_uses_model_specific_native_levels(capsys: pytest.CaptureFixture[str]) -> None:
    transport = _NativeTransport(
        _REASONING,
        ModelRegistry([_REASONING]),
        supported={"reasoning": ("low", "high")},
    )

    _show_effort(EffortRuntime(transport))

    output = capsys.readouterr().out
    assert "Valid values: default, low, high" in output
    assert "medium" not in output


def test_child_transport_and_prompt_inherit_effective_effort() -> None:
    transport = _NativeTransport(_REASONING, ModelRegistry([_REASONING]))
    native = EffortRuntime(transport)
    native.configure("high")

    child_transport = _clone_transport_for_spawn(transport)

    assert child_transport.native_effort == "high"
    fallback = EffortRuntime(object())
    fallback.configure("xhigh")
    child_system = fallback.system_prompt("child base")
    assert "Effort guidance (xhigh)" in child_system
