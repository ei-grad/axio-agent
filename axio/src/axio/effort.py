"""Transport-owned effort control with a prompt fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

type EffortLevel = Literal["none", "low", "medium", "high", "xhigh", "max"]

EFFORT_LEVELS: tuple[EffortLevel, ...] = ("none", "low", "medium", "high", "xhigh", "max")


class EffortMechanism(StrEnum):
    native_effort = "native-effort"
    native_budget = "native-budget"
    prompt_fallback = "prompt-fallback"


@dataclass(frozen=True, slots=True)
class EffortState:
    requested: EffortLevel | None
    mechanism: EffortMechanism
    provider_value: str | int | None = None
    allowed: tuple[EffortLevel, ...] = EFFORT_LEVELS
    note: str = ""

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "requested": self.requested or "default",
            "mechanism": self.mechanism.value,
            "provider_value": self.provider_value,
            "note": self.note or None,
        }


@runtime_checkable
class EffortControl(Protocol):
    """Structural interface implemented by transports with native effort control."""

    def configure_effort(self, requested: str | None) -> EffortState: ...


def parse_effort(requested: str | None) -> EffortLevel | None:
    if requested is None or requested == "default":
        return None
    normalized = requested.casefold()
    if normalized not in EFFORT_LEVELS:
        raise ValueError(f"Invalid effort {requested!r}. Valid values: default, {', '.join(EFFORT_LEVELS)}")
    return normalized


@dataclass(slots=True)
class PromptEffortAdapter:
    """Fallback for transports without verified granular native control."""

    def configure_effort(self, requested: str | None) -> EffortState:
        level = parse_effort(requested)
        note = ""
        if level is not None:
            note = (
                "Prompt guidance changes observable analysis behavior only; it does not control provider reasoning "
                "tokens, latency, or cost."
            )
            if level == "none":
                note += " It cannot disable provider reasoning."
        return EffortState(level, EffortMechanism.prompt_fallback, note=note)


_PROMPT_GUIDANCE: dict[EffortLevel, str] = {
    "none": (
        "Use a direct approach with only the analysis needed for a correct answer. Avoid unnecessary exploration, "
        "and check the final result for obvious errors. This guidance does not disable internal reasoning."
    ),
    "low": (
        "Use focused analysis. Consider an alternative only when the first approach has a material weakness, and "
        "check the result against the task's explicit requirements."
    ),
    "medium": (
        "Use balanced analysis. Identify material assumptions, compare plausible alternatives when useful, and "
        "verify the result against the task's requirements."
    ),
    "high": (
        "Analyze the task deeply. Test important assumptions, compare plausible alternatives, and verify the result "
        "for correctness and completeness before answering."
    ),
    "xhigh": (
        "Use extensive analysis. Explore competing approaches and failure modes, resolve important uncertainties, "
        "and perform a thorough correctness and completeness check."
    ),
    "max": (
        "Use exhaustive task-focused analysis. Systematically examine alternatives, edge cases, and failure modes, "
        "then verify every material requirement and the final result."
    ),
}


@dataclass(slots=True)
class EffortRuntime:
    """Hold requested effort and render the effective prompt overlay."""

    transport: object
    state: EffortState = field(init=False)
    _fallback: PromptEffortAdapter = field(default_factory=PromptEffortAdapter, init=False, repr=False)

    def __post_init__(self) -> None:
        self.state = self._configure(None)

    def _configure(self, requested: str | None) -> EffortState:
        if isinstance(self.transport, EffortControl):
            return self.transport.configure_effort(requested)
        return self._fallback.configure_effort(requested)

    def configure(self, requested: str | None) -> EffortState:
        self.state = self._configure(requested)
        return self.state

    def reapply(self) -> EffortState:
        requested = self.state.requested
        self.state = self._configure(requested)
        return self.state

    def system_prompt(self, base: str) -> str:
        if self.state.mechanism is not EffortMechanism.prompt_fallback or self.state.requested is None:
            return base
        overlay = _PROMPT_GUIDANCE[self.state.requested]
        return f"{base}\n\nEffort guidance ({self.state.requested}):\n{overlay}"
