"""Primitive types: ToolName, ToolCallID, StopReason, Usage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

type ToolName = str
type ToolCallID = str


class StopReason(StrEnum):
    end_turn = "end_turn"
    tool_use = "tool_use"
    max_tokens = "max_tokens"
    error = "error"


class CostSource(StrEnum):
    provider = "provider"
    estimated = "estimated"
    mixed = "mixed"


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None
    cost_source: CostSource | None = None

    def __post_init__(self) -> None:
        if (self.cost_usd is None) != (self.cost_source is None):
            raise ValueError("cost_usd and cost_source must be provided together")
        if self.cost_usd is not None and (not math.isfinite(self.cost_usd) or self.cost_usd < 0):
            raise ValueError("cost_usd must be finite and non-negative")

    def __add__(self, other: Usage) -> Usage:
        cost_usd: float | None = None
        cost_source: CostSource | None = None
        if self.cost_usd is not None and other.cost_usd is not None:
            cost_usd = self.cost_usd + other.cost_usd
            cost_source = (
                self.cost_source
                if self.cost_source == other.cost_source and self.cost_source is not CostSource.mixed
                else CostSource.mixed
            )
        elif self.input_tokens == 0 and self.output_tokens == 0 and self.cost_usd is None:
            cost_usd = other.cost_usd
            cost_source = other.cost_source
        elif other.input_tokens == 0 and other.output_tokens == 0 and other.cost_usd is None:
            cost_usd = self.cost_usd
            cost_source = self.cost_source
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost_usd,
            cost_source=cost_source,
        )
