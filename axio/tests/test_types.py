"""Tests for axio.types: Usage, StopReason, ToolName, ToolCallID."""

import pytest

from axio.types import CostSource, StopReason, Usage


class TestUsage:
    def test_add(self) -> None:
        a = Usage(10, 5)
        b = Usage(3, 7)
        assert a + b == Usage(13, 12)

    def test_add_associative(self) -> None:
        a, b, c = Usage(1, 2), Usage(3, 4), Usage(5, 6)
        assert (a + b) + c == a + (b + c)

    def test_add_commutative(self) -> None:
        a = Usage(10, 5)
        b = Usage(3, 7)
        assert a + b == b + a

    def test_frozen(self) -> None:
        u = Usage(1, 2)
        with pytest.raises(AttributeError):
            u.input_tokens = 99  # type: ignore[misc]

    def test_identity(self) -> None:
        zero = Usage(0, 0)
        a = Usage(10, 5)
        assert a + zero == a

    def test_provider_cost_is_accumulated(self) -> None:
        first = Usage(10, 2, cost_usd=0.1, cost_source=CostSource.provider)
        second = Usage(20, 3, cost_usd=0.2, cost_source=CostSource.provider)

        total = first + second
        assert (total.input_tokens, total.output_tokens, total.cost_source) == (30, 5, CostSource.provider)
        assert total.cost_usd == pytest.approx(0.3)

    def test_mixed_cost_sources_remain_distinguishable(self) -> None:
        reported = Usage(10, 2, cost_usd=0.1, cost_source=CostSource.provider)
        estimated = Usage(20, 3, cost_usd=0.2, cost_source=CostSource.estimated)

        total = reported + estimated
        assert (total.input_tokens, total.output_tokens, total.cost_source) == (30, 5, CostSource.mixed)
        assert total.cost_usd == pytest.approx(0.3)

    def test_unknown_cost_makes_a_nonempty_total_unknown(self) -> None:
        reported = Usage(10, 2, cost_usd=0.1, cost_source=CostSource.provider)

        assert (reported + Usage(1, 1)).cost_usd is None
        assert (Usage(1, 1) + reported).cost_usd is None

    def test_empty_accumulator_is_a_cost_identity(self) -> None:
        reported = Usage(10, 2, cost_usd=0.1, cost_source=CostSource.provider)

        assert Usage(0, 0) + reported == reported
        assert reported + Usage(0, 0) == reported

    @pytest.mark.parametrize("cost", [-0.1, float("inf"), float("nan")])
    def test_rejects_invalid_cost(self, cost: float) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            Usage(1, 1, cost_usd=cost, cost_source=CostSource.provider)

    def test_rejects_cost_without_source(self) -> None:
        with pytest.raises(ValueError, match="provided together"):
            Usage(1, 1, cost_usd=0.1)


class TestStopReason:
    def test_values(self) -> None:
        assert set(StopReason) == {
            StopReason.end_turn,
            StopReason.tool_use,
            StopReason.max_tokens,
            StopReason.error,
        }

    def test_is_str(self) -> None:
        assert isinstance(StopReason.end_turn, str)

    def test_str_values(self) -> None:
        assert StopReason.end_turn == "end_turn"
        assert StopReason.tool_use == "tool_use"
        assert StopReason.max_tokens == "max_tokens"
        assert StopReason.error == "error"


class TestAliases:
    def test_tool_name_is_str(self) -> None:
        name: str = "my_tool"
        assert isinstance(name, str)

    def test_tool_call_id_is_str(self) -> None:
        call_id: str = "call_123"
        assert isinstance(call_id, str)
