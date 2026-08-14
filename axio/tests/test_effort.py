from __future__ import annotations

from axio.context import MemoryContextStore
from axio.effort import EffortMechanism, EffortRuntime


async def test_prompt_overlay_is_replaced_and_default_removes_it() -> None:
    runtime = EffortRuntime(object())
    context = MemoryContextStore()

    runtime.configure("high")
    high_prompt = runtime.system_prompt("base prompt")
    runtime.configure("low")
    low_prompt = runtime.system_prompt("base prompt")
    runtime.configure("default")
    default_prompt = runtime.system_prompt("base prompt")

    assert high_prompt.count("Effort guidance") == 1
    assert "Effort guidance (high)" in high_prompt
    assert "Effort guidance (low)" in low_prompt
    assert "Effort guidance (high)" not in low_prompt
    assert default_prompt == "base prompt"
    assert await context.get_history() == []


def test_prompt_none_states_its_limitation() -> None:
    runtime = EffortRuntime(object())

    state = runtime.configure("none")

    assert state.mechanism is EffortMechanism.prompt_fallback
    assert "cannot disable provider reasoning" in state.note
    assert "does not disable internal reasoning" in runtime.system_prompt("base")
