from __future__ import annotations

import pytest
from axio.events import (
    Error,
    ReasoningDelta,
    SessionEndEvent,
    TextDelta,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.types import StopReason, Usage
from axio_tools_agents.runtime import (
    AgentEventEnvelope,
    AgentStarted,
    ExecutionMode,
    ForegroundEntered,
    ForegroundExited,
    RuntimeEvent,
    TurnStatus,
)

from axio_repl import ReplRenderer, render_runtime_event


async def test_one_shot_renderer_replays_buffered_background_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(buffer_background_events=True)

    await renderer.render("child", TextDelta(index=0, delta="background report"))
    await renderer.render(
        "child",
        SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=Usage(input_tokens=1, output_tokens=2)),
    )

    assert capsys.readouterr().out == ""

    renderer.set_focus("child")

    output = capsys.readouterr().out
    assert "background report" in output
    assert "[1in/2out tokens]" in output


async def test_one_shot_renderer_replays_each_buffered_background_agent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(buffer_background_events=True)

    await renderer.render("child-a", TextDelta(index=0, delta="first report"))
    await renderer.render("child-b", TextDelta(index=0, delta="second report"))

    assert capsys.readouterr().out == ""

    renderer.set_focus("child-a")
    first_output = capsys.readouterr().out
    assert "first report" in first_output
    assert "second report" not in first_output

    renderer.set_focus("child-b")
    second_output = capsys.readouterr().out
    assert "second report" in second_output


async def test_foreground_child_streams_without_changing_input_focus(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = ReplRenderer()

    def envelope(event: RuntimeEvent) -> AgentEventEnvelope:
        return AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id="child-run",
            agent_id="child",
            parent_agent_id="main",
            turn_id="child-turn",
            execution_mode=ExecutionMode.FOREGROUND,
            parent_tool_use_id="call-1",
            event=event,
        )

    await render_runtime_event(renderer, envelope(AgentStarted(name="child", kind="foreground-agent")))
    await render_runtime_event(renderer, envelope(ForegroundEntered(parent_agent_id="main")))
    await render_runtime_event(renderer, envelope(TextDelta(index=0, delta="live child output")))

    assert renderer.focused_agent == "main"
    assert renderer.foreground_agent == "child"
    assert "live child output" in capsys.readouterr().out

    await render_runtime_event(renderer, envelope(ForegroundExited(status=TurnStatus.SUCCEEDED)))
    assert renderer.focused_agent == "main"
    assert renderer.foreground_agent == "main"


async def test_foreground_child_reasoning_and_tool_actions_keep_the_active_streaming_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    def envelope(event: RuntimeEvent) -> AgentEventEnvelope:
        return AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id="child-run",
            agent_id="child",
            parent_agent_id="main",
            turn_id="child-turn",
            execution_mode=ExecutionMode.FOREGROUND,
            parent_tool_use_id="parent-call",
            event=event,
        )

    await render_runtime_event(renderer, envelope(ForegroundEntered(parent_agent_id="main")))
    await render_runtime_event(renderer, envelope(ReasoningDelta(index=0, delta="checking")))
    assert "> checking" in capsys.readouterr().out

    await render_runtime_event(
        renderer,
        envelope(ToolUseStart(index=0, tool_use_id="child-tool", name="shell")),
    )
    assert "▶ shell" in capsys.readouterr().out

    await render_runtime_event(
        renderer,
        envelope(ToolInputDelta(index=0, tool_use_id="child-tool", partial_json='{"command":"echo hi"}')),
    )
    arguments = capsys.readouterr().out
    assert "command" in arguments
    assert "echo hi" in arguments

    await render_runtime_event(
        renderer,
        envelope(ToolOutputDelta(tool_use_id="child-tool", name="shell", key="stdout", delta="hi\n")),
    )
    assert "hi" in capsys.readouterr().out

    await render_runtime_event(
        renderer,
        envelope(
            ToolResult(
                tool_use_id="child-tool",
                name="shell",
                is_error=False,
                content="hi\n",
            )
        ),
    )
    assert "hi" not in capsys.readouterr().out


async def test_a_background_failure_is_reported_with_its_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # "errors=1" says something went wrong and not what, which is the half that
    # decides whether to retry, fix, or give up.
    renderer = ReplRenderer()

    await renderer.render("child", Error(exception=RuntimeError("Stopped after 25 iterations")))
    await renderer.render(
        "child",
        SessionEndEvent(stop_reason=StopReason.error, total_usage=Usage(input_tokens=1, output_tokens=0)),
    )

    assert "Stopped after 25 iterations" in capsys.readouterr().out


async def test_a_half_written_line_waits_while_the_prompt_is_up(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Forcing it out is what put output and the prompt on the same row, and cost
    # each line its first characters.
    import sys
    from unittest.mock import patch

    renderer = ReplRenderer()

    with patch.object(sys.stdout, "flush") as flush:
        renderer.set_input_active(True)
        await renderer.render("main", TextDelta(index=0, delta="half a line"))
        assert flush.call_count == 0

        renderer.set_input_active(False)
        await renderer.render("main", TextDelta(index=0, delta=" and the rest"))
        assert flush.call_count == 1


async def test_an_arriving_message_is_shown_not_only_forwarded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # It used to reach the model's prompt and nothing else, so the only account
    # of a spawned agent's report was the model's summary of it.
    renderer = ReplRenderer()

    await renderer.incoming("Report from background agent child:\n\n## Findings\nAll good.")

    output = capsys.readouterr().out
    assert "## Findings" in output
    assert "All good." in output
    assert "incoming" in output


async def test_an_interrupted_answer_is_kept(capsys: pytest.CaptureFixture[str]) -> None:
    # The agent stores an iteration once the model stops talking. Cut it off
    # before that and the words are on screen and nowhere else, so the next turn
    # is answered by a model with no memory of saying them.
    renderer = ReplRenderer()

    await renderer.render("main", TextDelta(index=0, delta="I looked at the "))
    await renderer.render("main", TextDelta(index=0, delta="transport and"))

    assert renderer.take_pending_text("main") == "I looked at the transport and"
    assert renderer.take_pending_text("main") == ""


async def test_a_finished_iteration_leaves_nothing_to_keep() -> None:
    # It is already in the context; keeping it here too would say it twice.
    from axio.events import IterationEnd
    from axio.types import StopReason as _StopReason

    renderer = ReplRenderer()

    await renderer.render("main", TextDelta(index=0, delta="all done"))
    await renderer.render("main", IterationEnd(iteration=1, stop_reason=_StopReason.end_turn, usage=Usage(1, 1)))

    assert renderer.take_pending_text("main") == ""
