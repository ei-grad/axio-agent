from __future__ import annotations

import asyncio
import io
import re
from collections.abc import AsyncIterator
from contextlib import redirect_stdout
from typing import Any

import pytest
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import (
    Error,
    IterationEnd,
    ReasoningDelta,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.messages import Message
from axio.tool import Tool
from axio.types import StopReason, Usage
from axio_tools_agents.peers import run_agent, set_run_agent_factory, set_session_event_hub
from axio_tools_agents.runtime import (
    AgentEventEnvelope,
    AgentStarted,
    ExecutionMode,
    ForegroundEntered,
    ForegroundExited,
    RuntimeEvent,
    SessionEventHub,
    TurnStatus,
)

from axio_repl import ReplRenderer, render_runtime_event, run_prompt
from axio_repl._multiplexer import DisplayMode

_ACTION_FRAME = re.compile(r"\x1b\[0m\n── agent .*?── /agent .*?\n\x1b\[0m\n", re.DOTALL)


async def _queue_background_tool_action(renderer: ReplRenderer, agent_id: str = "child") -> None:
    await renderer.render(agent_id, ToolUseStart(index=0, tool_use_id=f"{agent_id}-call", name="shell"))
    await renderer.render(
        agent_id,
        ToolInputDelta(index=0, tool_use_id=f"{agent_id}-call", partial_json='{"command":"echo hi"}'),
    )


async def test_focusing_an_agent_does_not_replay_hidden_prose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    await renderer.render("child", TextDelta(index=0, delta="unique hidden report"))
    await renderer.render(
        "child",
        SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=Usage(input_tokens=1, output_tokens=2)),
    )

    assert "unique hidden report" not in capsys.readouterr().out

    renderer.set_focus("child")
    assert "unique hidden report" not in capsys.readouterr().out

    await renderer.incoming("Report from child:\n\nunique hidden report")
    assert capsys.readouterr().out.count("unique hidden report") == 1


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


async def test_all_actions_preserves_active_bytes_and_inserts_only_after_a_paragraph(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)

    await renderer.render("main", TextDelta(index=0, delta="first paragraph"))
    await _queue_background_tool_action(renderer)
    await renderer.render("main", TextDelta(index=0, delta="\n\nsecond paragraph"))

    output = capsys.readouterr().out
    assert output.index("first paragraph\n\n") < output.index("agent child · tool call")
    assert output.index("agent child · tool call") < output.index("second paragraph")
    assert _ACTION_FRAME.sub("", output) == "first paragraph\n\nsecond paragraph"


async def test_paragraph_boundary_split_across_deltas_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)

    await renderer.render("main", TextDelta(index=0, delta="first\n"))
    await _queue_background_tool_action(renderer)
    assert "agent child" not in capsys.readouterr().out

    await renderer.render("main", TextDelta(index=0, delta="\nsecond"))
    output = capsys.readouterr().out

    assert output.startswith("\n")
    assert "agent child · tool call" in output
    assert output.endswith("second")


async def test_background_actions_wait_until_reasoning_closes(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)

    await renderer.render("main", ReasoningDelta(index=0, delta="checking"))
    await _queue_background_tool_action(renderer)
    assert "agent child" not in capsys.readouterr().out

    await renderer.render("main", TextDelta(index=0, delta="answer"))
    output = capsys.readouterr().out

    assert output.index("agent child") < output.index("answer")


async def test_background_actions_wait_for_all_parallel_active_tools(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="one", name="shell"))
    await renderer.render("main", ToolInputDelta(index=0, tool_use_id="one", partial_json='{"command":"one"}'))
    await renderer.render("main", ToolUseStart(index=1, tool_use_id="two", name="shell"))
    await renderer.render("main", ToolInputDelta(index=1, tool_use_id="two", partial_json='{"command":"two"}'))
    await _queue_background_tool_action(renderer)
    assert "agent child" not in capsys.readouterr().out

    await renderer.render("main", ToolResult(tool_use_id="one", name="shell", is_error=False, content="one"))
    assert "agent child" not in capsys.readouterr().out

    await renderer.render(
        "main",
        ToolOutputDelta(tool_use_id="two", name="shell", key="stdout", delta="two\n"),
    )
    assert "agent child" not in capsys.readouterr().out

    await renderer.render("main", ToolResult(tool_use_id="two", name="shell", is_error=False, content="two"))
    assert "agent child · tool call" in capsys.readouterr().out


async def test_active_stream_is_identical_when_there_are_no_background_actions() -> None:
    events: list[StreamEvent] = [
        TextDelta(index=0, delta="live"),
        TextDelta(index=0, delta=" output\n\nnext"),
        ToolUseStart(index=0, tool_use_id="call", name="shell"),
        ToolInputDelta(index=0, tool_use_id="call", partial_json='{"command":"echo hi"}'),
        ToolOutputDelta(tool_use_id="call", name="shell", key="stdout", delta="hi\n"),
        ToolResult(tool_use_id="call", name="shell", is_error=False, content="hi\n"),
    ]

    async def render(mode: DisplayMode) -> str:
        output = io.StringIO()
        renderer = ReplRenderer(display_mode=mode)
        with redirect_stdout(output):
            for event in events:
                await renderer.render("main", event)
        return output.getvalue()

    assert await render(DisplayMode.ACTIVE_ONLY) == await render(DisplayMode.ALL_ACTIONS)


async def test_foreground_result_is_correlated_and_not_printed_twice(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="parent-call", name="run_agent"))
    await renderer.render("main", ToolInputDelta(index=0, tool_use_id="parent-call", partial_json='{"task":"go"}'))
    await renderer.enter_foreground("child", "parent-call")
    await renderer.render("child", TextDelta(index=0, delta="unique child answer"))
    await renderer.exit_foreground("child", TurnStatus.SUCCEEDED)
    await renderer.render(
        "main",
        ToolResult(
            tool_use_id="parent-call",
            name="run_agent",
            is_error=False,
            content="unique child answer",
        ),
    )

    output = capsys.readouterr().out
    assert output.count("unique child answer") == 1
    assert "foreground agent returned its result to the parent" in output
    assert renderer.focused_agent == "main"
    assert renderer.foreground_agent == "main"


async def test_parent_sibling_tool_stream_drains_at_child_paragraph_boundary_exactly_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="agent-call", name="run_agent"))
    await renderer.render("main", ToolInputDelta(index=0, tool_use_id="agent-call", partial_json='{"task":"go"}'))
    await renderer.render("main", ToolUseStart(index=1, tool_use_id="shell-call", name="shell"))
    await renderer.render(
        "main",
        ToolInputDelta(index=1, tool_use_id="shell-call", partial_json='{"command":"echo sibling"}'),
    )
    await renderer.enter_foreground("child", "agent-call")
    capsys.readouterr()

    await renderer.render("child", TextDelta(index=0, delta="child paragraph"))
    await renderer.render(
        "main",
        ToolOutputDelta(tool_use_id="shell-call", name="shell", key="stdout", delta="unique sibling line\n"),
    )
    await renderer.render(
        "main",
        ToolResult(tool_use_id="shell-call", name="shell", is_error=False, content="unique sibling line\n"),
    )
    assert "unique sibling line" not in capsys.readouterr().out

    await renderer.render("child", TextDelta(index=0, delta="\n\nchild continues"))
    output = capsys.readouterr().out

    assert output.index("\n\n") < output.index("agent main · shell stdout")
    assert output.index("shell completed") < output.index("child continues")
    assert output.count("unique sibling line") == 1


async def test_real_run_agent_and_streaming_sibling_preserve_nearest_safe_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    child_started = asyncio.Event()
    parent_output_observed = asyncio.Event()

    class ParentTransport:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(
            self,
            messages: list[Message],
            tools: list[Tool[Any]],
            system: str,
        ) -> AsyncIterator[StreamEvent]:
            del messages, tools, system
            self.calls += 1
            if self.calls == 1:
                yield ToolUseStart(index=0, tool_use_id="agent-call", name="run_agent")
                yield ToolInputDelta(index=0, tool_use_id="agent-call", partial_json='{"task":"inspect"}')
                yield ToolUseStart(index=1, tool_use_id="shell-call", name="shell")
                yield ToolInputDelta(index=1, tool_use_id="shell-call", partial_json='{"command":"echo sibling"}')
                yield IterationEnd(iteration=1, stop_reason=StopReason.tool_use, usage=Usage(1, 1))
                return
            yield TextDelta(index=0, delta="parent done")
            yield IterationEnd(iteration=2, stop_reason=StopReason.end_turn, usage=Usage(1, 1))

    class ChildTransport:
        async def stream(
            self,
            messages: list[Message],
            tools: list[Tool[Any]],
            system: str,
        ) -> AsyncIterator[StreamEvent]:
            del messages, tools, system
            child_started.set()
            yield TextDelta(index=0, delta="child paragraph")
            await asyncio.wait_for(parent_output_observed.wait(), timeout=1)
            yield TextDelta(index=0, delta="\n\nchild continues")
            yield IterationEnd(iteration=1, stop_reason=StopReason.end_turn, usage=Usage(1, 1))

    async def shell_handler(command: str) -> str:
        return command

    async def shell_stream(command: str) -> AsyncIterator[tuple[str, str]]:
        del command
        await asyncio.wait_for(child_started.wait(), timeout=1)
        yield "stdout", "unique concurrent sibling\n"

    setattr(shell_handler, "stream", shell_stream)
    hub = SessionEventHub(session_id="concurrent-render")
    renderer = ReplRenderer()

    async def coordinate(envelope: AgentEventEnvelope) -> None:
        if isinstance(envelope.event, ToolOutputDelta) and envelope.event.tool_use_id == "shell-call":
            parent_output_observed.set()

    async def render(envelope: AgentEventEnvelope) -> None:
        await render_runtime_event(renderer, envelope)

    hub.subscribe(coordinate)
    hub.subscribe(render)
    set_session_event_hub(hub)

    async def child_factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        assert not inherit_context
        return Agent(system="child", transport=ChildTransport()), MemoryContextStore()

    set_run_agent_factory(child_factory)
    parent = Agent(
        system="parent",
        transport=ParentTransport(),
        tools=[
            Tool(name="run_agent", handler=run_agent, concurrency=1, detachable=False),
            Tool(name="shell", handler=shell_handler),
        ],
    )
    try:
        outcome = await run_prompt(parent, MemoryContextStore(), "go", hub, "parent-run", source="test")
    finally:
        set_run_agent_factory(None)
        set_session_event_hub(None)

    output = capsys.readouterr().out
    assert outcome.succeeded
    assert output.count("unique concurrent sibling") == 1
    assert output.index("child paragraph\n\n") < output.index("unique concurrent sibling")
    assert output.index("unique concurrent sibling") < output.index("child continues")
    assert "foreground agent returned its result to the parent" in output


async def test_enabling_actions_does_not_replay_events_seen_while_off(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()
    await _queue_background_tool_action(renderer)

    change = await renderer.set_display_mode(DisplayMode.ALL_ACTIONS)
    await renderer.mark_idle()

    assert change.discarded_frames == 0
    assert capsys.readouterr().out == ""


async def test_background_action_is_drained_immediately_while_foreground_is_at_a_safe_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)
    await renderer.render("main", TextDelta(index=0, delta="paragraph\n\n"))
    capsys.readouterr()

    await _queue_background_tool_action(renderer)

    assert "agent child · tool call" in capsys.readouterr().out


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


async def test_all_actions_uses_a_labelled_error_frame_instead_of_the_legacy_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)
    await renderer.render("main", TextDelta(index=0, delta="paragraph\n\n"))
    capsys.readouterr()

    await renderer.render("child", Error(exception=RuntimeError("child failed")))

    output = capsys.readouterr().out
    assert "agent child · error" in output
    assert "RuntimeError: child failed" in output
    assert "[background" not in output


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


async def test_action_frame_does_not_force_a_prompt_toolkit_partial_line_flush(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import sys
    from unittest.mock import patch

    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)
    renderer.set_input_active(True)
    with patch.object(sys.stdout, "flush") as flush:
        await renderer.render("main", TextDelta(index=0, delta="paragraph\n"))
        await _queue_background_tool_action(renderer)
        await renderer.render("main", TextDelta(index=0, delta="\ncontinued"))

        assert flush.call_count == 0

    output = capsys.readouterr().out
    assert "paragraph\n\n" in output
    assert output.endswith("continued")


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
