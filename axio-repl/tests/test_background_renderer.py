from __future__ import annotations

import asyncio
import dataclasses
import io
import re
from collections.abc import AsyncIterator
from contextlib import redirect_stderr, redirect_stdout
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
from axio_tools_agents.peers import PeerMessage, run_agent, set_run_agent_factory, set_session_event_hub
from axio_tools_agents.runtime import (
    AgentEventEnvelope,
    AgentStarted,
    AgentStopped,
    ExecutionMode,
    ForegroundEntered,
    ForegroundExited,
    RuntimeEvent,
    SessionEventHub,
    TurnFinished,
    TurnStarted,
    TurnStatus,
)

from axio_repl import DIM, RED, RESET, ReplRenderer, _peer_incoming_prompt, render_runtime_event, run_prompt
from axio_repl._multiplexer import ActionMultiplexer, DisplayMode
from axio_repl._powerline import agent_header
from axio_repl._theme import MONOCHROME_THEME

_ACTION_FRAME = re.compile(r"\x1b\[0m\n── agent .*?── /agent .*?\n\x1b\[0m\n", re.DOTALL)


async def _queue_background_tool_action(renderer: ReplRenderer, agent_id: str = "child") -> None:
    await renderer.render(agent_id, ToolUseStart(index=0, tool_use_id=f"{agent_id}-call", name="shell"))
    await renderer.render(
        agent_id,
        ToolInputDelta(index=0, tool_use_id=f"{agent_id}-call", partial_json='{"command":"echo hi"}'),
    )


async def test_panel_status_tracks_main_agent_phase() -> None:
    renderer = ReplRenderer()

    assert renderer.agent_status() == "main: idle"
    await renderer.start_turn(
        "main",
        TurnStarted(prompt="inspect"),
        run_id="run",
        turn_id="turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    assert renderer.agent_status() == "main: waiting for model"

    await renderer.render(
        "main",
        ReasoningDelta(index=0, delta="thinking"),
        run_id="run",
        turn_id="turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    assert renderer.agent_status() == "main: reasoning"

    await renderer.render(
        "main",
        ToolUseStart(index=0, tool_use_id="one", name="spawn_agent"),
        run_id="run",
        turn_id="turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    await renderer.render(
        "main",
        ToolUseStart(index=1, tool_use_id="two", name="spawn_agent"),
        run_id="run",
        turn_id="turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    assert renderer.agent_status() == "main: tools: spawn_agent ×2"

    await renderer.finish_turn(
        "main",
        TurnFinished(status=TurnStatus.SUCCEEDED, stop_reason=StopReason.end_turn),
        run_id="run",
        turn_id="turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    assert renderer.agent_status() == "main: idle"


async def test_ui_notice_stays_in_panel_and_is_terminal_safe(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = ReplRenderer()

    await renderer.notice(f"{DIM}queued command{RESET}")

    assert capsys.readouterr().out == ""
    assert renderer.panel_message == "queued command"


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

    await render_runtime_event(renderer, envelope(AgentStarted(name="researcher", kind="foreground-agent")))
    await render_runtime_event(renderer, envelope(ForegroundEntered(parent_agent_id="main")))
    await render_runtime_event(renderer, envelope(TurnStarted(prompt="inspect")))
    await render_runtime_event(renderer, envelope(TextDelta(index=0, delta="live child output")))

    assert renderer.focused_agent == "main"
    assert renderer.foreground_agent == "child"
    output = capsys.readouterr().out
    assert output.count("── agent researcher (child) ──") == 1
    assert output.index("── agent researcher (child) ──") < output.index("live child output")

    await render_runtime_event(renderer, envelope(ForegroundExited(status=TurnStatus.SUCCEEDED)))
    assert renderer.focused_agent == "main"
    assert renderer.foreground_agent == "main"


async def test_main_turn_omits_redundant_source_header_but_keeps_error_attribution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(main_agent_name="axio-repl")

    def envelope(event: RuntimeEvent) -> AgentEventEnvelope:
        return AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id="main-run",
            agent_id="main",
            parent_agent_id=None,
            turn_id="main-turn",
            execution_mode=ExecutionMode.FOREGROUND,
            parent_tool_use_id=None,
            event=event,
        )

    await render_runtime_event(renderer, envelope(TurnStarted(prompt="inspect")))
    await render_runtime_event(renderer, envelope(TextDelta(index=0, delta="main answer")))
    await render_runtime_event(
        renderer,
        envelope(ToolUseStart(index=0, tool_use_id="main-tool", name="shell")),
    )
    await render_runtime_event(renderer, envelope(Error(exception=RuntimeError("main failed"))))

    captured = capsys.readouterr()
    header = "── agent axio-repl (main) ──"
    assert header not in captured.out
    assert captured.out.index("main answer") < captured.out.index("▶ shell")
    assert "Error from agent axio-repl (main): main failed" in captured.err
    assert "── main ──" not in captured.out


async def test_powerline_styles_live_tool_and_subagent_frames(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS, powerline=True)

    await renderer.start_turn(
        "main",
        TurnStarted(prompt="inspect"),
        run_id="main-run",
        turn_id="main-turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    await renderer.render(
        "main",
        ToolUseStart(index=0, tool_use_id="main-tool", name="shell"),
        run_id="main-run",
        turn_id="main-turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    await _queue_background_tool_action(renderer)
    await renderer.mark_idle()

    output = capsys.readouterr().out
    assert "\ue0b0" in output
    assert "\ue0b2" not in output
    assert "agent main" not in output
    assert "▶ shell" in output
    assert "agent child" in output
    assert "\033[0m\n" in output


async def test_monochrome_theme_reaches_plain_and_powerline_renderers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    plain = ReplRenderer(theme=MONOCHROME_THEME)
    await plain.render("main", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    await plain.render(
        "main",
        ToolOutputDelta(tool_use_id="call", name="shell", key="stderr", delta="warning"),
    )
    await plain.render("main", Error(exception=RuntimeError("failure")))

    captured = capsys.readouterr()
    assert "\033[1;97m▶ shell\033[0m" in captured.out
    assert "\033[1;97mwarning\033[0m" in captured.out
    assert "\033[1;7mError from agent main: failure\033[0m" in captured.err

    powerline = ReplRenderer(
        display_mode=DisplayMode.ALL_ACTIONS,
        powerline=True,
        theme=MONOCHROME_THEME,
    )
    await powerline.render("main", ToolUseStart(index=0, tool_use_id="main-tool", name="shell"))
    await _queue_background_tool_action(powerline)
    await powerline.mark_idle()

    output = capsys.readouterr().out
    assert "\033[1;30;107m ▶ shell " in output
    assert "\033[1;97;100m agent child " in output
    assert "\033[1;30;47m tool call " in output


async def test_powerline_labels_foreground_child_but_not_ordinary_main_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(main_agent_name="axio-repl", powerline=True)
    await renderer.start_turn(
        "main",
        TurnStarted(prompt="inspect"),
        run_id="main-run",
        turn_id="main-turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    await renderer.render(
        "main",
        TextDelta(index=0, delta="ordinary main response"),
        run_id="main-run",
        turn_id="main-turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )

    await renderer.remember_agent("child", "researcher")
    await renderer.enter_foreground("child", parent_agent_id="main")
    await renderer.start_turn(
        "child",
        TurnStarted(prompt="continue"),
        run_id="child-run",
        turn_id="child-turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    await renderer.render(
        "child",
        TextDelta(index=0, delta="child response"),
        run_id="child-run",
        turn_id="child-turn",
        execution_mode=ExecutionMode.FOREGROUND,
    )

    output = capsys.readouterr().out
    assert "agent axio-repl (main)" not in output
    child_header = " agent researcher (child) "
    assert output.count(child_header) == 1
    assert output.index(child_header) < output.index("child response")


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


async def test_multiline_reasoning_reapplies_dim_after_every_terminal_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    await renderer.render("main", ReasoningDelta(index=0, delta="first\nsecond"))
    await renderer.render("main", ReasoningDelta(index=0, delta=" continues\nthird"))
    await renderer.render("main", TextDelta(index=0, delta="answer"))

    output = capsys.readouterr().out
    assert f"{DIM}> first{RESET}\n{DIM}> second{RESET}" in output
    assert f"{DIM} continues{RESET}\n{DIM}> third{RESET}" in output
    assert output.endswith("\nanswer")


async def test_multiline_tool_values_reapply_dim_across_streamed_chunks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    await renderer.render("main", ToolUseStart(index=0, tool_use_id="call", name="write_file"))
    await renderer.render("main", ToolInputDelta(index=0, tool_use_id="call", partial_json='{"content":"value 1'))
    await renderer.render("main", ToolInputDelta(index=0, tool_use_id="call", partial_json="\\nvalue 2"))
    await renderer.render("main", ToolInputDelta(index=0, tool_use_id="call", partial_json='\\nvalue 3"}'))

    output = capsys.readouterr().out
    assert f": {DIM}value 1{RESET}" in output
    assert f"\n{DIM}value 2{RESET}" in output
    assert f"\n{DIM}value 3{RESET}" in output


async def test_multiline_tool_output_reapplies_its_style_after_newlines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    await renderer.render("main", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    await renderer.render(
        "main",
        ToolOutputDelta(tool_use_id="call", name="shell", key="stdout", delta="one\ntwo\nthree"),
    )

    output = capsys.readouterr().out
    assert f"{DIM}one{RESET}\n{DIM}two{RESET}\n{DIM}three{RESET}" in output


async def test_foreground_tool_output_preserves_channel_order_styles_and_no_success_replay(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    await renderer.render("main", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    await renderer.render(
        "main",
        ToolOutputDelta(tool_use_id="call", name="shell", key="stdout", delta="first"),
    )
    await renderer.render(
        "main",
        ToolOutputDelta(tool_use_id="call", name="shell", key="stderr", delta="second"),
    )
    await renderer.render(
        "main",
        ToolOutputDelta(tool_use_id="call", name="shell", key="stdout", delta="third"),
    )
    await renderer.render(
        "main",
        ToolResult(tool_use_id="call", name="shell", is_error=False, content="firstsecondthird"),
    )

    output = capsys.readouterr().out
    muted_amber = "\033[2;33m"
    first = f"{DIM}first{RESET}"
    second = f"{muted_amber}second{RESET}"
    third = f"{DIM}third{RESET}"
    assert output.index(first) < output.index(second) < output.index(third)
    assert output.count("first") == output.count("second") == output.count("third") == 1


async def test_background_tool_output_preserves_partial_channel_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)
    await renderer.render("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    await renderer.render("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="{}"))
    capsys.readouterr()

    await renderer.render(
        "child",
        ToolOutputDelta(tool_use_id="call", name="shell", key="stdout", delta="first"),
    )
    await renderer.render(
        "child",
        ToolOutputDelta(tool_use_id="call", name="shell", key="stderr", delta="second\n"),
    )
    await renderer.render(
        "child",
        ToolOutputDelta(tool_use_id="call", name="shell", key="stdout", delta="third"),
    )
    await renderer.render(
        "child",
        ToolResult(tool_use_id="call", name="shell", is_error=False, content="firstsecond\nthird"),
    )
    await renderer.mark_idle()

    output = capsys.readouterr().out
    first = output.index("agent child · shell stdout")
    second = output.index("agent child · shell stderr")
    third = output.index("agent child · shell stdout", first + 1)
    completed = output.index("agent child · tool result")
    assert first < second < third < completed
    assert output.count("first") == output.count("second") == output.count("third") == 1


async def test_tool_stderr_is_muted_until_the_tool_reports_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    await renderer.render("main", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    await renderer.render(
        "main",
        ToolOutputDelta(tool_use_id="call", name="shell", key="stderr", delta="warning one\nwarning two"),
    )
    await renderer.render(
        "main",
        ToolResult(tool_use_id="call", name="shell", is_error=True, content="command failed"),
    )

    output = capsys.readouterr().out
    muted_amber = "\033[2;33m"
    red = "\033[31m"
    assert f"{muted_amber}warning one{RESET}\n{muted_amber}warning two{RESET}" in output
    assert f"{red}command failed{RESET}" in output
    assert f"{red}warning one" not in output


async def test_agent_switch_closes_reasoning_style_before_other_agent_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    await renderer.render("main", ReasoningDelta(index=0, delta="main reasoning"))
    await renderer.render(
        "child",
        TextDelta(index=0, delta="child answer"),
        execution_mode=ExecutionMode.FOREGROUND,
    )
    await renderer.render("main", TextDelta(index=0, delta="main answer"))

    output = capsys.readouterr().out
    assert f"{DIM}> main reasoning{RESET}\n" in output
    assert "child answer" in output
    assert output.endswith("main answer")


async def test_structured_tool_field_closes_dim_style_before_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    await renderer.render(
        "main",
        ToolInputDelta(index=0, tool_use_id="call", partial_json='{"command":"first\\nsecond"}'),
    )
    await renderer.render("main", TextDelta(index=0, delta="answer"))

    output = capsys.readouterr().out
    assert f"{DIM}first{RESET}\n{DIM}second{RESET}" in output
    assert output.endswith("\nanswer")


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


async def test_background_actions_stream_while_monitor_is_waiting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="wait", name="monitor"))
    await renderer.render(
        "main",
        ToolInputDelta(
            index=0,
            tool_use_id="wait",
            partial_json='{"agents":["child"],"timeout":120}',
        ),
    )
    capsys.readouterr()

    await _queue_background_tool_action(renderer)

    output = capsys.readouterr().out
    assert "agent child · tool call" in output
    assert renderer.queued_action_count == 0


async def test_monitor_does_not_open_boundary_for_parallel_foreground_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="wait", name="monitor"))
    await renderer.render(
        "main",
        ToolInputDelta(index=0, tool_use_id="wait", partial_json='{"agents":["child"]}'),
    )
    await renderer.render("main", ToolUseStart(index=1, tool_use_id="shell", name="shell"))
    await renderer.render(
        "main",
        ToolInputDelta(index=1, tool_use_id="shell", partial_json='{"command":"sleep 1"}'),
    )
    await _queue_background_tool_action(renderer)

    assert "agent child" not in capsys.readouterr().out

    await renderer.render("main", ToolResult(tool_use_id="shell", name="shell", is_error=False, content=""))

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
    await renderer.remember_agent("child-id", "researcher")
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="parent-call", name="run_agent"))
    await renderer.render("main", ToolInputDelta(index=0, tool_use_id="parent-call", partial_json='{"task":"go"}'))
    await renderer.enter_foreground("child-id", "parent-call")
    await renderer.render("child-id", TextDelta(index=0, delta="unique child answer"))
    await renderer.exit_foreground("child-id", TurnStatus.SUCCEEDED)
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
    assert "foreground agent researcher (child-id) returned its result to the parent" in output
    assert renderer.focused_agent == "main"
    assert renderer.foreground_agent == "main"


async def _prime_legacy_foreground_result(renderer: ReplRenderer, tool_use_id: str = "reused") -> None:
    await renderer.render("main", ToolUseStart(index=0, tool_use_id=tool_use_id, name="run_agent"))
    await renderer.enter_foreground("child", tool_use_id, "main")
    await renderer.exit_foreground("child", TurnStatus.SUCCEEDED)


async def test_legacy_foreground_marker_is_consumed_before_a_parent_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()
    await _prime_legacy_foreground_result(renderer)
    capsys.readouterr()

    await renderer.render(
        "main",
        ToolResult(tool_use_id="reused", name="run_agent", is_error=False, content="child result"),
    )

    output = capsys.readouterr().out
    assert "foreground agent child returned its result to the parent" in output
    assert "child result" not in output


async def test_iteration_end_purges_an_unconsumed_legacy_foreground_marker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()
    await _prime_legacy_foreground_result(renderer)
    await renderer.render(
        "main",
        IterationEnd(iteration=1, stop_reason=StopReason.end_turn, usage=Usage(1, 1)),
    )
    capsys.readouterr()

    await renderer.render(
        "main",
        ToolResult(tool_use_id="reused", name="shell", is_error=False, content="unrelated result"),
    )

    output = capsys.readouterr().out
    assert "unrelated result" in output
    assert "foreground agent" not in output


async def test_session_end_purges_an_unconsumed_legacy_foreground_marker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()
    await _prime_legacy_foreground_result(renderer)
    await renderer.render(
        "main",
        SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=Usage(1, 1)),
    )
    capsys.readouterr()

    await renderer.render(
        "main",
        ToolResult(tool_use_id="reused", name="shell", is_error=False, content="unrelated result"),
    )

    output = capsys.readouterr().out
    assert "unrelated result" in output
    assert "foreground agent" not in output


async def test_foreground_result_suppression_does_not_cross_parent_turns_when_tool_ids_repeat(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    def envelope(turn_id: str, event: RuntimeEvent) -> AgentEventEnvelope:
        return AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id="parent-run",
            agent_id="main",
            parent_agent_id=None,
            turn_id=turn_id,
            execution_mode=ExecutionMode.FOREGROUND,
            parent_tool_use_id=None,
            event=event,
        )

    await render_runtime_event(renderer, envelope("parent-turn-1", TurnStarted(prompt="delegate")))
    await render_runtime_event(
        renderer,
        envelope("parent-turn-1", ToolUseStart(index=0, tool_use_id="reused-call", name="run_agent")),
    )
    child = AgentEventEnvelope(
        seq=2,
        session_id="session",
        run_id="child-run",
        agent_id="child",
        parent_agent_id="main",
        turn_id="child-turn",
        execution_mode=ExecutionMode.FOREGROUND,
        parent_tool_use_id="reused-call",
        event=ForegroundEntered(parent_agent_id="main"),
    )
    await render_runtime_event(renderer, child)
    await render_runtime_event(
        renderer,
        dataclasses.replace(child, event=ForegroundExited(status=TurnStatus.SUCCEEDED)),
    )
    await render_runtime_event(
        renderer,
        envelope(
            "parent-turn-1",
            TurnFinished(status=TurnStatus.CANCELLED, stop_reason=None, error="cancelled"),
        ),
    )
    capsys.readouterr()

    await render_runtime_event(renderer, envelope("parent-turn-2", TurnStarted(prompt="retry")))
    await render_runtime_event(
        renderer,
        envelope(
            "parent-turn-2",
            ToolResult(
                tool_use_id="reused-call",
                name="shell",
                is_error=False,
                content="unrelated result",
            ),
        ),
    )

    output = capsys.readouterr().out
    assert "unrelated result" in output
    assert "foreground agent" not in output


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


async def test_suspended_tool_collector_obeys_the_total_retained_byte_cap(
    capsys: pytest.CaptureFixture[str],
) -> None:
    retained_limit = 4096
    suspended = ActionMultiplexer(
        DisplayMode.ALL_ACTIONS,
        max_queued_bytes=retained_limit,
        max_retained_bytes=retained_limit,
        output_chunk_chars=1_000_000,
    )
    renderer = ReplRenderer(suspended_action_multiplexer=suspended)
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="agent-call", name="run_agent"))
    await renderer.render("main", ToolInputDelta(index=0, tool_use_id="agent-call", partial_json='{"task":"go"}'))
    await renderer.render("main", ToolUseStart(index=1, tool_use_id="shell-call", name="shell"))
    await renderer.render("main", ToolInputDelta(index=1, tool_use_id="shell-call", partial_json="{}"))
    await renderer.enter_foreground("child", "agent-call")
    capsys.readouterr()

    for _ in range(10_000):
        await renderer.render(
            "main",
            ToolOutputDelta(tool_use_id="shell-call", name="shell", key="stdout", delta="x" * 1024),
        )

    assert suspended.retained_bytes <= retained_limit
    assert suspended.retained_collector_bytes == 0
    assert suspended.retained_tool_count == 0
    assert suspended.retained_suppression_bytes > 0
    assert renderer.retained_action_bytes <= renderer.max_retained_action_bytes

    await renderer.render(
        "main",
        ToolResult(tool_use_id="shell-call", name="shell", is_error=False, content="done"),
    )
    assert suspended.retained_tool_count == 0
    assert suspended.retained_agent_count == 0


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
    assert re.search(r"foreground agent [^\n]+ \([^)]+\) returned its result to the parent", output)


async def test_enabling_actions_does_not_replay_events_seen_while_off(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    await render_runtime_event(
        renderer,
        AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id="child-run",
            agent_id="child",
            parent_agent_id="main",
            turn_id=None,
            execution_mode=ExecutionMode.BACKGROUND,
            parent_tool_use_id="spawn-call",
            event=AgentStarted(name="analyst", kind="spawned-agent"),
        ),
    )
    await _queue_background_tool_action(renderer)

    change = await renderer.set_display_mode(DisplayMode.ALL_ACTIONS)
    await renderer.mark_idle()

    assert change.discarded_frames == 0
    assert capsys.readouterr().out == ""

    await _queue_background_tool_action(renderer)
    assert "agent analyst (child) · tool call" in capsys.readouterr().out


async def test_background_action_is_drained_immediately_while_foreground_is_at_a_safe_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)
    await renderer.render("main", TextDelta(index=0, delta="paragraph\n\n"))
    capsys.readouterr()

    await _queue_background_tool_action(renderer)

    assert "agent child · tool call" in capsys.readouterr().out


async def test_two_focused_turns_each_get_one_canonical_source_header(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    def envelope(turn_id: str | None, event: RuntimeEvent) -> AgentEventEnvelope:
        return AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id="child-run",
            agent_id="child-id",
            parent_agent_id="main",
            turn_id=turn_id,
            execution_mode=ExecutionMode.BACKGROUND,
            parent_tool_use_id="spawn-call",
            event=event,
        )

    await render_runtime_event(renderer, envelope(None, AgentStarted(name="analyst", kind="spawned-agent")))
    renderer.set_focus("child-id")
    for index in (1, 2):
        turn_id = f"child-turn-{index}"
        await render_runtime_event(renderer, envelope(turn_id, TurnStarted(prompt=f"prompt {index}")))
        await render_runtime_event(renderer, envelope(turn_id, TextDelta(index=0, delta=f"answer {index}")))
        await render_runtime_event(
            renderer,
            envelope(
                turn_id,
                SessionEndEvent(
                    stop_reason=StopReason.end_turn,
                    total_usage=Usage(input_tokens=1, output_tokens=1),
                ),
            ),
        )
        await render_runtime_event(
            renderer,
            envelope(
                turn_id,
                TurnFinished(status=TurnStatus.SUCCEEDED, stop_reason=StopReason.end_turn),
            ),
        )

    output = capsys.readouterr().out
    header = "── agent analyst (child-id) ──"
    assert output.count(header) == 2
    assert output.index(header) < output.index("answer 1")
    assert output.rindex(header) < output.index("answer 2")


async def test_focused_final_reply_is_not_replayed_by_incoming_delivery(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    def envelope(event: RuntimeEvent) -> AgentEventEnvelope:
        return AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id="child-run",
            agent_id="child-id",
            parent_agent_id="main",
            turn_id="focused-turn",
            execution_mode=ExecutionMode.BACKGROUND,
            parent_tool_use_id="spawn-call",
            event=event,
        )

    await render_runtime_event(renderer, envelope(AgentStarted(name="analyst", kind="spawned-agent")))
    renderer.set_focus("child-id")
    await render_runtime_event(renderer, envelope(TurnStarted(prompt="inspect")))
    await render_runtime_event(renderer, envelope(TextDelta(index=0, delta="unique focused final")))
    await render_runtime_event(
        renderer,
        envelope(
            SessionEndEvent(
                stop_reason=StopReason.end_turn,
                total_usage=Usage(input_tokens=1, output_tokens=1),
            )
        ),
    )
    await render_runtime_event(
        renderer,
        envelope(TurnFinished(status=TurnStatus.SUCCEEDED, stop_reason=StopReason.end_turn)),
    )
    agent_name, suppress_display = await renderer.take_background_outcome_presentation(
        "child-id", "child-run", "focused-turn"
    )
    await renderer.incoming(
        "Report from background agent analyst (child-id):\n\nunique focused final",
        agent_id="child-id",
        agent_name=agent_name,
        suppress_display=suppress_display,
    )

    output = capsys.readouterr().out
    assert output.count("unique focused final") == 1
    assert "incoming from agent analyst (child-id)" not in output


async def test_focusing_a_running_hidden_turn_keeps_the_whole_turn_background(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    def envelope(event: RuntimeEvent) -> AgentEventEnvelope:
        return AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id="child-run",
            agent_id="child-id",
            parent_agent_id="main",
            turn_id="child-turn",
            execution_mode=ExecutionMode.BACKGROUND,
            parent_tool_use_id="spawn-call",
            event=event,
        )

    await render_runtime_event(renderer, envelope(AgentStarted(name="analyst", kind="spawned-agent")))
    await render_runtime_event(renderer, envelope(TurnStarted(prompt="inspect")))
    await render_runtime_event(renderer, envelope(TextDelta(index=0, delta="hidden prefix ")))
    renderer.set_focus("child-id")
    await render_runtime_event(renderer, envelope(TextDelta(index=0, delta="hidden suffix")))
    await render_runtime_event(
        renderer,
        envelope(SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=Usage(1, 1))),
    )
    await render_runtime_event(
        renderer,
        envelope(TurnFinished(status=TurnStatus.SUCCEEDED, stop_reason=StopReason.end_turn)),
    )
    agent_name, suppress_display = await renderer.take_background_outcome_presentation(
        "child-id", "child-run", "child-turn"
    )
    await renderer.incoming(
        "Report from background agent analyst (child-id):\n\nhidden prefix hidden suffix",
        agent_id="child-id",
        agent_name=agent_name,
        suppress_display=suppress_display,
    )

    output = capsys.readouterr().out
    assert output.count("hidden prefix hidden suffix") == 1
    assert "── agent analyst (child-id) ──" not in output
    assert "incoming from agent analyst (child-id)" in output


async def test_powerline_incoming_report_uses_the_shared_agent_badge(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(powerline=True)

    await renderer.incoming(
        "Report from background agent analyst (child-id):\n\nreport body",
        agent_id="child-id",
        agent_name="analyst",
    )

    output = capsys.readouterr().out
    identity = "analyst (child-id)"
    assert f"\n{agent_header(identity)}\n" in output
    assert "report body" in output
    assert "incoming from agent" not in output


async def test_focusing_away_from_a_running_live_turn_keeps_the_whole_turn_live(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    def envelope(event: RuntimeEvent) -> AgentEventEnvelope:
        return AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id="child-run",
            agent_id="child-id",
            parent_agent_id="main",
            turn_id="child-turn",
            execution_mode=ExecutionMode.BACKGROUND,
            parent_tool_use_id="spawn-call",
            event=event,
        )

    await render_runtime_event(renderer, envelope(AgentStarted(name="analyst", kind="spawned-agent")))
    renderer.set_focus("child-id")
    await render_runtime_event(renderer, envelope(TurnStarted(prompt="inspect")))
    await render_runtime_event(renderer, envelope(TextDelta(index=0, delta="live prefix ")))
    renderer.set_focus("main")
    await render_runtime_event(renderer, envelope(TextDelta(index=0, delta="live suffix")))
    await render_runtime_event(
        renderer,
        envelope(SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=Usage(1, 1))),
    )
    await render_runtime_event(
        renderer,
        envelope(TurnFinished(status=TurnStatus.SUCCEEDED, stop_reason=StopReason.end_turn)),
    )
    agent_name, suppress_display = await renderer.take_background_outcome_presentation(
        "child-id", "child-run", "child-turn"
    )
    await renderer.incoming(
        "Report from background agent analyst (child-id):\n\nlive prefix live suffix",
        agent_id="child-id",
        agent_name=agent_name,
        suppress_display=suppress_display,
    )

    output = capsys.readouterr().out
    assert output.count("live prefix live suffix") == 1
    assert output.count("── agent analyst (child-id) ──") == 1
    assert "incoming from agent analyst (child-id)" not in output


async def test_actions_on_labels_same_named_agents_with_their_distinct_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)

    def envelope(agent_id: str) -> AgentEventEnvelope:
        return AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id=f"{agent_id}-run",
            agent_id=agent_id,
            parent_agent_id="main",
            turn_id=None,
            execution_mode=ExecutionMode.BACKGROUND,
            parent_tool_use_id="spawn-call",
            event=AgentStarted(name="worker", kind="spawned-agent"),
        )

    await render_runtime_event(renderer, envelope("child-a"))
    await render_runtime_event(renderer, envelope("child-b"))
    await renderer.mark_idle()

    output = capsys.readouterr().out
    assert "agent worker (child-a) · lifecycle" in output
    assert "agent worker (child-b) · lifecycle" in output
    assert "── agent worker ·" not in output


async def test_actions_off_failure_uses_one_canonical_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    def envelope(event: RuntimeEvent) -> AgentEventEnvelope:
        return AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id="child-run",
            agent_id="child-id",
            parent_agent_id="main",
            turn_id="failed-turn",
            execution_mode=ExecutionMode.BACKGROUND,
            parent_tool_use_id="spawn-call",
            event=event,
        )

    await render_runtime_event(renderer, envelope(AgentStarted(name="analyst", kind="spawned-agent")))
    await render_runtime_event(renderer, envelope(TurnStarted(prompt="inspect")))
    await render_runtime_event(renderer, envelope(Error(exception=RuntimeError("child failed"))))
    await render_runtime_event(
        renderer,
        envelope(SessionEndEvent(stop_reason=StopReason.error, total_usage=Usage(input_tokens=1, output_tokens=0))),
    )
    await render_runtime_event(
        renderer,
        envelope(TurnFinished(status=TurnStatus.FAILED, stop_reason=StopReason.error, error="child failed")),
    )
    await renderer.mark_idle()

    output = capsys.readouterr().out
    assert "[background" not in output
    assert renderer.panel_message.count("Background agent analyst (child-id) completed") == 1
    assert "child failed" not in output


async def test_all_actions_uses_a_labelled_error_frame_instead_of_the_legacy_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS)
    await renderer.render("main", TextDelta(index=0, delta="paragraph\n\n"))
    capsys.readouterr()

    await renderer.render("child", Error(exception=RuntimeError("child failed")))

    output = capsys.readouterr().out
    assert "agent child · error" in output
    assert "agent stream failed" in output
    assert "child failed" not in output
    assert "[background" not in output


async def test_error_only_turn_is_self_labelled_on_stderr_without_a_stdout_header() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    renderer = ReplRenderer(main_agent_name="axio-repl")
    envelope = AgentEventEnvelope(
        seq=1,
        session_id="session",
        run_id="main-run",
        agent_id="main",
        parent_agent_id=None,
        turn_id="error-turn",
        execution_mode=ExecutionMode.FOREGROUND,
        parent_tool_use_id=None,
        event=TurnStarted(prompt="fail"),
    )

    with redirect_stdout(stdout), redirect_stderr(stderr):
        await render_runtime_event(renderer, envelope)
        assert stdout.getvalue() == ""
        await render_runtime_event(
            renderer,
            dataclasses.replace(envelope, event=Error(exception=RuntimeError("boom"))),
        )
        await render_runtime_event(
            renderer,
            dataclasses.replace(
                envelope,
                event=SessionEndEvent(stop_reason=StopReason.error, total_usage=Usage(1, 0)),
            ),
        )

    assert stdout.getvalue() == ""
    assert f"{RED}Error from agent axio-repl (main): boom{RESET}" in stderr.getvalue()


async def test_error_after_stdout_keeps_error_attribution_without_a_main_header() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    renderer = ReplRenderer(main_agent_name="axio-repl")
    envelope = AgentEventEnvelope(
        seq=1,
        session_id="session",
        run_id="main-run",
        agent_id="main",
        parent_agent_id=None,
        turn_id="partial-error-turn",
        execution_mode=ExecutionMode.FOREGROUND,
        parent_tool_use_id=None,
        event=TurnStarted(prompt="fail later"),
    )

    with redirect_stdout(stdout), redirect_stderr(stderr):
        await render_runtime_event(renderer, envelope)
        await render_runtime_event(renderer, dataclasses.replace(envelope, event=TextDelta(0, "partial")))
        await render_runtime_event(
            renderer,
            dataclasses.replace(envelope, event=Error(exception=RuntimeError("later boom"))),
        )

    assert "── agent axio-repl (main) ──" not in stdout.getvalue()
    assert stdout.getvalue().endswith("partial")
    assert "Error from agent axio-repl (main): later boom" in stderr.getvalue()


async def test_agent_lifecycle_and_incoming_labels_do_not_grow_renderer_state() -> None:
    renderer = ReplRenderer(max_identity_cache=16)

    for index in range(2_000):
        agent_id = f"child-{index}"
        base = AgentEventEnvelope(
            seq=index * 2,
            session_id="session",
            run_id=f"run-{index}",
            agent_id=agent_id,
            parent_agent_id="main",
            turn_id=None,
            execution_mode=ExecutionMode.BACKGROUND,
            parent_tool_use_id="spawn-call",
            event=AgentStarted(name=f"worker-{index}", kind="spawned-agent"),
        )
        await render_runtime_event(renderer, base)
        await render_runtime_event(
            renderer,
            dataclasses.replace(base, event=AgentStopped(status=TurnStatus.SUCCEEDED)),
        )

    state_count = renderer.retained_agent_state_count
    identity_count = renderer.retained_identity_count
    await renderer.incoming("hello", agent_id="unregistered", agent_name="external")

    assert state_count == 1
    assert renderer.retained_agent_state_count == state_count
    assert identity_count <= 16
    assert renderer.retained_identity_count == identity_count


def test_peer_identity_is_sanitized_for_display_without_mutating_parent_payload() -> None:
    raw_name = "analyst\x1b[31m\nnext\tname\x9b2J"
    raw_id = "peer\x1b]2;owned\x07\x85id\x9dtitle\x9c"
    message = PeerMessage(
        id="message",
        from_id=raw_id,
        from_name=raw_name,
        to_id="main",
        body="body",
        sent_at=1.0,
    )

    prompt = _peer_incoming_prompt(message)

    assert raw_name in prompt.text
    assert raw_id in prompt.text
    assert prompt.display_text is not None
    metadata = prompt.display_text.split(":\n\n", 1)[0]
    assert "analyst next name (peer id)" in metadata
    assert not any(ord(character) < 32 or 0x7F <= ord(character) <= 0x9F for character in metadata)
    assert "\x1b" not in metadata


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
    await renderer.remember_agent("child-id", "analyst")

    await renderer.incoming(
        "Report from background agent analyst (child-id):\n\n## Findings\nAll good.",
        agent_id="child-id",
    )

    output = capsys.readouterr().out
    assert "## Findings" in output
    assert "All good." in output
    assert "incoming from agent analyst (child-id)" in output


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
