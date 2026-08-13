from __future__ import annotations

from axio.events import ToolInputDelta, ToolOutputDelta, ToolResult, ToolUseStart
from axio_tools_agents.runtime import AgentStarted, TurnStarted

from axio_repl._multiplexer import ActionMultiplexer, DisplayMode, sanitize_terminal_text


def test_display_mode_accepts_cli_and_descriptive_names() -> None:
    assert DisplayMode.parse("off") is DisplayMode.ACTIVE_ONLY
    assert DisplayMode.parse("active-only") is DisplayMode.ACTIVE_ONLY
    assert DisplayMode.parse("on") is DisplayMode.ALL_ACTIONS
    assert DisplayMode.parse("all-actions") is DisplayMode.ALL_ACTIONS


def test_tool_arguments_are_framed_only_after_complete_json() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json='{"command":"ec'))

    assert mux.drain() == []

    mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json='ho hi"}'))
    frames = mux.drain()

    assert len(frames) == 1
    assert "agent child · tool call" in frames[0]
    assert 'arguments: {"command": "echo hi"}' in frames[0]


def test_streaming_output_is_grouped_by_lines_and_flushed_at_result() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, output_chunk_chars=16)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="{}"))
    assert "tool call" in mux.drain()[0]

    mux.observe(
        "child",
        ToolOutputDelta(tool_use_id="call", name="shell", key="stdout", delta="first line\npartial"),
    )
    line_frame = mux.drain()[0]
    assert "shell stdout" in line_frame
    assert "first line" in line_frame
    assert "partial" not in line_frame

    mux.observe("child", ToolResult(tool_use_id="call", name="shell", is_error=False, content="ignored duplicate"))
    tail, result = mux.drain(max_frames=2)
    assert "partial" in tail
    assert "shell completed" in result
    assert "ignored duplicate" not in result


def test_round_robin_keeps_one_agent_from_monopolizing_a_boundary() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    mux.observe("alpha", TurnStarted(prompt="one"))
    mux.observe("alpha", TurnStarted(prompt="two"))
    mux.observe("beta", TurnStarted(prompt="three"))

    frames = mux.drain(max_frames=3)

    assert ["agent alpha" in frames[0], "agent beta" in frames[1], "agent alpha" in frames[2]] == [True] * 3


def test_overflow_is_bounded_and_reported_explicitly() -> None:
    mux = ActionMultiplexer(
        DisplayMode.ALL_ACTIONS,
        max_queued_frames=3,
        max_queued_bytes=100_000,
        max_frames_per_agent=2,
    )
    for _ in range(5):
        mux.observe("child", TurnStarted(prompt="not displayed"))

    assert mux.queued_count == 3
    frames = mux.drain(max_frames=3)

    assert "3 action frames suppressed" in frames[0]
    assert len(frames) == 3


def test_toggling_has_no_replay_and_reports_discarded_backlog() -> None:
    mux = ActionMultiplexer()
    mux.observe("child", AgentStarted(name="hidden", kind="background-agent"))

    enabled = mux.set_mode(DisplayMode.ALL_ACTIONS)
    assert enabled.discarded_frames == 0
    assert mux.drain() == []

    mux.observe("child", AgentStarted(name="visible", kind="background-agent"))
    disabled = mux.set_mode(DisplayMode.ACTIVE_ONLY)
    assert disabled.discarded_frames == 1
    assert disabled.discarded_bytes > 0

    mux.set_mode(DisplayMode.ALL_ACTIONS)
    assert mux.drain() == []


def test_frames_strip_ansi_osc_and_control_characters() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    mux.observe("child\x1b[31m", AgentStarted(name="\x1b]0;owned\x07safe\x00", kind="background-agent"))

    frame = mux.drain()[0]

    assert "\x1b[31m" not in frame
    assert "\x1b]0;owned" not in frame
    assert "\x00" not in frame
    assert "safe" in frame
    assert frame.endswith("\n")
    assert sanitize_terminal_text("one\r\ntwo\rthree") == "one\ntwo\nthree"


def test_boundary_drain_honours_frame_and_byte_budgets() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, max_frame_bytes=512)
    for agent_id in ("alpha", "beta", "gamma"):
        mux.observe(agent_id, TurnStarted(prompt="ignored"))

    first = mux.drain(max_frames=1, max_bytes=512)

    assert len(first) == 1
    assert len(first[0].encode()) <= 512
    assert mux.queued_count == 2


def test_frame_size_is_bounded_even_when_labels_contain_escapes() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, max_frame_bytes=180)
    agent_id = ("agent\x1b[31m" * 50) + "\x1b]0;owned\x07"
    mux.observe(agent_id, AgentStarted(name="x" * 10_000, kind="background-agent"))

    frame = mux.drain(max_bytes=180)[0]

    assert len(frame.encode()) <= 180
    assert "\x1b[31m" not in frame
    assert "\x1b]0;owned" not in frame
