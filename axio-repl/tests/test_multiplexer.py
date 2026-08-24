from __future__ import annotations

import pytest
from axio.events import SessionEndEvent, ToolInputDelta, ToolOutputDelta, ToolResult, ToolUseStart
from axio.types import StopReason, Usage
from axio_tools_agents.runtime import AgentStarted, AgentStopped, TurnFinished, TurnStarted, TurnStatus

from axio_repl._multiplexer import ActionMultiplexer, DisplayMode, sanitize_terminal_text
from axio_repl._theme import DEFAULT_THEME, MONOCHROME_THEME, NO_COLOR_THEME
from axio_repl._tool_calls import ToolCallRegistry, tool_display_name


def test_display_mode_accepts_cli_and_descriptive_names() -> None:
    assert DisplayMode.parse("off") is DisplayMode.ACTIVE_ONLY
    assert DisplayMode.parse("active-only") is DisplayMode.ACTIVE_ONLY
    assert DisplayMode.parse("on") is DisplayMode.ALL_ACTIONS
    assert DisplayMode.parse("all-actions") is DisplayMode.ALL_ACTIONS


def test_tool_registry_binding_requires_a_strictly_fresh_multiplexer() -> None:
    registry = ToolCallRegistry()
    fresh = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    fresh.bind_tool_calls(registry)
    assert fresh.tool_calls is registry

    queued_lifecycle = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    queued_lifecycle.observe("child", TurnStarted(prompt="run"))
    with pytest.raises(RuntimeError, match="observed activity"):
        queued_lifecycle.bind_tool_calls(registry)

    queued_tool = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    queued_tool.observe("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    queued_tool.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="{}"))
    with pytest.raises(RuntimeError, match="observed activity"):
        queued_tool.bind_tool_calls(registry)

    completed = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    completed.observe("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    completed.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="{}"))
    completed.observe(
        "child",
        ToolResult(tool_use_id="call", name="shell", is_error=False, content="done"),
    )
    completed.drain(max_frames=10)
    with pytest.raises(RuntimeError, match="observed activity"):
        completed.bind_tool_calls(registry)

    suppressed = ActionMultiplexer(DisplayMode.ALL_ACTIONS, max_tools=1)
    suppressed.observe("child", ToolUseStart(index=0, tool_use_id="one", name="shell"))
    suppressed.observe("child", ToolUseStart(index=1, tool_use_id="two", name="shell"))
    assert suppressed.retained_suppression_bytes > 0
    with pytest.raises(RuntimeError, match="observed activity"):
        suppressed.bind_tool_calls(registry)


def test_no_color_action_frames_emit_no_ansi() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, theme=NO_COLOR_THEME)
    mux.observe("child", AgentStarted(name="child", kind="spawned-agent"))

    [frame] = mux.drain()

    assert "agent child" in frame
    assert "\x1b[" not in frame


def test_tool_badge_name_and_frame_accounting_remain_bounded() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, theme=NO_COLOR_THEME, max_frame_bytes=180)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name="λ" * 10_000))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="{}"))

    [frame] = mux.drain(max_bytes=180)

    assert len(frame.encode("utf-8")) <= 180
    assert "▶ " in frame and "… #001" in frame


def test_background_tool_name_controls_surrogates_and_multibyte_text_are_safe() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, theme=NO_COLOR_THEME)
    name = ("λ" * 10_000) + "\nowned\udcff\033[2J"
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name=name))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="{}"))
    mux.observe("child", ToolResult(tool_use_id="call", name=name, is_error=False, content="done"))

    call, result = mux.drain(max_frames=2)
    combined = call + result
    assert "\udcff" not in combined
    assert "\033[2J" not in combined
    assert "owned" not in combined
    assert max(len(line.encode("utf-8")) for line in combined.splitlines()) <= 100


def test_megabyte_tool_names_are_bounded_in_collector_and_evict_under_cap() -> None:
    mux = ActionMultiplexer(
        DisplayMode.ALL_ACTIONS,
        theme=NO_COLOR_THEME,
        max_tools=1,
        max_frame_bytes=512,
        max_retained_bytes=4096,
    )
    first_name = ("λ" * 500_000) + "\nfirst\udcff\033[2J"
    second_name = ("μ" * 500_000) + "\nsecond\udcff\033[2J"

    mux.observe("child", ToolUseStart(index=0, tool_use_id="one", name=first_name))
    first_expected = len("child") + len("one") + len(tool_display_name(first_name).encode()) + 16 + len("#001")
    assert mux.retained_collector_bytes == first_expected
    mux.observe("child", ToolUseStart(index=1, tool_use_id="two", name=second_name))
    second_expected = len("child") + len("two") + len(tool_display_name(second_name).encode()) + 16 + len("#002")
    assert mux.retained_collector_bytes == second_expected
    assert mux.retained_suppression_bytes > 0
    mux.observe("child", ToolInputDelta(index=1, tool_use_id="two", partial_json="{}"))
    assert mux.retained_bytes <= 4096

    suppressed, call = mux.drain(max_frames=2)
    assert "incomplete action" in suppressed
    assert "▶ " in call and "#002" in call
    assert max(len(line.encode("utf-8")) for line in call.splitlines()) <= 100

    mux.observe(
        "child",
        ToolResult(tool_use_id="two", name=second_name, is_error=False, content="done"),
    )
    [result] = mux.drain()
    assert "✓ " in result and "#002" in result
    assert "name mismatch" not in result
    assert mux.retained_collector_bytes == 0


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
    assert "✓ shell #001" in result
    assert "ignored duplicate" not in result


@pytest.mark.parametrize("powerline", [False, True])
def test_background_patch_result_has_owned_semantic_colors_and_reuses_argument_path(powerline: bool) -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, powerline=powerline)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="patch", name="patch_file"))
    mux.observe(
        "child",
        ToolInputDelta(
            index=0,
            tool_use_id="patch",
            partial_json='{"path":"src/app.py","from_line":1,"to_line":1,"content":"new\\n"}',
        ),
    )
    event = ToolResult(
        tool_use_id="patch",
        name="patch_file",
        is_error=False,
        content=("+1 -1\n@@ -1 +1 @@ run\n-old\n\\ No newline at end of file\n+\x1b[2Jnew\n"),
    )
    mux.observe("child", event)

    call, result = mux.drain(max_frames=2)
    combined = call + result
    assert sanitize_terminal_text(combined).count("src/app.py") == 1
    if powerline:
        assert "\033[1;30;42m ✓ patch_file #001 \033[22;32;49m\ue0b0\033[0m\n" in result
    else:
        assert f"{DEFAULT_THEME.success.ansi}✓ patch_file #001{DEFAULT_THEME.reset}\n" in result
    assert f"{DEFAULT_THEME.stdout.ansi}+1 -1{DEFAULT_THEME.reset}\n" in result
    assert f"{DEFAULT_THEME.tool.ansi}@@ -1 +1 @@ run{DEFAULT_THEME.reset}\n" in result
    assert f"{DEFAULT_THEME.error.ansi}-old{DEFAULT_THEME.reset}\n" in result
    assert f"{DEFAULT_THEME.stdout.ansi}\\ No newline at end of file{DEFAULT_THEME.reset}\n" in result
    assert f"{DEFAULT_THEME.success.ansi}+new{DEFAULT_THEME.reset}\n" in result
    assert result.count(DEFAULT_THEME.error.ansi) == 1
    assert "\x1b[2J" not in result
    assert "\x1b[2J" in event.content
    assert "Wrote" not in result and "Changed" not in result
    assert "---" not in result and "+++" not in result


def test_background_patch_result_respects_no_color_even_if_powerline_is_requested() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, powerline=True, theme=NO_COLOR_THEME)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="patch", name="patch_file"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="patch", partial_json="{}"))
    mux.observe(
        "child",
        ToolResult(
            tool_use_id="patch",
            name="patch_file",
            is_error=False,
            content="+1 -1\n@@ -1 +1 @@ run\n-old\n+new\n",
        ),
    )

    call, result = mux.drain(max_frames=2)
    assert "+1 -1\n@@ -1 +1 @@ run\n-old\n+new" in result
    assert "\x1b[" not in call + result
    assert "\ue0b0" not in call + result


def test_background_malformed_hunk_fails_open_without_diff_styles() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    content = "+1 -1\n@@ -1,2 +1 @@ run\n-old\n+new\n"
    event = ToolResult(tool_use_id="patch", name="patch_file", is_error=False, content=content)

    mux.observe("child", event)

    [result] = mux.drain()
    assert content.rstrip() in sanitize_terminal_text(result)
    assert f"{DEFAULT_THEME.success.ansi}+new" not in result
    assert f"{DEFAULT_THEME.error.ansi}-old" not in result
    assert event.content == content


def test_background_styled_patch_frame_byte_accounting_includes_ansi_and_truncation() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, powerline=True, max_frame_bytes=512)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="patch", name="patch_file"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="patch", partial_json="{}"))
    mux.drain()
    mux.observe(
        "child",
        ToolResult(
            tool_use_id="patch",
            name="patch_file",
            is_error=False,
            content=("+20 -20\n@@ -20,20 +20,20 @@ run\n-old\n+new\n...[diff truncated]\n"),
        ),
    )
    retained = mux.queued_bytes

    [result] = mux.drain(max_frames=1, max_bytes=512)

    assert retained == len(result.encode("utf-8"))
    assert len(result.encode("utf-8")) <= 512
    assert f"{DEFAULT_THEME.stdout.ansi}...[diff truncated]{DEFAULT_THEME.reset}" in result


@pytest.mark.parametrize("powerline", [False, True])
def test_background_styled_patch_frame_remains_bounded_when_body_is_cut(powerline: bool) -> None:
    mux = ActionMultiplexer(
        DisplayMode.ALL_ACTIONS,
        powerline=powerline,
        theme=MONOCHROME_THEME,
        max_frame_bytes=260,
    )
    mux.observe("child", ToolUseStart(index=0, tool_use_id="patch", name="patch_file"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="patch", partial_json="{}"))
    mux.drain()
    mux.observe(
        "child",
        ToolResult(
            tool_use_id="patch",
            name="patch_file",
            is_error=False,
            content=f"+1 -1\n@@ -1 +1 @@ run\n-{'x' * 300}\n+{'y' * 300}\n",
        ),
    )
    retained = mux.queued_bytes

    [result] = mux.drain(max_frames=1, max_bytes=260)

    assert retained == len(result.encode("utf-8"))
    assert len(result.encode("utf-8")) <= 260
    notice = f"{MONOCHROME_THEME.stdout.ansi}[… truncated]{MONOCHROME_THEME.reset}"
    assert notice in result
    assert f"{MONOCHROME_THEME.success.ansi}[… truncated]" not in result
    assert f"{MONOCHROME_THEME.error.ansi}[… truncated]" not in result
    assert f"{MONOCHROME_THEME.reasoning.ansi}[… truncated]" not in result
    if powerline:
        assert " /agent child " in result
    else:
        assert "── /agent child ──" in result


def test_background_generated_truncation_notice_respects_no_color() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, theme=NO_COLOR_THEME, max_frame_bytes=220)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="patch", name="patch_file"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="patch", partial_json="{}"))
    mux.drain()
    mux.observe(
        "child",
        ToolResult(
            tool_use_id="patch",
            name="patch_file",
            is_error=False,
            content=f"+1 -1\n@@ -1 +1 @@ run\n-{'x' * 300}\n+{'y' * 300}\n",
        ),
    )
    retained = mux.queued_bytes

    [result] = mux.drain(max_frames=1, max_bytes=220)

    assert retained == len(result.encode("utf-8"))
    assert len(result.encode("utf-8")) <= 220
    assert "[… truncated]" in result
    assert "\x1b[" not in result


def test_background_write_ack_suppression_is_structural_and_errors_remain_visible() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="write", name="write_file"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="write", partial_json='{"path":"app.py"}'))
    assert "tool call" in mux.drain()[0]

    mux.observe(
        "child",
        ToolResult(tool_use_id="write", name="write_file", is_error=False, content="Wrote 5 bytes to app.py"),
    )
    [response] = mux.drain()
    assert "✓ write_file #001" in response
    assert "Wrote" not in response

    mux.observe(
        "child",
        ToolResult(tool_use_id="error", name="write_file", is_error=True, content="provider failed"),
    )
    mux.observe(
        "child",
        ToolResult(tool_use_id="read", name="read_file", is_error=False, content="Wrote 5 bytes to app.py"),
    )
    error, unrelated = mux.drain(max_frames=2)
    assert "provider failed" in error
    assert "Wrote 5 bytes to app.py" in unrelated


def test_background_mismatch_uses_remembered_name_and_fails_open() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, theme=NO_COLOR_THEME)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name="write_file"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="{}"))
    mux.drain()

    mux.observe(
        "child",
        ToolResult(
            tool_use_id="call",
            name="patch_file",
            is_error=False,
            content="Wrote 5 bytes to app.py",
        ),
    )

    [result] = mux.drain()
    assert "✗ write_file #001" in result
    assert "name mismatch" in result
    assert "Wrote 5 bytes to app.py" in result
    assert "\x1b[" not in result


def test_late_old_turn_result_does_not_consume_reused_background_call() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, theme=NO_COLOR_THEME)
    mux.observe(
        "child",
        ToolUseStart(index=0, tool_use_id="same", name="shell"),
        run_id="old-run",
        turn_id="old-turn",
    )
    mux.observe(
        "child",
        ToolInputDelta(index=0, tool_use_id="same", partial_json="{}"),
        run_id="old-run",
        turn_id="old-turn",
    )
    mux.drain()
    mux.observe(
        "child",
        TurnFinished(status=TurnStatus.CANCELLED, stop_reason=None, error="cancelled"),
        run_id="old-run",
        turn_id="old-turn",
    )
    mux.drain()
    mux.observe(
        "child",
        ToolUseStart(index=0, tool_use_id="same", name="shell"),
        run_id="new-run",
        turn_id="new-turn",
    )
    mux.observe(
        "child",
        ToolInputDelta(index=0, tool_use_id="same", partial_json="{}"),
        run_id="new-run",
        turn_id="new-turn",
    )
    [current_call] = mux.drain()
    assert "▶ shell #002" in current_call

    mux.observe(
        "child",
        ToolResult(tool_use_id="same", name="shell", is_error=False, content="late old"),
        run_id="old-run",
        turn_id="old-turn",
    )
    [late] = mux.drain()
    assert "✓ shell #003" in late
    assert "[orphan tool result]" in late

    mux.observe(
        "child",
        ToolResult(tool_use_id="same", name="shell", is_error=False, content="current"),
        run_id="new-run",
        turn_id="new-turn",
    )
    [current] = mux.drain()
    assert "✓ shell #002" in current
    assert "[orphan tool result]" not in current


def test_partial_output_segments_flush_in_executor_observed_order() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, output_chunk_chars=64)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="{}"))
    assert "tool call" in mux.drain()[0]

    mux.observe("child", ToolOutputDelta(tool_use_id="call", name="shell", key="stdout", delta="first"))
    mux.observe("child", ToolOutputDelta(tool_use_id="call", name="shell", key="stderr", delta="second\n"))
    mux.observe("child", ToolOutputDelta(tool_use_id="call", name="shell", key="stdout", delta="third"))
    assert mux.drain() == []

    mux.observe(
        "child",
        ToolResult(tool_use_id="call", name="shell", is_error=False, content="firstsecond\nthird"),
    )
    first, second, third, result = mux.drain(max_frames=4)

    assert "shell stdout" in first and "first" in first
    assert "shell stderr" in second and "second" in second
    assert "shell stdout" in third and "third" in third
    assert "✓ shell #001" in result
    assert "firstsecond" not in result


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


def test_high_cardinality_agents_bound_every_retained_index() -> None:
    mux = ActionMultiplexer(
        DisplayMode.ALL_ACTIONS,
        max_queued_frames=32,
        max_queued_bytes=100_000,
        max_frames_per_agent=4,
        max_agents=16,
        max_tools=24,
        max_tools_per_agent=4,
    )

    for index in range(10_000):
        agent_id = f"agent-{index}"
        mux.observe(agent_id, ToolUseStart(index=0, tool_use_id=f"call-{index}", name="shell"))
        mux.observe(agent_id, AgentStarted(name=agent_id, kind="background-agent"))

    assert mux.retained_agent_count <= 16
    assert mux.retained_tool_count <= 24
    assert mux.retained_queue_count <= 32
    assert mux.round_robin_count <= 32
    assert "incomplete action" in mux.drain(max_frames=1)[0]


def test_incomplete_calls_are_globally_bounded_and_terminal_events_cleanup_state() -> None:
    mux = ActionMultiplexer(
        DisplayMode.ALL_ACTIONS,
        max_queued_frames=64,
        max_queued_bytes=100_000,
        max_frames_per_agent=64,
        max_agents=8,
        max_tools=10,
        max_tools_per_agent=6,
    )
    for index in range(100):
        mux.observe("child", ToolUseStart(index=index, tool_use_id=f"call-{index}", name="shell"))

    assert mux.retained_agent_count == 1
    assert mux.retained_tool_count == 6

    mux.observe("child", AgentStopped(status=TurnStatus.CANCELLED))

    assert mux.retained_agent_count == 0
    assert mux.retained_tool_count == 0
    frames = mux.drain(max_frames=8)
    assert "incomplete action" in frames[0]
    assert any("stopped (cancelled)" in frame for frame in frames)


def test_invalid_partial_input_payload_is_included_in_the_global_retained_byte_cap() -> None:
    retained_limit = 4096
    mux = ActionMultiplexer(
        DisplayMode.ALL_ACTIONS,
        max_queued_bytes=retained_limit,
        max_retained_bytes=retained_limit,
    )
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))

    for _ in range(10_000):
        mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="x" * 1024))

    assert mux.queued_bytes == 0
    assert mux.retained_bytes <= retained_limit
    assert mux.retained_collector_bytes == 0
    assert mux.retained_tool_count == 0
    assert mux.retained_suppression_bytes > 0
    assert mux.retained_bytes == mux.queued_bytes + mux.retained_suppression_bytes
    assert "incomplete action discarded" in mux.drain(max_frames=1, max_bytes=retained_limit)[0]


def test_ordered_output_fragments_are_included_in_the_global_retained_byte_cap() -> None:
    retained_limit = 4096
    mux = ActionMultiplexer(
        DisplayMode.ALL_ACTIONS,
        max_queued_bytes=retained_limit,
        max_retained_bytes=retained_limit,
        output_chunk_chars=1_000_000,
    )
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="{}"))

    for index in range(10_000):
        mux.observe(
            "child",
            ToolOutputDelta(
                tool_use_id="call",
                name="shell",
                key="stdout" if index % 2 == 0 else "stderr",
                delta="x" * 1024,
            ),
        )

    assert mux.retained_bytes <= retained_limit
    assert mux.retained_collector_bytes == 0
    assert mux.retained_tool_count == 0
    assert mux.retained_suppression_bytes > 0
    assert mux.retained_bytes == mux.queued_bytes + mux.retained_suppression_bytes
    assert "incomplete action discarded" in mux.drain(max_frames=1, max_bytes=retained_limit)[0]


@pytest.mark.parametrize(
    ("max_tools", "max_tools_per_agent"),
    [(1, 2), (2, 1), (1, 1), (2, 3), (3, 2)],
)
def test_global_and_per_agent_tool_caps_never_leave_detached_indexes(
    max_tools: int,
    max_tools_per_agent: int,
) -> None:
    mux = ActionMultiplexer(
        DisplayMode.ALL_ACTIONS,
        max_tools=max_tools,
        max_tools_per_agent=max_tools_per_agent,
    )
    tool_ids = [f"call-{index}" for index in range(6)]
    for index, tool_use_id in enumerate(tool_ids):
        mux.observe("child", ToolUseStart(index=index, tool_use_id=tool_use_id, name="shell"))

    assert mux.retained_tool_count <= min(max_tools, max_tools_per_agent)
    for tool_use_id in tool_ids:
        mux.observe("child", ToolResult(tool_use_id=tool_use_id, name="shell", is_error=False, content="done"))
    mux.observe(
        "child",
        SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=Usage(input_tokens=1, output_tokens=1)),
    )

    assert mux.retained_tool_count == 0
    assert mux.retained_agent_count == 0
    assert mux.retained_collector_bytes == 0
    assert mux.retained_bytes <= mux.max_retained_bytes


def test_session_end_cleans_incomplete_collector_but_keeps_lifecycle_frame() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    mux.observe(
        "child",
        SessionEndEvent(stop_reason=StopReason.error, total_usage=Usage(input_tokens=1, output_tokens=0)),
    )

    assert mux.retained_agent_count == 0
    assert mux.retained_tool_count == 0
    frames = mux.drain(max_frames=2)
    assert "incomplete action discarded" in frames[0]
    assert "session ended (error)" in frames[1]


def test_overflow_prefers_dropping_verbose_frames_before_a_critical_result() -> None:
    mux = ActionMultiplexer(
        DisplayMode.ALL_ACTIONS,
        max_queued_frames=3,
        max_queued_bytes=100_000,
        max_frames_per_agent=3,
    )
    mux.observe(
        "child",
        ToolResult(tool_use_id="unknown", name="shell", is_error=True, content="critical failure"),
    )
    for _ in range(8):
        mux.observe("child", TurnStarted(prompt="verbose"))

    frames = mux.drain(max_frames=4)

    assert any("critical failure" in frame for frame in frames)
    assert "action frames suppressed" in frames[0]


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


def test_disabling_accounts_for_and_clears_unframed_collector_payload() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="incomplete"))
    retained = mux.retained_bytes

    assert retained > mux.queued_bytes
    disabled = mux.set_mode(DisplayMode.ACTIVE_ONLY)

    assert disabled.discarded_bytes == retained
    assert mux.retained_bytes == 0
    assert mux.retained_tool_count == 0


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


def test_powerline_suppression_frame_honours_drain_and_retained_byte_budgets() -> None:
    mux = ActionMultiplexer(
        DisplayMode.ALL_ACTIONS,
        powerline=True,
        max_queued_frames=2,
        max_queued_bytes=2048,
        max_frames_per_agent=2,
        max_frame_bytes=512,
        max_retained_bytes=4096,
    )
    for _ in range(6):
        mux.observe("child", TurnStarted(prompt="ignored"))

    assert mux.retained_bytes <= mux.max_retained_bytes
    frames = mux.drain(max_frames=1, max_bytes=512)

    assert len(frames) == 1
    assert len(frames[0].encode()) <= 512
    assert "4 action frames suppressed" in frames[0]
    assert "\ue0b2" not in frames[0]
    assert "\ue0b0" in frames[0]
    assert mux.queued_count == 2


def test_frame_size_is_bounded_even_when_labels_contain_escapes() -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS, max_frame_bytes=180)
    agent_id = ("agent\x1b[31m" * 50) + "\x1b]0;owned\x07"
    mux.observe(agent_id, AgentStarted(name="x" * 10_000, kind="background-agent"))

    frame = mux.drain(max_bytes=180)[0]

    assert len(frame.encode()) <= 180
    assert "\x1b[31m" not in frame
    assert "\x1b]0;owned" not in frame
