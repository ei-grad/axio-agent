from __future__ import annotations

import pytest
from axio.events import ToolInputDelta, ToolOutputDelta, ToolResult, ToolUseStart
from axio.types import StopReason
from axio_tools_agents.runtime import ExecutionMode, TurnFinished, TurnStarted, TurnStatus

from axio_repl import ReplRenderer
from axio_repl._multiplexer import DisplayMode, sanitize_terminal_text
from axio_repl._theme import NO_COLOR_THEME


async def test_concurrent_live_results_keep_start_ordinals_when_completed_in_reverse(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(theme=NO_COLOR_THEME)

    await renderer.render("main", ToolUseStart(index=0, tool_use_id="provider-a", name="shell"))
    await renderer.render("main", ToolUseStart(index=1, tool_use_id="provider-b", name="write_file"))
    await renderer.render(
        "main",
        ToolResult(tool_use_id="provider-b", name="write_file", is_error=False, content="Wrote 1 bytes"),
    )
    await renderer.render(
        "main",
        ToolResult(tool_use_id="provider-a", name="shell", is_error=False, content="done"),
    )

    output = capsys.readouterr().out
    assert output.index("▶ shell #001") < output.index("▶ write_file #002")
    assert output.index("✓ write_file #002") < output.index("✓ shell #001")
    assert renderer.active_tool_call_count == 0
    assert renderer.next_tool_ordinal == 3


async def test_same_provider_id_is_new_after_completion(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = ReplRenderer(theme=NO_COLOR_THEME)

    for ordinal, expected in enumerate(("#001", "#002"), start=1):
        run_id = f"run-{ordinal}"
        turn_id = f"turn-{ordinal}"
        await renderer.start_turn(
            "main",
            TurnStarted(prompt="run"),
            run_id=run_id,
            turn_id=turn_id,
            execution_mode=ExecutionMode.FOREGROUND,
        )
        await renderer.render(
            "main",
            ToolUseStart(index=0, tool_use_id="reused", name="shell"),
            run_id=run_id,
            turn_id=turn_id,
            execution_mode=ExecutionMode.FOREGROUND,
        )
        await renderer.render(
            "main",
            ToolResult(tool_use_id="reused", name="shell", is_error=False, content="done"),
            run_id=run_id,
            turn_id=turn_id,
            execution_mode=ExecutionMode.FOREGROUND,
        )
        await renderer.finish_turn(
            "main",
            TurnFinished(status=TurnStatus.SUCCEEDED, stop_reason=StopReason.end_turn),
            run_id=run_id,
            turn_id=turn_id,
            execution_mode=ExecutionMode.FOREGROUND,
        )
        output = capsys.readouterr().out
        assert f"▶ shell {expected}" in output
        assert f"✓ shell {expected}" in output


async def test_turn_cleanup_releases_unfinished_call_without_reusing_ordinal() -> None:
    renderer = ReplRenderer(theme=NO_COLOR_THEME)
    await renderer.start_turn(
        "main",
        TurnStarted(prompt="run"),
        run_id="run-1",
        turn_id="turn-1",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    await renderer.render(
        "main",
        ToolUseStart(index=0, tool_use_id="unfinished", name="shell"),
        run_id="run-1",
        turn_id="turn-1",
        execution_mode=ExecutionMode.FOREGROUND,
    )
    assert renderer.active_tool_call_count == 1

    await renderer.finish_turn(
        "main",
        TurnFinished(status=TurnStatus.CANCELLED, stop_reason=None, error="cancelled"),
        run_id="run-1",
        turn_id="turn-1",
        execution_mode=ExecutionMode.FOREGROUND,
    )

    assert renderer.active_tool_call_count == 0
    assert renderer.next_tool_ordinal == 2


async def test_mismatch_and_orphan_results_fail_open_with_explicit_correlation_notice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(theme=NO_COLOR_THEME)
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="known", name="write_file"))
    capsys.readouterr()

    await renderer.render(
        "main",
        ToolResult(
            tool_use_id="known",
            name="patch_file",
            is_error=False,
            content="Wrote 5 bytes to app.py",
        ),
    )
    mismatch = capsys.readouterr().out
    assert "✗ write_file #001" in mismatch
    assert "name mismatch" in mismatch
    assert "Wrote 5 bytes to app.py" in mismatch

    await renderer.render(
        "main",
        ToolResult(tool_use_id="missing", name="read_file", is_error=False, content="full body"),
    )
    orphan = capsys.readouterr().out
    assert "✓ read_file #002" in orphan
    assert "[orphan tool result]" in orphan
    assert "full body" in orphan
    assert renderer.active_tool_call_count == 0


async def test_suppressed_write_and_streamed_shell_still_render_response_badge_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(theme=NO_COLOR_THEME)
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="write", name="write_file"))
    capsys.readouterr()

    await renderer.render(
        "main",
        ToolResult(tool_use_id="write", name="write_file", is_error=False, content="Wrote 5 bytes to app.py"),
    )
    assert capsys.readouterr().out == "\n✓ write_file #001\n"

    await renderer.render("main", ToolUseStart(index=1, tool_use_id="shell", name="shell"))
    await renderer.render(
        "main",
        ToolOutputDelta(tool_use_id="shell", name="shell", key="stdout", delta="unique output"),
    )
    capsys.readouterr()
    await renderer.render(
        "main",
        ToolResult(tool_use_id="shell", name="shell", is_error=False, content="unique output"),
    )
    assert capsys.readouterr().out == "\n✓ shell #002\n"


async def test_patch_response_badge_precedes_compact_diff(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = ReplRenderer(theme=NO_COLOR_THEME)
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="patch", name="patch_file"))
    capsys.readouterr()

    await renderer.render(
        "main",
        ToolResult(
            tool_use_id="patch",
            name="patch_file",
            is_error=False,
            content="+1 -1\n@@ -1 +1 @@ run\n-old\n+new\n",
        ),
    )

    output = capsys.readouterr().out
    assert output.index("✓ patch_file #001") < output.index("+1 -1")
    assert "-old\n+new" in output


async def test_background_actions_share_live_allocator_across_agents(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer(display_mode=DisplayMode.ALL_ACTIONS, theme=NO_COLOR_THEME)
    await renderer.render("main", ToolUseStart(index=0, tool_use_id="same", name="shell"))
    await renderer.render("child", ToolUseStart(index=0, tool_use_id="same", name="read_file"))
    await renderer.render(
        "child",
        ToolInputDelta(index=0, tool_use_id="same", partial_json='{"path":"app.py"}'),
    )
    await renderer.render(
        "child",
        ToolResult(tool_use_id="same", name="read_file", is_error=False, content="body"),
    )
    await renderer.mark_idle()

    output = sanitize_terminal_text(capsys.readouterr().out)
    assert "▶ shell #001" in output
    assert "▶ read_file #002" in output
    assert "✓ read_file #002" in output
    assert renderer.active_tool_call_count == 1

    await renderer.render(
        "main",
        ToolResult(tool_use_id="same", name="shell", is_error=False, content="done"),
    )
    assert renderer.active_tool_call_count == 0
