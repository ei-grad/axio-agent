from __future__ import annotations

from axio_repl._theme import DEFAULT_THEME, NO_COLOR_THEME
from axio_repl._tool_calls import (
    ToolBadgeKind,
    ToolCallKey,
    ToolCallRegistry,
    tool_badge,
    tool_display_name,
)


def test_registry_correlates_concurrent_calls_and_never_reuses_ordinals() -> None:
    registry = ToolCallRegistry()
    first_key = ToolCallKey("main", "run", "turn", "provider-a")
    second_key = ToolCallKey("main", "run", "turn", "provider-b")

    first = registry.start(first_key, "shell")
    second = registry.start(second_key, "write_file")

    assert (first.marker, second.marker) == ("#001", "#002")
    assert registry.result(second_key, "write_file").call == second
    registry.complete(second_key)
    assert registry.result(first_key, "shell").call == first
    registry.complete(first_key)
    assert registry.active_count == 0

    reused = registry.start(ToolCallKey("main", "next-run", "next-turn", "provider-a"), "shell")
    other_agent = registry.start(ToolCallKey("child", "run", "turn", "provider-a"), "shell")
    assert (reused.marker, other_agent.marker) == ("#003", "#004")


def test_registry_labels_orphans_and_bounds_cleanup_without_reusing_numbers() -> None:
    registry = ToolCallRegistry()
    orphan_key = ToolCallKey("child", "run", "turn", "missing")

    orphan = registry.result(orphan_key, "read_file")

    assert orphan.orphan
    assert orphan.call.marker == "#001"
    registry.complete(orphan_key)
    registry.start(ToolCallKey("child", "run", "turn", "one"), "shell")
    registry.start(ToolCallKey("child", "run", "turn", "two"), "shell")
    registry.discard_turn(agent_id="child", run_id="run", turn_id="turn")
    assert registry.active_count == 0
    assert registry.next_ordinal == 4


def test_deferred_registry_ownership_survives_turn_cleanup_until_delivery() -> None:
    registry = ToolCallRegistry()
    key = ToolCallKey("main", "run", "turn", "provider")
    call = registry.start(key, "shell")

    assert registry.defer(key) == call
    registry.discard_turn(agent_id="main", run_id="run", turn_id="turn")
    assert registry.active_count == 1

    result = registry.take_deferred(key, "shell")
    assert result is not None
    assert result.call.marker == "#001"
    assert registry.active_count == 0
    assert registry.take_deferred(key, "shell") is None


def test_registry_markers_expand_after_999() -> None:
    registry = ToolCallRegistry()
    last_marker = ""
    for ordinal in range(1, 1001):
        key = ToolCallKey("main", "run", f"turn-{ordinal}", "provider")
        call = registry.start(key, "shell")
        registry.complete(key)
        last_marker = call.marker

    assert last_marker == "#1000"
    assert registry.active_count == 0


def test_badges_have_exact_plain_powerline_and_no_color_forms() -> None:
    assert (
        tool_badge(
            ToolBadgeKind.CALL,
            "write_file",
            "#001",
            powerline=False,
            theme=DEFAULT_THEME,
        )
        == "\033[1m\033[36m▶ write_file #001\033[0m"
    )
    assert (
        tool_badge(
            ToolBadgeKind.SUCCESS,
            "write_file",
            "#001",
            powerline=True,
            theme=DEFAULT_THEME,
        )
        == "\033[1;30;42m ✓ write_file #001 \033[22;32;49m\ue0b0\033[0m"
    )
    assert (
        tool_badge(
            ToolBadgeKind.ERROR,
            "write_file",
            "#001",
            powerline=False,
            theme=DEFAULT_THEME,
        )
        == "\033[31m✗ write_file #001\033[0m"
    )
    assert (
        tool_badge(
            ToolBadgeKind.SUCCESS,
            "write_file",
            "#001",
            powerline=True,
            theme=NO_COLOR_THEME,
        )
        == "✓ write_file #001"
    )


def test_badge_sanitizes_untrusted_tool_name_and_owns_reset() -> None:
    badge = tool_badge(
        ToolBadgeKind.CALL,
        "shell\nowned\033[2J",
        "#001",
        powerline=False,
        theme=DEFAULT_THEME,
    )

    assert badge == "\033[1m\033[36m▶ shell owned #001\033[0m"
    assert "\033[2J" not in badge


def test_badge_bounds_untrusted_tool_name_by_utf8_bytes() -> None:
    badge = tool_badge(
        ToolBadgeKind.CALL,
        "λ" * 10_000,
        "#001",
        powerline=False,
        theme=NO_COLOR_THEME,
    )

    assert badge.startswith("▶ ") and badge.endswith("… #001")
    assert len(badge.encode("utf-8")) <= 90


def test_badge_replaces_lone_surrogates_with_valid_unicode() -> None:
    badge = tool_badge(
        ToolBadgeKind.ERROR,
        "bad\udcffname",
        "#001",
        powerline=False,
        theme=NO_COLOR_THEME,
    )

    assert badge == "✗ bad�name #001"
    assert badge.encode("utf-8")


def test_shared_tool_display_name_is_one_safe_bounded_utf8_line() -> None:
    name = tool_display_name(("λ" * 10_000) + "\nowned\033[2J\udcff")

    assert "\n" not in name
    assert "\033" not in name
    assert len(name.encode("utf-8")) <= 80
    assert name.endswith("…")


def test_bounded_name_identity_detects_mismatch_after_identical_display_prefix() -> None:
    registry = ToolCallRegistry()
    key = ToolCallKey("main", "run", "turn", "call")
    prefix = "λ" * 100
    call = registry.start(key, prefix + "first")

    result = registry.result(key, prefix + "second")

    assert call.name == result.event_name
    assert len(call.name_identity) == 16
    assert result.name_mismatch
