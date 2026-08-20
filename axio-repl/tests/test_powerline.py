import pytest

from axio_repl._powerline import (
    RESET,
    PowerlineBadge,
    action_frame_footer,
    action_frame_header,
    agent_header,
    tool_title,
)


def test_single_segment_badges_fill_between_both_outer_separators() -> None:
    assert tool_title("shell") == (f"\033[22;36;49m\033[1;30;46m ▶ shell \033[22;36;49m{RESET}")
    assert agent_header("main") == (f"\033[22;35;49m\033[1;97;45m agent main \033[22;35;49m{RESET}")
    assert action_frame_footer("child") == (f"\033[22;35;49m\033[1;97;45m /agent child \033[22;35;49m{RESET}")


def test_multi_segment_badge_transitions_between_continuous_fills() -> None:
    assert action_frame_header("child", "tool output") == (
        f"\033[22;35;49m\033[1;97;45m agent child \033[22;35;43m\033[1;30;43m tool output \033[22;33;49m{RESET}"
    )


def test_badge_rejects_an_empty_segment_list() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        PowerlineBadge(())
