from __future__ import annotations

import pytest

from axio_repl._terminal_sanitizer import IncrementalTerminalSanitizer, sanitize_terminal_text


def test_incremental_sanitizer_strips_controls_split_across_chunks() -> None:
    sanitizer = IncrementalTerminalSanitizer()

    assert sanitizer.feed("before\x1b[") == "before"
    assert sanitizer.feed("?1049hafter\x1b]52;c;secret") == "after"
    assert sanitizer.feed("\x07done\x1bPpayload\x1b") == "done"
    assert sanitizer.feed("\\tail") == "tail"


def test_reset_discards_incomplete_sequence_without_poisoning_next_stream() -> None:
    sanitizer = IncrementalTerminalSanitizer()

    assert sanitizer.feed("kept\x1b]") == "kept"
    sanitizer.reset()
    assert sanitizer.feed("ordinary next stream") == "ordinary next stream"


def test_complete_sanitizer_preserves_text_newlines_and_tabs_only() -> None:
    assert sanitize_terminal_text("one\n\ttwo\x00\x1b[2Jthree\x1b]0;title\x07") == "one\n\ttwothree"


def test_incremental_sanitizer_rejects_non_positive_control_bound() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        IncrementalTerminalSanitizer(max_control_chars=0)
