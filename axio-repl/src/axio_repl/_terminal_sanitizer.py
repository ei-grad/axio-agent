"""Incremental terminal-control filtering for untrusted live streams."""

from __future__ import annotations

from enum import StrEnum


class _State(StrEnum):
    TEXT = "text"
    ESCAPE = "escape"
    CSI = "csi"
    CONTROL_STRING = "control-string"
    CONTROL_STRING_ESCAPE = "control-string-escape"


class IncrementalTerminalSanitizer:
    """Strip terminal controls even when their byte sequence spans chunks."""

    def __init__(self, *, max_control_chars: int = 4096) -> None:
        if max_control_chars <= 0:
            raise ValueError("max_control_chars must be positive")
        self._max_control_chars = max_control_chars
        self._state = _State.TEXT
        self._control_chars = 0

    def feed(self, text: str) -> str:
        output: list[str] = []
        for character in text:
            self._feed_character(character, output)
        return "".join(output)

    def reset(self) -> None:
        """Discard an incomplete control sequence at a semantic boundary."""

        self._state = _State.TEXT
        self._control_chars = 0

    def _feed_character(self, character: str, output: list[str]) -> None:
        codepoint = ord(character)
        if self._state is _State.TEXT:
            if character == "\x1b":
                self._start(_State.ESCAPE)
            elif codepoint == 0x9B:
                self._start(_State.CSI)
            elif codepoint in {0x90, 0x9D, 0x9E, 0x9F}:
                self._start(_State.CONTROL_STRING)
            elif codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
                if character in {"\n", "\t"}:
                    output.append(character)
            else:
                output.append(character)
            return

        self._control_chars += 1
        if self._control_chars > self._max_control_chars:
            self.reset()
            return

        if self._state is _State.ESCAPE:
            if character == "[":
                self._state = _State.CSI
            elif character in {"]", "P", "^", "_"}:
                self._state = _State.CONTROL_STRING
            else:
                self.reset()
            return

        if self._state is _State.CSI:
            if 0x40 <= codepoint <= 0x7E:
                self.reset()
            return

        if self._state is _State.CONTROL_STRING_ESCAPE:
            if character == "\\":
                self.reset()
            elif character != "\x1b":
                self._state = _State.CONTROL_STRING
            return

        if character in {"\x07", "\x9c"}:
            self.reset()
        elif character == "\x1b":
            self._state = _State.CONTROL_STRING_ESCAPE

    def _start(self, state: _State) -> None:
        self._state = state
        self._control_chars = 1


def sanitize_terminal_text(value: object) -> str:
    """Strip complete or unterminated controls from one bounded value."""

    sanitizer = IncrementalTerminalSanitizer()
    return sanitizer.feed(str(value))
