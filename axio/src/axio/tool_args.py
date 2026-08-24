"""Incremental streaming parser for tool call JSON arguments.

Feeds partial JSON chunks (from ``ToolInputDelta.partial_json``) and emits
structured ``ToolField*`` events as top-level object fields are discovered.

Top-level *string* values are decoded (escape sequences resolved, quotes
stripped). All other top-level values are emitted as raw JSON fragments. This
is a best-effort presentation parser: agent execution separately retains every
raw ``ToolInputDelta`` fragment for strict final JSON validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType

from .events import ToolFieldDelta, ToolFieldEnd, ToolFieldStart

type ToolFieldEvent = ToolFieldStart | ToolFieldDelta | ToolFieldEnd


class State(IntEnum):
    INIT = 0
    OBJ = 1
    KEY = 2
    COLON = 3
    VAL = 4
    STR = 5
    RAW = 6
    AFTER = 7
    ESC = 8
    UESC = 9


ESCAPES: Mapping[str, str] = MappingProxyType(
    {
        "n": "\n",
        "t": "\t",
        "r": "\r",
        "b": "\b",
        "f": "\f",
        '"': '"',
        "\\": "\\",
        "/": "/",
    }
)

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_REPLACEMENT = "\ufffd"


class ToolArgStream:
    """O(1)-per-character streaming parser for tool argument JSON.

    Usage::

        stream = ToolArgStream("call_1")
        events = stream.feed('{"path":"/tmp/f')
        # [ToolFieldStart(0, "call_1", "path"), ToolFieldDelta(0, "call_1", "path", "/tmp/f")]
        events = stream.feed('oo.py"}')
        # [ToolFieldDelta(0, "call_1", "path", "oo.py"), ToolFieldEnd(0, "call_1", "path")]
    """

    __slots__ = (
        "_id",
        "_idx",
        "_st",
        "_key_chars",
        "_key",
        "_buf",
        "_u",
        "_high",
        "_depth",
        "_raw_str",
        "_raw_esc",
        "_esc_key",
        "_esc_ret",
        "_events",
        "_done",
    )

    def __init__(self, tool_use_id: str, index: int = 0) -> None:
        self._id = tool_use_id
        self._idx = index
        self._st = State.INIT
        self._key_chars: list[str] = []
        self._key = ""
        self._buf: list[str] = []
        self._u: list[str] = []
        self._high = 0
        self._depth = 0
        self._raw_str = False
        self._raw_esc = False
        self._esc_key = False
        self._esc_ret = State.KEY
        self._events: list[ToolFieldEvent] = []
        self._done = False

    @property
    def current_key(self) -> str:
        """The field currently being streamed, or ``""``."""
        return self._key

    @property
    def done(self) -> bool:
        """Whether the top-level JSON object has been fully parsed."""
        return self._done

    def feed(self, chunk: str) -> list[ToolFieldEvent]:
        """Process a partial JSON chunk and return any field events produced."""
        self._events = []
        for ch in chunk:
            self._step(ch)
        self._flush()
        return self._events

    def finish(self) -> list[ToolFieldEvent]:
        """Finish an interrupted preview without emitting invalid Unicode."""

        self._events = []
        if self._st is State.UESC:
            self._flush_high_surrogate()
            self._append_decoded(_REPLACEMENT)
            self._u.clear()
            self._st = self._esc_ret
        elif self._st is State.ESC:
            self._flush_high_surrogate()
            self._st = self._esc_ret
        else:
            self._flush_high_surrogate()
        self._flush()
        return self._events

    def _flush(self) -> None:
        if self._buf:
            self._events.append(ToolFieldDelta(self._idx, self._id, self._key, "".join(self._buf)))
            self._buf.clear()

    def _start(self) -> None:
        self._flush()
        self._events.append(ToolFieldStart(self._idx, self._id, self._key))

    def _end(self) -> None:
        self._flush_high_surrogate()
        self._flush()
        self._events.append(ToolFieldEnd(self._idx, self._id, self._key))

    def _append_decoded(self, text: str) -> None:
        target = self._key_chars if self._esc_key else self._buf
        target.append(text)

    def _flush_high_surrogate(self) -> None:
        if self._high:
            self._append_decoded(_REPLACEMENT)
            self._high = 0

    def _append_unicode_code_unit(self, code: int) -> None:
        if self._high:
            if 0xDC00 <= code <= 0xDFFF:
                full = 0x10000 + (self._high - 0xD800) * 0x400 + (code - 0xDC00)
                self._append_decoded(chr(full))
                self._high = 0
                return
            self._append_decoded(_REPLACEMENT)
            self._high = 0

        if 0xD800 <= code <= 0xDBFF:
            self._high = code
        elif 0xDC00 <= code <= 0xDFFF:
            self._append_decoded(_REPLACEMENT)
        else:
            self._append_decoded(chr(code))

    def _append_literal_character(self, ch: str) -> None:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            self._append_unicode_code_unit(code)
            return
        self._flush_high_surrogate()
        self._append_decoded(ch)

    @staticmethod
    def _safe_raw_character(ch: str) -> str:
        return _REPLACEMENT if 0xD800 <= ord(ch) <= 0xDFFF else ch

    def _step(self, ch: str) -> None:  # noqa: PLR0912
        match self._st:
            case State.INIT:
                if ch == "{":
                    self._st = State.OBJ

            case State.OBJ:
                if ch == '"':
                    self._key_chars.clear()
                    self._esc_key = True
                    self._st = State.KEY
                elif ch == "}":
                    self._done = True
                    self._st = State.INIT

            case State.KEY:
                if ch == "\\":
                    self._esc_key = True
                    self._esc_ret = State.KEY
                    self._st = State.ESC
                elif ch == '"':
                    self._flush_high_surrogate()
                    self._key = "".join(self._key_chars)
                    self._st = State.COLON
                else:
                    self._append_literal_character(ch)

            case State.COLON:
                if ch == ":":
                    self._start()
                    self._st = State.VAL

            case State.VAL:
                if ch in " \t\r\n":
                    pass
                elif ch == '"':
                    self._esc_key = False
                    self._st = State.STR
                else:
                    self._buf.append(self._safe_raw_character(ch))
                    self._depth = 1 if ch in "{[" else 0
                    self._raw_str = False
                    self._raw_esc = False
                    self._st = State.RAW

            case State.STR:
                if ch == "\\":
                    self._esc_key = False
                    self._esc_ret = State.STR
                    self._st = State.ESC
                elif ch == '"':
                    self._end()
                    self._st = State.AFTER
                else:
                    self._append_literal_character(ch)

            case State.RAW:
                if self._raw_str:
                    self._buf.append(self._safe_raw_character(ch))
                    if self._raw_esc:
                        self._raw_esc = False
                    elif ch == "\\":
                        self._raw_esc = True
                    elif ch == '"':
                        self._raw_str = False
                elif self._depth == 0 and ch in " \t\r\n,}":
                    # simple value (number/bool/null) ends on whitespace or delimiter
                    self._end()
                    self._st = State.AFTER
                    if ch in ",}":
                        self._step(ch)  # reprocess delimiter
                elif ch == '"':
                    self._buf.append(ch)
                    self._raw_str = True
                elif ch in "{[":
                    self._buf.append(ch)
                    self._depth += 1
                elif ch in "}]":
                    self._buf.append(ch)
                    self._depth -= 1
                    if self._depth == 0:
                        self._end()
                        self._st = State.AFTER
                else:
                    self._buf.append(self._safe_raw_character(ch))

            case State.AFTER:
                if ch == ",":
                    self._st = State.OBJ
                elif ch == "}":
                    self._done = True
                    self._st = State.INIT

            case State.ESC:
                if ch == "u":
                    self._u.clear()
                    self._st = State.UESC
                else:
                    self._flush_high_surrogate()
                    dec = ESCAPES.get(ch, ch)
                    self._append_decoded(self._safe_raw_character(dec))
                    self._st = self._esc_ret

            case State.UESC:
                if ch not in _HEX_DIGITS:
                    self._flush_high_surrogate()
                    self._append_decoded(_REPLACEMENT)
                    self._u.clear()
                    self._st = self._esc_ret
                    self._step(ch)
                    return
                self._u.append(ch)
                if len(self._u) == 4:
                    code = int("".join(self._u), 16)
                    self._append_unicode_code_unit(code)
                    self._st = self._esc_ret
