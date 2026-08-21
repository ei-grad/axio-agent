"""Versioned binary replay logs for interactive REPL sessions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import struct
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.output import Output

SCHEMA_VERSION = 1
REPLAY_FILENAME = "replay.axrp"
MAGIC = b"AXIOREPLAY\x00\x01"
MAX_PENDING_BYTES = 8 * 1024 * 1024
MAX_FRAME_BYTES = 64 * 1024 * 1024

_OUTPUT_OPERATIONS = frozenset(
    {
        "set_title",
        "clear_title",
        "erase_screen",
        "enter_alternate_screen",
        "quit_alternate_screen",
        "enable_mouse_support",
        "disable_mouse_support",
        "erase_end_of_line",
        "erase_down",
        "reset_attributes",
        "set_attributes",
        "disable_autowrap",
        "enable_autowrap",
        "cursor_goto",
        "cursor_up",
        "cursor_down",
        "cursor_forward",
        "cursor_backward",
        "hide_cursor",
        "show_cursor",
        "set_cursor_shape",
        "reset_cursor_shape",
        "ask_for_cpr",
        "bell",
        "enable_bracketed_paste",
        "disable_bracketed_paste",
        "reset_cursor_key_mode",
        "scroll_buffer_to_prompt",
    }
)
_FRAME_OPERATIONS = frozenset({"write", "write_raw", *_OUTPUT_OPERATIONS})
_REPLAY_KINDS = frozenset(
    {
        "session_start",
        "session_end",
        "terminal_geometry",
        "terminal_frame",
        "terminal_fallback",
        "key_press",
        "editor_state",
        "input_submission",
        "runtime_event",
    }
)

_FRAME_HEADER = struct.Struct(">II")

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type DegradedCallback = Callable[[BaseException], None]


class ReplayCorruptionError(ValueError):
    """A complete replay frame is malformed or violates the schema."""


class ReplayQueueFullError(RuntimeError):
    """The replay writer could not retain another terminal/input frame."""


class ReplaySchemaError(ValueError):
    """A producer supplied a replay kind or payload outside schema v1."""


@dataclass(frozen=True, slots=True)
class ReplayReadResult:
    """Decoded replay frames and any ignored final incomplete frame."""

    records: tuple[dict[str, JsonValue], ...]
    discarded_tail_bytes: int


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, BaseException):
        return {
            "exception_type": type(value).__name__,
            "message": str(value),
        }
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return [_json_value(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        binary = bytes(value)
        return {
            "type": "binary_reference",
            "sha256": hashlib.sha256(binary).hexdigest(),
            "size": len(binary),
        }
    raise ReplaySchemaError(f"unsupported replay value type: {type(value).__name__}")


def _mapping_payload(payload: object, kind: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise ReplaySchemaError(f"{kind} payload must be an object with string keys")
    return cast(Mapping[str, object], payload)


def _require_keys(
    payload: Mapping[str, object],
    kind: str,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = required.difference(payload)
    extra = set(payload).difference(required, optional)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unknown {', '.join(sorted(extra))}")
        raise ReplaySchemaError(f"{kind} payload has {'; '.join(details)}")


def _required_string(payload: Mapping[str, object], kind: str, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ReplaySchemaError(f"{kind}.{key} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, object], kind: str, key: str) -> None:
    value = payload.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ReplaySchemaError(f"{kind}.{key} must be null or a non-empty string")


def _required_nonnegative_int(payload: Mapping[str, object], kind: str, key: str, *, positive: bool = False) -> int:
    value = payload.get(key)
    lower_bound = 1 if positive else 0
    if type(value) is not int or value < lower_bound:
        qualifier = "positive" if positive else "non-negative"
        raise ReplaySchemaError(f"{kind}.{key} must be a {qualifier} integer")
    return value


def _validate_terminal_frame(payload: Mapping[str, object]) -> None:
    _require_keys(payload, "terminal_frame", required=frozenset({"operations"}))
    operations = payload.get("operations")
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes, bytearray)) or not operations:
        raise ReplaySchemaError("terminal_frame.operations must be a non-empty list")
    for index, raw_operation in enumerate(operations):
        if not isinstance(raw_operation, Mapping) or not all(isinstance(key, str) for key in raw_operation):
            raise ReplaySchemaError(f"terminal_frame.operations[{index}] must be an object")
        operation = cast(Mapping[str, object], raw_operation)
        _require_keys(
            operation,
            f"terminal_frame.operations[{index}]",
            required=frozenset({"op"}),
            optional=frozenset({"args", "kwargs"}),
        )
        name = _required_string(operation, f"terminal_frame.operations[{index}]", "op")
        if name not in _FRAME_OPERATIONS:
            raise ReplaySchemaError(f"terminal_frame.operations[{index}].op is unsupported: {name}")
        args = operation.get("args", [])
        if not isinstance(args, Sequence) or isinstance(args, (str, bytes, bytearray)):
            raise ReplaySchemaError(f"terminal_frame.operations[{index}].args must be a list")
        kwargs = operation.get("kwargs", {})
        if not isinstance(kwargs, Mapping) or not all(isinstance(key, str) for key in kwargs):
            raise ReplaySchemaError(f"terminal_frame.operations[{index}].kwargs must be an object")


def _validate_payload(kind: str, payload: object) -> None:
    if kind not in _REPLAY_KINDS:
        raise ReplaySchemaError(f"unsupported replay kind: {kind}")
    value = _mapping_payload(payload, kind)
    if kind == "session_start":
        _require_keys(
            value,
            kind,
            required=frozenset({"application", "version", "cwd", "mode"}),
        )
        _required_string(value, kind, "application")
        _required_string(value, kind, "version")
        cwd = value.get("cwd")
        if not isinstance(cwd, (str, Path)) or not str(cwd):
            raise ReplaySchemaError("session_start.cwd must be a non-empty path")
        if _required_string(value, kind, "mode") != "interactive":
            raise ReplaySchemaError("session_start.mode must be interactive")
    elif kind == "session_end":
        _require_keys(value, kind, required=frozenset({"status"}), optional=frozenset({"exception"}))
        _required_string(value, kind, "status")
    elif kind == "terminal_geometry":
        _require_keys(value, kind, required=frozenset({"rows", "columns", "source"}))
        _required_nonnegative_int(value, kind, "rows", positive=True)
        _required_nonnegative_int(value, kind, "columns", positive=True)
        if _required_string(value, kind, "source") not in {"initial", "resize"}:
            raise ReplaySchemaError("terminal_geometry.source must be initial or resize")
    elif kind == "terminal_frame":
        _validate_terminal_frame(value)
    elif kind == "terminal_fallback":
        _require_keys(value, kind, required=frozenset({"content", "stream", "destination"}))
        _required_string(value, kind, "content")
        if _required_string(value, kind, "stream") not in {"stdout", "stderr"}:
            raise ReplaySchemaError("terminal_fallback.stream must be stdout or stderr")
        if _required_string(value, kind, "destination") not in {"fallback", "late"}:
            raise ReplaySchemaError("terminal_fallback.destination must be fallback or late")
    elif kind == "key_press":
        _require_keys(value, kind, required=frozenset({"key", "data"}))
        _required_string(value, kind, "key")
        if not isinstance(value.get("data"), str):
            raise ReplaySchemaError("key_press.data must be a string")
    elif kind == "editor_state":
        _require_keys(value, kind, required=frozenset({"text", "cursor_position"}))
        if not isinstance(value.get("text"), str):
            raise ReplaySchemaError("editor_state.text must be a string")
        _required_nonnegative_int(value, kind, "cursor_position")
    elif kind == "input_submission":
        _require_keys(
            value,
            kind,
            required=frozenset({"text", "target_agent_id", "disposition", "input_id", "arrival_seq"}),
        )
        _required_string(value, kind, "text")
        _required_string(value, kind, "target_agent_id")
        if _required_string(value, kind, "disposition") not in {"pending", "command", "retained"}:
            raise ReplaySchemaError("input_submission.disposition is invalid")
        _optional_string(value, kind, "input_id")
        arrival_seq = value.get("arrival_seq")
        if arrival_seq is not None and (type(arrival_seq) is not int or arrival_seq < 1):
            raise ReplaySchemaError("input_submission.arrival_seq must be null or positive")
    else:
        _require_keys(
            value,
            kind,
            required=frozenset(
                {
                    "hub_seq",
                    "run_id",
                    "agent_id",
                    "parent_agent_id",
                    "turn_id",
                    "context_id",
                    "execution_mode",
                    "parent_tool_use_id",
                    "kind",
                    "payload",
                }
            ),
        )
        _required_nonnegative_int(value, kind, "hub_seq", positive=True)
        _required_string(value, kind, "run_id")
        _required_string(value, kind, "agent_id")
        for key in ("parent_agent_id", "turn_id", "context_id", "parent_tool_use_id"):
            _optional_string(value, kind, key)
        if _required_string(value, kind, "execution_mode") not in {"foreground", "background"}:
            raise ReplaySchemaError("runtime_event.execution_mode is invalid")
        _required_string(value, kind, "kind")
        _mapping_payload(value.get("payload"), "runtime_event.payload")


def _encode_record(record: Mapping[str, JsonValue]) -> bytes:
    raw = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if not 0 < len(raw) <= MAX_FRAME_BYTES:
        raise ReplaySchemaError(f"replay frame is outside the {MAX_FRAME_BYTES}-byte uncompressed limit")
    compressed = zlib.compress(raw, level=6)
    if not 0 < len(compressed) <= MAX_FRAME_BYTES:
        raise ReplaySchemaError(f"replay frame is outside the {MAX_FRAME_BYTES}-byte compressed limit")
    return _FRAME_HEADER.pack(len(compressed), len(raw)) + compressed


def _write_all(file_descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written == 0:
            raise OSError("replay write returned zero bytes")
        remaining = remaining[written:]


def _sync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    file_descriptor = os.open(directory, flags)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def read_replay(replay_path: Path) -> ReplayReadResult:
    """Decode a replay, ignoring only one incomplete final frame."""

    raw = replay_path.read_bytes()
    if not raw.startswith(MAGIC):
        raise ReplayCorruptionError("replay header or schema version is invalid")
    offset = len(MAGIC)
    records: list[dict[str, JsonValue]] = []
    session_id: str | None = None
    previous_offset_ns = -1
    saw_initial_geometry = False
    saw_session_end = False
    while offset < len(raw):
        frame_start = offset
        if len(raw) - offset < _FRAME_HEADER.size:
            return ReplayReadResult(tuple(records), len(raw) - frame_start)
        compressed_size, uncompressed_size = _FRAME_HEADER.unpack_from(raw, offset)
        offset += _FRAME_HEADER.size
        if not 0 < compressed_size <= MAX_FRAME_BYTES or not 0 < uncompressed_size <= MAX_FRAME_BYTES:
            raise ReplayCorruptionError(f"invalid replay frame size at frame {len(records) + 1}")
        frame_end = offset + compressed_size
        if frame_end > len(raw):
            return ReplayReadResult(tuple(records), len(raw) - frame_start)
        compressed = raw[offset:frame_end]
        offset = frame_end
        try:
            decompressor = zlib.decompressobj()
            payload = decompressor.decompress(compressed, uncompressed_size + 1)
        except zlib.error as error:
            raise ReplayCorruptionError(f"invalid compressed replay frame {len(records) + 1}: {error}") from error
        if (
            len(payload) != uncompressed_size
            or not decompressor.eof
            or decompressor.unconsumed_tail
            or decompressor.unused_data
        ):
            raise ReplayCorruptionError(f"replay frame {len(records) + 1} has an invalid uncompressed size")
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayCorruptionError(f"invalid JSON in replay frame {len(records) + 1}: {error}") from error
        if not isinstance(decoded, dict):
            raise ReplayCorruptionError(f"replay frame {len(records) + 1} is not an object")
        record = cast(dict[str, JsonValue], decoded)
        expected_seq = len(records) + 1
        record_schema = record.get("schema_version")
        record_seq = record.get("seq")
        if (
            type(record_schema) is not int
            or record_schema != SCHEMA_VERSION
            or type(record_seq) is not int
            or record_seq != expected_seq
        ):
            raise ReplayCorruptionError(f"invalid schema or sequence at replay frame {expected_seq}")
        record_session_id = record.get("session_id")
        if not isinstance(record_session_id, str) or not record_session_id:
            raise ReplayCorruptionError(f"invalid session id at replay frame {expected_seq}")
        if session_id is None:
            session_id = record_session_id
        elif record_session_id != session_id:
            raise ReplayCorruptionError(f"session id changed at replay frame {expected_seq}")
        record_kind = record.get("kind")
        if not isinstance(record_kind, str) or not record_kind:
            raise ReplayCorruptionError(f"invalid kind at replay frame {expected_seq}")
        if expected_seq == 1 and record_kind != "session_start":
            raise ReplayCorruptionError("first replay frame is not session_start")
        if expected_seq > 1 and record_kind == "session_start":
            raise ReplayCorruptionError("replay has a second session_start")
        if saw_session_end:
            raise ReplayCorruptionError("replay has a frame after session_end")
        if record_kind == "session_end":
            saw_session_end = True
        if "payload" not in record:
            raise ReplayCorruptionError(f"missing payload at replay frame {expected_seq}")
        try:
            _validate_payload(record_kind, record["payload"])
        except ReplaySchemaError as error:
            raise ReplayCorruptionError(f"invalid payload at replay frame {expected_seq}: {error}") from error
        if record_kind == "terminal_geometry":
            geometry = cast(dict[str, JsonValue], record["payload"])
            if geometry.get("source") == "initial":
                if saw_initial_geometry:
                    raise ReplayCorruptionError("replay has duplicate initial terminal geometry")
                saw_initial_geometry = True
            elif not saw_initial_geometry:
                raise ReplayCorruptionError("terminal resize precedes initial geometry")
        elif (
            record_kind
            in {
                "terminal_frame",
                "terminal_fallback",
                "key_press",
                "editor_state",
                "input_submission",
            }
            and not saw_initial_geometry
        ):
            raise ReplayCorruptionError(f"{record_kind} precedes initial terminal geometry")
        record_offset_ns = record.get("offset_ns")
        if type(record_offset_ns) is not int or record_offset_ns < previous_offset_ns:
            raise ReplayCorruptionError(f"invalid monotonic offset at replay frame {expected_seq}")
        previous_offset_ns = record_offset_ns
        records.append(record)
    return ReplayReadResult(tuple(records), 0)


class ReplayLog:
    """Thread-safe, bounded producer with one asynchronous binary writer."""

    def __init__(
        self,
        *,
        replay_path: Path,
        file_descriptor: int,
        session_id: str,
        started_ns: int,
        loop: asyncio.AbstractEventLoop,
        max_pending_bytes: int,
        on_degraded: DegradedCallback | None,
    ) -> None:
        self.replay_path = replay_path
        self.session_id = session_id
        self._file_descriptor = file_descriptor
        self._started_ns = started_ns
        self._loop = loop
        self._max_pending_bytes = max_pending_bytes
        self._on_degraded = on_degraded
        self._lock = threading.RLock()
        self._pending: deque[bytes] = deque()
        self._pending_bytes = 0
        self._next_seq = 2
        self._accepting = True
        self._closing = False
        self._initial_geometry_recorded = False
        self._wake = asyncio.Event()
        self._degraded_reason: BaseException | None = None
        self._writer_task = asyncio.create_task(self._writer_loop(), name=f"axio-replay-{session_id}")

    @classmethod
    async def open(
        cls,
        *,
        session_dir: Path,
        session_id: str,
        start_payload: object,
        max_pending_bytes: int = MAX_PENDING_BYTES,
        on_degraded: DegradedCallback | None = None,
    ) -> ReplayLog:
        if max_pending_bytes < 1:
            raise ValueError("max_pending_bytes must be positive")
        _validate_payload("session_start", start_payload)
        replay_path = session_dir / REPLAY_FILENAME
        file_descriptor = os.open(replay_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        started_ns = time.monotonic_ns()
        try:
            os.fchmod(file_descriptor, 0o600)
            _write_all(file_descriptor, MAGIC)
            start: dict[str, JsonValue] = {
                "schema_version": SCHEMA_VERSION,
                "seq": 1,
                "offset_ns": 0,
                "session_id": session_id,
                "kind": "session_start",
                "payload": _json_value(start_payload),
            }
            _write_all(file_descriptor, _encode_record(start))
            os.fsync(file_descriptor)
            _sync_directory(session_dir)
        except BaseException:
            # Opening is a resource transaction; cancellation must close the
            # descriptor just like a serialization or filesystem failure.
            os.close(file_descriptor)
            raise
        return cls(
            replay_path=replay_path,
            file_descriptor=file_descriptor,
            session_id=session_id,
            started_ns=started_ns,
            loop=asyncio.get_running_loop(),
            max_pending_bytes=max_pending_bytes,
            on_degraded=on_degraded,
        )

    @property
    def degraded_reason(self) -> BaseException | None:
        with self._lock:
            return self._degraded_reason

    def record(self, kind: str, payload: object = None) -> bool:
        """Admit a replay event without blocking the UI or a logging thread."""

        if not kind:
            raise ValueError("replay kind must not be empty")
        with self._lock:
            if not self._accepting or self._degraded_reason is not None:
                return False
            geometry_source: str | None = None
            try:
                if kind in {"session_start", "session_end"}:
                    raise ReplaySchemaError(f"{kind} is owned by the replay lifecycle")
                if kind == "terminal_geometry":
                    geometry = _mapping_payload(payload, kind)
                    geometry_source = cast(str, geometry.get("source"))
                    if geometry_source == "initial" and self._initial_geometry_recorded:
                        raise ReplaySchemaError("duplicate initial terminal geometry")
                    if geometry_source == "resize" and not self._initial_geometry_recorded:
                        raise ReplaySchemaError("terminal resize precedes initial geometry")
                elif (
                    kind
                    in {
                        "terminal_frame",
                        "terminal_fallback",
                        "key_press",
                        "editor_state",
                        "input_submission",
                    }
                    and not self._initial_geometry_recorded
                ):
                    raise ReplaySchemaError(f"{kind} precedes initial terminal geometry")
                raw = self._make_record_locked(kind, payload)
            except Exception as error:
                # Frontend payloads can contain provider/plugin objects with
                # serialization behavior outside the replay layer's control.
                self._mark_degraded_locked(error)
                return False
            if self._pending_bytes + len(raw) > self._max_pending_bytes:
                self._mark_degraded_locked(
                    ReplayQueueFullError(f"replay queue reached its {self._max_pending_bytes}-byte limit")
                )
                return False
            should_wake = not self._pending
            self._pending.append(raw)
            self._pending_bytes += len(raw)
            if geometry_source == "initial":
                self._initial_geometry_recorded = True
        if should_wake:
            self._loop.call_soon_threadsafe(self._wake.set)
        return True

    async def close(self, end_payload: object = None) -> None:
        if end_payload is None:
            end_payload = {"status": "complete"}
        with self._lock:
            if self._closing:
                writer = self._writer_task
            else:
                self._accepting = False
                if self._degraded_reason is None:
                    try:
                        raw = self._make_record_locked("session_end", end_payload)
                    except Exception as error:
                        # SessionJournal historically accepts arbitrary end
                        # payloads; replay schema failure remains auxiliary.
                        self._mark_degraded_locked(error)
                    else:
                        if self._pending_bytes + len(raw) > self._max_pending_bytes:
                            self._mark_degraded_locked(
                                ReplayQueueFullError(f"replay queue reached its {self._max_pending_bytes}-byte limit")
                            )
                        else:
                            self._pending.append(raw)
                            self._pending_bytes += len(raw)
                self._closing = True
                writer = self._writer_task
        self._wake.set()
        await writer

    async def __aenter__(self) -> ReplayLog:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        payload: object = {"status": "complete"}
        if exc_value is not None:
            payload = {"status": "error", "exception": repr(exc_value)}
        await self.close(payload)

    def _make_record_locked(self, kind: str, payload: object) -> bytes:
        _validate_payload(kind, payload)
        seq = self._next_seq
        self._next_seq += 1
        record: dict[str, JsonValue] = {
            "schema_version": SCHEMA_VERSION,
            "seq": seq,
            "offset_ns": max(0, time.monotonic_ns() - self._started_ns),
            "session_id": self.session_id,
            "kind": kind,
            "payload": _json_value(payload),
        }
        return json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")

    async def _writer_loop(self) -> None:
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                while True:
                    with self._lock:
                        if self._pending:
                            raw = self._pending.popleft()
                            self._pending_bytes -= len(raw)
                        else:
                            raw = None
                            closing = self._closing
                    if raw is None:
                        if closing:
                            return
                        break
                    try:
                        frame = await asyncio.to_thread(
                            _encode_record,
                            cast(dict[str, JsonValue], json.loads(raw)),
                        )
                        await asyncio.to_thread(_write_all, self._file_descriptor, frame)
                    except Exception as error:
                        # Compression, serialization, and filesystem adapters
                        # expose backend-specific exception types.
                        self._mark_degraded(error)
                        with self._lock:
                            self._pending.clear()
                            self._pending_bytes = 0
                            self._closing = True
                        return
        finally:
            try:
                await asyncio.to_thread(os.fsync, self._file_descriptor)
            except OSError as error:
                self._mark_degraded(error)
            try:
                os.close(self._file_descriptor)
            except OSError as error:
                self._mark_degraded(error)

    def _mark_degraded(self, error: BaseException) -> None:
        with self._lock:
            self._mark_degraded_locked(error)

    def _mark_degraded_locked(self, error: BaseException) -> None:
        if self._degraded_reason is not None:
            return
        self._degraded_reason = error
        self._accepting = False
        if self._on_degraded is not None:
            self._loop.call_soon_threadsafe(self._notify_degraded, error)

    def _notify_degraded(self, error: BaseException) -> None:
        assert self._on_degraded is not None
        try:
            self._on_degraded(error)
        except Exception:
            # A UI notification hook must not terminate the replay writer.
            pass


class RecordingInput(Input):
    """Input adapter that records parsed key presses before dispatch."""

    def __init__(self, delegate: Input, replay: ReplayLog) -> None:
        self._delegate = delegate
        self._replay = replay

    def fileno(self) -> int:
        return self._delegate.fileno()

    def typeahead_hash(self) -> str:
        return self._delegate.typeahead_hash()

    def read_keys(self) -> list[KeyPress]:
        keys = self._delegate.read_keys()
        self._record(keys)
        return keys

    def flush_keys(self) -> list[KeyPress]:
        keys = self._delegate.flush_keys()
        self._record(keys)
        return keys

    def flush(self) -> None:
        self._delegate.flush()

    @property
    def closed(self) -> bool:
        return self._delegate.closed

    def raw_mode(self) -> Any:
        return self._delegate.raw_mode()

    def cooked_mode(self) -> Any:
        return self._delegate.cooked_mode()

    def attach(self, input_ready_callback: Callable[[], None]) -> Any:
        return self._delegate.attach(input_ready_callback)

    def detach(self) -> Any:
        return self._delegate.detach()

    def close(self) -> None:
        self._delegate.close()

    def _record(self, keys: list[KeyPress]) -> None:
        for key in keys:
            self._replay.record("key_press", {"key": str(key.key), "data": key.data})


class RecordingOutput:
    """Output proxy that records one ordered operation list per physical flush."""

    def __init__(self, delegate: Output, replay: ReplayLog) -> None:
        self._delegate = delegate
        self._replay = replay
        self._lock = threading.RLock()
        self._operations: list[dict[str, JsonValue]] = []
        initial_size = delegate.get_size()
        self._last_size = (initial_size.rows, initial_size.columns)
        replay.record(
            "terminal_geometry",
            {
                "rows": initial_size.rows,
                "columns": initial_size.columns,
                "source": "initial",
            },
        )

    def write(self, data: str) -> None:
        self._append("write", (data,), {})
        self._delegate.write(data)

    def write_raw(self, data: str) -> None:
        self._append("write_raw", (data,), {})
        self._delegate.write_raw(data)

    def flush(self) -> None:
        self._delegate.flush()
        with self._lock:
            operations = self._operations
            self._operations = []
        if operations:
            self._replay.record("terminal_frame", {"operations": operations})

    def get_size(self) -> Size:
        size = self._delegate.get_size()
        current = (size.rows, size.columns)
        if current != self._last_size:
            self._last_size = current
            self._replay.record(
                "terminal_geometry",
                {
                    "rows": size.rows,
                    "columns": size.columns,
                    "source": "resize",
                },
            )
        return size

    def record_fallback(self, content: str, stream: str, destination: str) -> None:
        self._replay.record(
            "terminal_fallback",
            {
                "content": content,
                "stream": stream,
                "destination": destination,
            },
        )

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._delegate, name)
        if name not in _OUTPUT_OPERATIONS or not callable(attribute):
            return attribute

        def recorded(*args: object, **kwargs: object) -> object:
            self._append(name, args, kwargs)
            return cast(Callable[..., object], attribute)(*args, **kwargs)

        return recorded

    def _append(self, name: str, args: Sequence[object], kwargs: Mapping[str, object]) -> None:
        operation: dict[str, JsonValue] = {"op": name}
        if args:
            operation["args"] = _json_value(args)
        if kwargs:
            operation["kwargs"] = _json_value(kwargs)
        with self._lock:
            self._operations.append(operation)


def recording_input(delegate: Input, replay: ReplayLog | None) -> Input:
    return delegate if replay is None else RecordingInput(delegate, replay)


def recording_output(delegate: Output, replay: ReplayLog | None) -> Output:
    return delegate if replay is None else cast(Output, RecordingOutput(delegate, replay))


def record_terminal_fallback(
    output: Output,
    content: str,
    stream: str,
    destination: str,
) -> None:
    """Record bytes written outside prompt_toolkit's normal output path."""

    if isinstance(output, RecordingOutput):
        output.record_fallback(content, stream, destination)
