"""Durable, append-only JSONL journals for REPL sessions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import cast

from axio.events import (
    TextDelta,
    ToolFieldDelta,
    ToolFieldEnd,
    ToolFieldStart,
    ToolInputDelta,
    ToolOutputDelta,
    ToolUseStart,
)

from axio_repl._replay import ReplayLog

LEGACY_SCHEMA_VERSION = 1
SCHEMA_VERSION = 2
SEMANTIC_FILENAME = "session.jsonl"
REDACTED = "[REDACTED]"
CHECKPOINT_INTERVAL_NS = 2_000_000_000
CHECKPOINT_CHARS = 16 * 1024

_LOGGER = logging.getLogger(__name__)
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_AUTH_PATTERN = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9+/._~=-]{6,}")
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret)\b\s*[:=]\s*)"
    r"[^\s,;]+"
)
_PREFIXED_TOKEN_PATTERN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{8,})\b")
_JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type DegradedCallback = Callable[[BaseException], None]


class JournalQueueFullError(RuntimeError):
    """The bounded journal queue could not accept another record."""


class JournalStartError(OSError):
    """The journal could not make its initial record durable."""


class JournalCorruptionError(ValueError):
    """A newline-terminated journal record is not valid JSON."""


@dataclass(frozen=True, slots=True)
class JournalReadResult:
    """The valid JSONL prefix and any ignored unterminated tail."""

    records: tuple[dict[str, JsonValue], ...]
    valid_prefix_bytes: int
    discarded_tail_bytes: int


@dataclass(frozen=True, slots=True)
class _PendingRecord:
    seq: int
    timestamp: str
    monotonic_ns: int
    kind: str
    payload: object
    agent_id: str | None
    parent_agent_id: str | None
    turn_id: str | None
    context_id: str | None
    execution_mode: str | None
    parent_tool_use_id: str | None


@dataclass(frozen=True, slots=True)
class _SyncRequest:
    result: asyncio.Future[bool]


@dataclass(frozen=True, slots=True)
class _CloseRequest:
    pass


type _QueueItem = _PendingRecord | _SyncRequest | _CloseRequest


@dataclass(slots=True)
class _TurnCheckpoint:
    agent_id: str
    parent_agent_id: str | None
    turn_id: str
    context_id: str | None
    execution_mode: str | None
    parent_tool_use_id: str | None
    text: list[str]
    tool_names: dict[str, str]
    tool_arguments: dict[str, list[str]]
    tool_fields: dict[str, dict[str, list[str]]]
    tool_output: dict[str, list[str]]
    pending_chars: int = 0
    last_flush_ns: int | None = None

    @property
    def dirty(self) -> bool:
        return bool(self.text or self.tool_names or self.tool_arguments or self.tool_fields or self.tool_output)


def default_journal_root(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the XDG-compatible root used for session journals."""

    values = os.environ if environ is None else environ
    configured = values.get("XDG_STATE_HOME", "")
    configured_root = Path(configured).expanduser() if configured else None
    if configured_root is not None and configured_root.is_absolute():
        state_home = configured_root
    else:
        state_home = (Path.home() if home is None else home) / ".local" / "state"
    return state_home / "axio" / "sessions"


def session_directory(
    session_id: str,
    *,
    root: Path | None = None,
    started_at: datetime | None = None,
) -> Path:
    """Return the date-partitioned directory for one session."""

    if _SAFE_SESSION_ID.fullmatch(session_id) is None:
        raise ValueError("session_id must contain only letters, digits, dots, underscores, and hyphens")
    if started_at is not None and started_at.tzinfo is None:
        raise ValueError("started_at must be timezone-aware")
    timestamp = datetime.now(UTC) if started_at is None else started_at.astimezone(UTC)
    journal_root = default_journal_root() if root is None else root
    return journal_root / timestamp.strftime("%Y") / timestamp.strftime("%m") / timestamp.strftime("%d") / session_id


def read_journal(events_path: Path) -> JournalReadResult:
    """Read the valid JSONL prefix, ignoring only a final unterminated line.

    A writer crash can interrupt its current ``write(2)`` and leave that one
    line incomplete. Any malformed newline-terminated record is corruption and
    is rejected, including corruption before the final line.
    """

    raw = events_path.read_bytes()
    valid_prefix_bytes = raw.rfind(b"\n") + 1
    records: list[dict[str, JsonValue]] = []
    for line_number, line in enumerate(raw[:valid_prefix_bytes].splitlines(), start=1):
        if not line:
            raise JournalCorruptionError(f"empty record at line {line_number}")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise JournalCorruptionError(f"invalid JSON at line {line_number}: {error}") from error
        if not isinstance(value, dict):
            raise JournalCorruptionError(f"journal record at line {line_number} is not an object")
        records.append(cast(dict[str, JsonValue], value))
    return JournalReadResult(
        records=tuple(records),
        valid_prefix_bytes=valid_prefix_bytes,
        discarded_tail_bytes=len(raw) - valid_prefix_bytes,
    )


def recover_journal_tail(events_path: Path) -> JournalReadResult:
    """Validate a stopped journal and truncate only its unterminated tail."""

    result = read_journal(events_path)
    if result.discarded_tail_bytes == 0:
        return result
    file_descriptor = os.open(events_path, os.O_WRONLY)
    try:
        current_size = os.fstat(file_descriptor).st_size
        expected_size = result.valid_prefix_bytes + result.discarded_tail_bytes
        if current_size != expected_size:
            raise RuntimeError("journal changed while its tail was being recovered")
        os.ftruncate(file_descriptor, result.valid_prefix_bytes)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    _sync_directory(events_path.parent)
    return result


def _is_secret_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    compact = normalized.replace("_", "")
    exact = {
        "apikey",
        "authorization",
        "clientsecret",
        "cookie",
        "credential",
        "idtoken",
        "password",
        "passwd",
        "privatekey",
        "proxyauthorization",
        "refreshtoken",
        "secret",
        "sessiontoken",
        "setcookie",
        "token",
    }
    if compact in exact:
        return True
    parts = normalized.split("_")
    if "password" in parts or "passwd" in parts or "secret" in parts or "credential" in parts:
        return True
    return normalized.endswith(("_api_key", "_access_token", "_refresh_token", "_private_key"))


def _redact_string(value: str) -> str:
    redacted = _PRIVATE_KEY_PATTERN.sub(REDACTED, value)
    redacted = _AUTH_PATTERN.sub(lambda match: f"{match.group(1)} {REDACTED}", redacted)
    redacted = _PREFIXED_TOKEN_PATTERN.sub(REDACTED, redacted)
    redacted = _JWT_PATTERN.sub(REDACTED, redacted)
    return _ASSIGNMENT_PATTERN.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)


def _write_all(file_descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(file_descriptor, view)
        if written == 0:
            raise OSError("journal write returned zero bytes")
        view = view[written:]


def _append_line(file_descriptor: int, line: bytes) -> None:
    _write_all(file_descriptor, line)


def _sync_file(file_descriptor: int) -> None:
    os.fsync(file_descriptor)


def _sync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    file_descriptor = os.open(directory, flags)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _sync_directory_chain(directory: Path, existing_ancestor: Path) -> None:
    current = directory
    while True:
        _sync_directory(current)
        if current == existing_ancestor:
            return
        parent = current.parent
        if parent == current:
            raise OSError(f"{existing_ancestor} is not an ancestor of {directory}")
        current = parent


def _sync_and_close(file_descriptor: int) -> None:
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _open_storage(directory: Path) -> tuple[int, Path, Path]:
    existing_ancestor = directory
    while not existing_ancestor.exists():
        parent = existing_ancestor.parent
        if parent == existing_ancestor:
            break
        existing_ancestor = parent
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    semantic_path = directory / SEMANTIC_FILENAME
    file_descriptor = os.open(semantic_path, os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.fchmod(file_descriptor, 0o600)
    except OSError:
        os.close(file_descriptor)
        raise
    return file_descriptor, semantic_path, existing_ancestor


class _AttachmentStore:
    def __init__(self, session_dir: Path) -> None:
        self._session_dir = session_dir
        self._directory = session_dir / "attachments"
        self._ready = False

    def put(self, data: bytes, media_type: str | None) -> dict[str, JsonValue]:
        digest = hashlib.sha256(data).hexdigest()
        directory_created = False
        if not self._ready:
            self._directory.mkdir(mode=0o700, exist_ok=True)
            self._directory.chmod(0o700)
            self._ready = True
            directory_created = True
        target = self._directory / digest
        try:
            file_descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if target.stat().st_size != len(data):
                raise OSError(f"attachment {digest} exists with an unexpected size") from None
        else:
            try:
                os.fchmod(file_descriptor, 0o600)
                _write_all(file_descriptor, data)
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
            _sync_directory(self._directory)
            if directory_created:
                _sync_directory(self._session_dir)
        return {
            "type": "attachment",
            "sha256": digest,
            "size": len(data),
            "media_type": _redact_string(media_type) if media_type is not None else "application/octet-stream",
            "path": f"attachments/{digest}",
        }


class _JournalSerializer:
    def __init__(self, session_dir: Path) -> None:
        self._attachments = _AttachmentStore(session_dir)

    def convert(self, value: object, *, media_type: str | None = None) -> JsonValue:
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("journal timestamps must be timezone-aware")
            return value.isoformat(timespec="microseconds")
        if isinstance(value, str):
            return _redact_string(value)
        if isinstance(value, Enum):
            return self.convert(value.value)
        if isinstance(value, BaseException):
            return {
                "exception_type": type(value).__name__,
                "message": _redact_string(str(value)),
            }
        if isinstance(value, (bytes, bytearray, memoryview)):
            return self._attachments.put(bytes(value), media_type)
        if isinstance(value, Path):
            return _redact_string(str(value))
        if is_dataclass(value) and not isinstance(value, type):
            result: dict[str, JsonValue] = {"record_type": type(value).__name__}
            object_media_type = getattr(value, "media_type", None)
            for item in fields(value):
                key = item.name
                raw_value = getattr(value, key)
                if _is_secret_key(key):
                    result[key] = REDACTED
                    continue
                field_media_type = object_media_type if key == "data" and isinstance(object_media_type, str) else None
                result[key] = self.convert(raw_value, media_type=field_media_type)
            return result
        if isinstance(value, Mapping):
            result = {}
            raw_media_type = value.get("media_type")
            object_media_type = raw_media_type if isinstance(raw_media_type, str) else None
            for raw_key, raw_value in value.items():
                key = str(raw_key)
                if _is_secret_key(key):
                    result[key] = REDACTED
                    continue
                field_media_type = object_media_type if key == "data" else None
                result[key] = self.convert(raw_value, media_type=field_media_type)
            return result
        if isinstance(value, (set, frozenset)):
            return [self.convert(item) for item in sorted(value, key=repr)]
        if isinstance(value, Sequence):
            return [self.convert(item) for item in value]
        return _redact_string(str(value))


class SessionJournal:
    """A bounded JSONL writer with explicit durability boundaries.

    ``publish()`` only acknowledges admission to the in-memory queue. A
    successful ``sync()`` makes every record accepted before it durable.
    """

    def __init__(
        self,
        *,
        session_id: str,
        session_dir: Path,
        semantic_path: Path,
        file_descriptor: int,
        metadata_sync_root: Path,
        queue_size: int,
        on_degraded: DegradedCallback | None,
    ) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self.semantic_path = semantic_path
        # Kept as a source-compatible alias for journal consumers. New paths use
        # ``session.jsonl`` and old ``events.jsonl`` files remain readable.
        self.events_path = semantic_path
        self.attachments_dir = session_dir / "attachments"
        self._file_descriptor = file_descriptor
        self._metadata_sync_root = metadata_sync_root
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue(maxsize=queue_size)
        self._serializer = _JournalSerializer(session_dir)
        self._on_degraded = on_degraded
        self._degraded_reason: BaseException | None = None
        self._next_seq = 1
        self._accepting = True
        self._closed = False
        self._storage_metadata_synced = False
        self._close_lock = asyncio.Lock()
        self._checkpoints: dict[tuple[str, str], _TurnCheckpoint] = {}
        self._replay: ReplayLog | None = None
        self._replay_start_error: BaseException | None = None
        self._writer_task = asyncio.create_task(self._writer_loop(), name=f"axio-journal-{session_id}")

    @classmethod
    async def open(
        cls,
        *,
        session_id: str | None = None,
        root: Path | None = None,
        started_at: datetime | None = None,
        start_payload: object = None,
        queue_size: int = 4096,
        on_degraded: DegradedCallback | None = None,
        replay: bool = False,
        on_replay_degraded: DegradedCallback | None = None,
    ) -> SessionJournal:
        """Create a semantic journal whose ``session_start`` is durable."""

        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        resolved_session_id = uuid.uuid4().hex if session_id is None else session_id
        resolved_started_at = datetime.now(UTC) if started_at is None else started_at
        directory = session_directory(resolved_session_id, root=root, started_at=resolved_started_at)
        file_descriptor, semantic_path, metadata_sync_root = await asyncio.to_thread(_open_storage, directory)
        journal = cls(
            session_id=resolved_session_id,
            session_dir=directory,
            semantic_path=semantic_path,
            file_descriptor=file_descriptor,
            metadata_sync_root=metadata_sync_root,
            queue_size=queue_size,
            on_degraded=on_degraded,
        )
        if not journal._enqueue_nowait("session_start", start_payload):
            await journal.close()
            raise JournalStartError("session_start could not be queued")
        if not await journal.sync():
            reason = journal.degraded_reason
            await journal.close()
            message = "session_start could not be made durable"
            if reason is not None:
                raise JournalStartError(f"{message}: {type(reason).__name__}: {reason}") from reason
            raise JournalStartError(message)
        if replay:
            try:
                journal._replay = await ReplayLog.open(
                    session_dir=directory,
                    session_id=resolved_session_id,
                    start_payload=start_payload,
                    on_degraded=on_replay_degraded,
                )
            except Exception as error:
                # Replay is an auxiliary opt-in artifact; semantic durability
                # and resume remain available when its storage cannot start.
                journal._replay_start_error = error
                if on_replay_degraded is not None:
                    try:
                        on_replay_degraded(error)
                    except Exception:
                        # Notification is arbitrary UI code and must not leak
                        # the already-running semantic writer task.
                        _LOGGER.error("session replay degraded callback failed")
        return journal

    @property
    def degraded(self) -> bool:
        return self._degraded_reason is not None

    @property
    def degraded_reason(self) -> BaseException | None:
        return self._degraded_reason

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def replay_path(self) -> Path | None:
        return self._replay.replay_path if self._replay is not None else None

    @property
    def replay_log(self) -> ReplayLog | None:
        """Return the optional recorder for frontend adapters."""

        return self._replay

    @property
    def replay_degraded_reason(self) -> BaseException | None:
        if self._replay is not None:
            return self._replay.degraded_reason
        return self._replay_start_error

    def record_replay(self, kind: str, payload: object = None) -> bool:
        """Record one replay-only event when exact replay was enabled."""

        return self._replay is not None and self._replay.record(kind, payload)

    async def publish(
        self,
        kind: str,
        payload: object = None,
        *,
        agent_id: str | None = None,
        parent_agent_id: str | None = None,
        turn_id: str | None = None,
        context_id: str | None = None,
        execution_mode: str | None = None,
        parent_tool_use_id: str | None = None,
    ) -> bool:
        """Queue one record without waiting for serialization or persistence.

        ``True`` means the bounded in-memory queue accepted the record. The
        record is not durable until a later successful ``sync()`` or ``close()``.
        """

        if not kind:
            raise ValueError("kind must not be empty")
        return self._enqueue_nowait(
            kind,
            payload,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            turn_id=turn_id,
            context_id=context_id,
            execution_mode=execution_mode,
            parent_tool_use_id=parent_tool_use_id,
        )

    def observe_stream_event(
        self,
        event: object,
        *,
        agent_id: str,
        parent_agent_id: str | None,
        turn_id: str | None,
        context_id: str | None,
        execution_mode: str | None,
        parent_tool_use_id: str | None,
    ) -> bool:
        """Fold resumable stream fragments into sparse semantic checkpoints.

        Reasoning deltas are intentionally absent: they dominate journal size,
        are not canonical conversation state, and are available in opt-in exact
        replay through the terminal frames the user actually saw.
        """

        if turn_id is None or not isinstance(
            event,
            (TextDelta, ToolUseStart, ToolInputDelta, ToolFieldStart, ToolFieldDelta, ToolFieldEnd, ToolOutputDelta),
        ):
            return True
        key = (agent_id, turn_id)
        checkpoint = self._checkpoints.get(key)
        if checkpoint is None:
            checkpoint = _TurnCheckpoint(
                agent_id=agent_id,
                parent_agent_id=parent_agent_id,
                turn_id=turn_id,
                context_id=context_id,
                execution_mode=execution_mode,
                parent_tool_use_id=parent_tool_use_id,
                text=[],
                tool_names={},
                tool_arguments={},
                tool_fields={},
                tool_output={},
            )
            self._checkpoints[key] = checkpoint
        if isinstance(event, TextDelta):
            checkpoint.text.append(event.delta)
            checkpoint.pending_chars += len(event.delta)
        elif isinstance(event, ToolUseStart):
            checkpoint.tool_names[event.tool_use_id] = event.name
            checkpoint.tool_arguments.setdefault(event.tool_use_id, [])
            checkpoint.pending_chars += len(event.tool_use_id) + len(event.name)
        elif isinstance(event, ToolInputDelta):
            checkpoint.tool_arguments.setdefault(event.tool_use_id, []).append(event.partial_json)
            checkpoint.pending_chars += len(event.partial_json)
        elif isinstance(event, ToolFieldStart):
            checkpoint.tool_fields.setdefault(event.tool_use_id, {}).setdefault(event.key, [])
            checkpoint.pending_chars += len(event.key)
        elif isinstance(event, ToolFieldDelta):
            checkpoint.tool_fields.setdefault(event.tool_use_id, {}).setdefault(event.key, []).append(event.text)
            checkpoint.pending_chars += len(event.text)
        elif isinstance(event, ToolFieldEnd):
            checkpoint.tool_fields.setdefault(event.tool_use_id, {}).setdefault(event.key, [])
        else:
            checkpoint.tool_output.setdefault(event.tool_use_id, []).append(event.delta)
            checkpoint.pending_chars += len(event.delta)

        now = time.monotonic_ns()
        first_fragment = checkpoint.last_flush_ns is None
        interval_elapsed = (
            checkpoint.last_flush_ns is not None and now - checkpoint.last_flush_ns >= CHECKPOINT_INTERVAL_NS
        )
        if first_fragment or interval_elapsed or checkpoint.pending_chars >= CHECKPOINT_CHARS:
            return self._flush_checkpoint_nowait(key, now=now)
        return True

    def flush_checkpoints(self) -> bool:
        """Queue every pending resumable fragment before a durability barrier."""

        accepted = True
        now = time.monotonic_ns()
        for key in tuple(self._checkpoints):
            if not self._flush_checkpoint_nowait(key, now=now):
                accepted = False
        return accepted

    def finish_turn_checkpoint(self, agent_id: str, turn_id: str | None) -> None:
        """Forget accumulator metadata after a completed/cancelled turn."""

        if turn_id is not None:
            self._checkpoints.pop((agent_id, turn_id), None)

    async def sync(self) -> bool:
        """Drain prior accepted records and make their valid JSONL prefix durable."""

        async with self._close_lock:
            if self._closed or not self._accepting:
                return False
            if not self.degraded:
                self.flush_checkpoints()
            result: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            await self._queue.put(_SyncRequest(result))
            return await result

    async def close(self, end_payload: object = None) -> None:
        """Drain accepted records, append ``session_end``, fsync, and close."""

        async with self._close_lock:
            if self._closed:
                return
            if not self.degraded:
                self.flush_checkpoints()
            self._accepting = False
            if not self.degraded:
                record = self._make_record("session_end", end_payload)
                await self._queue.put(record)
                self._next_seq += 1
            await self._queue.put(_CloseRequest())
            await self._writer_task
            try:
                if self._replay is not None:
                    await self._replay.close(end_payload)
            finally:
                self._closed = True

    async def __aenter__(self) -> SessionJournal:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        if exc_value is None:
            await self.close({"status": "ok"})
        else:
            await self.close({"status": "error", "exception": exc_value})

    def _enqueue_nowait(
        self,
        kind: str,
        payload: object,
        *,
        agent_id: str | None = None,
        parent_agent_id: str | None = None,
        turn_id: str | None = None,
        context_id: str | None = None,
        execution_mode: str | None = None,
        parent_tool_use_id: str | None = None,
    ) -> bool:
        if not self._accepting or self.degraded:
            return False
        record = self._make_record(
            kind,
            payload,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            turn_id=turn_id,
            context_id=context_id,
            execution_mode=execution_mode,
            parent_tool_use_id=parent_tool_use_id,
        )
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self._mark_degraded(JournalQueueFullError(f"journal queue reached its {self._queue.maxsize}-record limit"))
            return False
        self._next_seq += 1
        return True

    def _make_record(
        self,
        kind: str,
        payload: object,
        *,
        agent_id: str | None = None,
        parent_agent_id: str | None = None,
        turn_id: str | None = None,
        context_id: str | None = None,
        execution_mode: str | None = None,
        parent_tool_use_id: str | None = None,
    ) -> _PendingRecord:
        return _PendingRecord(
            seq=self._next_seq,
            timestamp=datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            monotonic_ns=time.monotonic_ns(),
            kind=kind,
            payload=payload,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            turn_id=turn_id,
            context_id=context_id,
            execution_mode=execution_mode,
            parent_tool_use_id=parent_tool_use_id,
        )

    def _flush_checkpoint_nowait(self, key: tuple[str, str], *, now: int) -> bool:
        checkpoint = self._checkpoints.get(key)
        if checkpoint is None or not checkpoint.dirty:
            return True
        payload = {
            "text": "".join(checkpoint.text),
            "tool_names": checkpoint.tool_names,
            "tool_arguments": {
                tool_use_id: "".join(parts) for tool_use_id, parts in checkpoint.tool_arguments.items()
            },
            "tool_fields": {
                tool_use_id: {field: "".join(parts) for field, parts in values.items()}
                for tool_use_id, values in checkpoint.tool_fields.items()
            },
            "tool_output": {tool_use_id: "".join(parts) for tool_use_id, parts in checkpoint.tool_output.items()},
        }
        accepted = self._enqueue_nowait(
            "turn_checkpoint",
            payload,
            agent_id=checkpoint.agent_id,
            parent_agent_id=checkpoint.parent_agent_id,
            turn_id=checkpoint.turn_id,
            context_id=checkpoint.context_id,
            execution_mode=checkpoint.execution_mode,
            parent_tool_use_id=checkpoint.parent_tool_use_id,
        )
        if accepted:
            checkpoint.text.clear()
            checkpoint.tool_names = {}
            checkpoint.tool_arguments = {}
            checkpoint.tool_fields = {}
            checkpoint.tool_output = {}
            checkpoint.pending_chars = 0
            checkpoint.last_flush_ns = now
        return accepted

    def _encode_record(self, pending: _PendingRecord) -> bytes:
        record: dict[str, JsonValue] = {
            "schema_version": SCHEMA_VERSION,
            "seq": pending.seq,
            "timestamp": pending.timestamp,
            "monotonic_ns": pending.monotonic_ns,
            "session_id": self.session_id,
            "agent_id": _redact_string(pending.agent_id) if pending.agent_id is not None else None,
            "parent_agent_id": (
                _redact_string(pending.parent_agent_id) if pending.parent_agent_id is not None else None
            ),
            "turn_id": _redact_string(pending.turn_id) if pending.turn_id is not None else None,
            "context_id": _redact_string(pending.context_id) if pending.context_id is not None else None,
            "execution_mode": _redact_string(pending.execution_mode) if pending.execution_mode is not None else None,
            "parent_tool_use_id": (
                _redact_string(pending.parent_tool_use_id) if pending.parent_tool_use_id is not None else None
            ),
            "kind": _redact_string(pending.kind),
            "payload": self._serializer.convert(pending.payload),
        }
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return encoded.encode("utf-8") + b"\n"

    async def _writer_loop(self) -> None:
        try:
            should_close = False
            while not should_close:
                item = await self._queue.get()
                try:
                    if isinstance(item, _PendingRecord):
                        if not self.degraded:
                            try:
                                line = await asyncio.to_thread(self._encode_record, item)
                                await asyncio.to_thread(_append_line, self._file_descriptor, line)
                            except Exception as error:
                                # Serialization and filesystem adapters can raise backend-specific exceptions.
                                self._mark_degraded(error)
                    elif isinstance(item, _SyncRequest):
                        try:
                            await asyncio.to_thread(_sync_file, self._file_descriptor)
                            if not self._storage_metadata_synced:
                                await asyncio.to_thread(
                                    _sync_directory_chain,
                                    self.session_dir,
                                    self._metadata_sync_root,
                                )
                                self._storage_metadata_synced = True
                        except Exception as error:
                            # fsync failures are platform- and filesystem-specific.
                            self._mark_degraded(error)
                        if not item.result.done():
                            item.result.set_result(not self.degraded)
                    else:
                        should_close = True
                finally:
                    self._queue.task_done()
        finally:
            try:
                await asyncio.to_thread(_sync_and_close, self._file_descriptor)
            except Exception as error:
                # Closing must remain non-fatal even when the backing filesystem has failed.
                self._mark_degraded(error)

    def _mark_degraded(self, error: BaseException) -> None:
        if self._degraded_reason is not None:
            return
        self._degraded_reason = error
        if self._on_degraded is not None:
            asyncio.get_running_loop().call_soon(self._notify_degraded, error)

    def _notify_degraded(self, error: BaseException) -> None:
        if self._on_degraded is None:
            return
        try:
            self._on_degraded(error)
        except Exception:
            # A UI notification hook is arbitrary application code and must not fail the journal task.
            _LOGGER.error("session journal degraded callback failed")
