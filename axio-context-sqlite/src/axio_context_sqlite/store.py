"""SQLiteContextStore: persistent conversation storage backed by SQLite."""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import threading
from collections import deque
from collections.abc import AsyncIterator
from concurrent.futures import Future, InvalidStateError
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import aiosqlite
from axio.context import ContextStore, SessionInfo
from axio.messages import Message

# Compress content payloads above this size (bytes of UTF-8 JSON).
COMPRESS_THRESHOLD = 512
_TRANSACTION_LOCK_ATTRIBUTE = "_axio_context_sqlite_transaction_lock"
_TRANSACTION_LOCK_CREATION_GUARD = threading.Lock()


class _ConnectionTransactionLock:
    """An event-loop-neutral lock shared by stores using one connection."""

    def __init__(self) -> None:
        self._state_guard = threading.Lock()
        self._locked = False
        self._waiters: deque[Future[None]] = deque()

    async def acquire(self) -> None:
        waiter: Future[None]
        with self._state_guard:
            if not self._locked:
                self._locked = True
                return
            waiter = Future()
            self._waiters.append(waiter)

        try:
            await asyncio.wrap_future(waiter)
        except BaseException:
            # If ownership was granted concurrently with cancellation, pass it on.
            if not waiter.cancel():
                self.release()
            raise

    def release(self) -> None:
        with self._state_guard:
            if not self._locked:
                raise RuntimeError("transaction lock is not acquired")
            while self._waiters:
                waiter = self._waiters.popleft()
                try:
                    waiter.set_result(None)
                except InvalidStateError:
                    continue
                return
            self._locked = False

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            self.release()


def _transaction_lock_for(conn: aiosqlite.Connection) -> _ConnectionTransactionLock:
    with _TRANSACTION_LOCK_CREATION_GUARD:
        lock = getattr(conn, _TRANSACTION_LOCK_ATTRIBUTE, None)
        if lock is None:
            lock = _ConnectionTransactionLock()
            setattr(conn, _TRANSACTION_LOCK_ATTRIBUTE, lock)
        if not isinstance(lock, _ConnectionTransactionLock):
            raise TypeError(f"unexpected {_TRANSACTION_LOCK_ATTRIBUTE} on connection")
        return lock


def compress_payload(data: str) -> str:
    raw = data.encode()
    if len(raw) < COMPRESS_THRESHOLD:
        return "plain:" + data
    return "gzip:" + base64.b64encode(gzip.compress(raw, compresslevel=6)).decode()


def decompress_payload(data: str) -> str:
    if data.startswith("gzip:"):
        return gzip.decompress(base64.b64decode(data[5:])).decode()
    if data.startswith("plain:"):
        return data[6:]
    # raw JSON
    return data


async def connect(db_path: str | Path) -> aiosqlite.Connection:
    """Open (or create) a SQLite database and initialise the schema.

    The caller is responsible for closing the returned connection.
    """
    path = Path(db_path)
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path))
    await conn.create_function("compress_payload", 1, compress_payload, deterministic=True)
    await conn.create_function("decompress_payload", 1, decompress_payload, deterministic=True)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS axio_context_messages ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id TEXT NOT NULL,"
        "  project TEXT NOT NULL,"
        "  position INTEGER NOT NULL,"
        "  role TEXT NOT NULL,"
        "  content TEXT NOT NULL,"
        "  provenance TEXT,"
        "  created_at TEXT NOT NULL DEFAULT (datetime('now')),"
        "  UNIQUE(session_id, position)"
        ")"
    )
    async with conn.execute("PRAGMA table_info(axio_context_messages)") as cursor:
        columns = {str(row[1]) for row in await cursor.fetchall()}
    if "provenance" not in columns:
        await conn.execute("ALTER TABLE axio_context_messages ADD COLUMN provenance TEXT")
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_axio_context_messages_session ON axio_context_messages(session_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_axio_context_messages_project ON axio_context_messages(project)"
    )
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS axio_context_tokens ("
        "  session_id TEXT NOT NULL,"
        "  project TEXT NOT NULL,"
        "  input_tokens INTEGER NOT NULL DEFAULT 0,"
        "  output_tokens INTEGER NOT NULL DEFAULT 0,"
        "  PRIMARY KEY(session_id, project)"
        ")"
    )
    await conn.commit()
    return conn


def _extract_preview(content_json: str, max_len: int = 80) -> str:
    """Extract text preview from serialized content JSON."""
    try:
        blocks = json.loads(content_json)
        for b in blocks:
            if b.get("type") == "text":
                text: str = b["text"]
                return text[:max_len] + ("..." if len(text) > max_len else "")
    except (json.JSONDecodeError, KeyError):
        pass
    return "(no preview)"


class SQLiteContextStore(ContextStore):
    """Persistent conversation storage backed by SQLite.

    The caller owns the connection and is responsible for closing it.
    Use :func:`connect` to open a properly initialized connection.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
        project: str | None = None,
        db_name: str = "axio_context",
    ) -> None:
        self._conn = conn
        self._db_name = db_name
        self._session_id = session_id
        self._project = project or str(Path.cwd().resolve())
        self._transaction_lock = _transaction_lock_for(conn)

    @property
    def session_id(self) -> str:
        return self._session_id

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        async with self._transaction_lock.hold():
            try:
                yield
            except BaseException:
                # Cancellation and non-Exception failures must release no partial transaction.
                await self._conn.rollback()
                raise

    async def append(self, message: Message) -> None:
        content_json = json.dumps(message.to_dict()["content"])
        provenance_json = json.dumps(message.provenance.to_dict()) if message.provenance is not None else None
        async with self._transaction():
            await self._conn.execute(
                "INSERT INTO axio_context_messages (session_id, project, position, role, content, provenance)"
                "VALUES (?, ?, (SELECT COUNT(*) FROM axio_context_messages WHERE session_id = ?), ?, "
                "compress_payload(?), ?)",
                (self._session_id, self._project, self._session_id, message.role, content_json, provenance_json),
            )
            await self._conn.commit()

    async def append_many(self, messages: list[Message]) -> None:
        if not messages:
            return
        rows = [
            (
                self._session_id,
                self._project,
                self._session_id,
                message.role,
                json.dumps(message.to_dict()["content"]),
                json.dumps(message.provenance.to_dict()) if message.provenance is not None else None,
            )
            for message in messages
        ]
        async with self._transaction():
            await self._conn.executemany(
                "INSERT INTO axio_context_messages (session_id, project, position, role, content, provenance)"
                "VALUES (?, ?, (SELECT COUNT(*) FROM axio_context_messages WHERE session_id = ?), ?, "
                "compress_payload(?), ?)",
                rows,
            )
            await self._conn.commit()

    async def get_history(self) -> list[Message]:
        async with self._conn.execute(
            "SELECT role, decompress_payload(content), provenance FROM axio_context_messages"
            " WHERE session_id = ? ORDER BY position",
            (self._session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        history: list[Message] = []
        for role, content, provenance in rows:
            data = {"role": role, "content": json.loads(content)}
            if provenance is not None:
                data["provenance"] = json.loads(provenance)
            history.append(Message.from_dict(data))
        return history

    async def clear(self) -> None:
        async with self._transaction():
            await self._conn.execute("DELETE FROM axio_context_messages WHERE session_id = ?", (self._session_id,))
            await self._conn.execute(
                "DELETE FROM axio_context_tokens WHERE session_id = ? AND project = ?",
                (self._session_id, self._project),
            )
            await self._conn.commit()

    async def fork(self) -> SQLiteContextStore:
        new_id = uuid4().hex
        async with self._transaction():
            await self._conn.execute(
                "INSERT INTO axio_context_messages (session_id, project, position, role, content, provenance)"
                "SELECT ?, project, position, role, content, provenance "
                "FROM axio_context_messages WHERE session_id = ?",
                (new_id, self._session_id),
            )
            await self._conn.execute(
                "INSERT OR IGNORE INTO axio_context_tokens (session_id, project, input_tokens, output_tokens) "
                "SELECT ?, project, input_tokens, output_tokens FROM axio_context_tokens "
                "WHERE session_id = ? AND project = ?",
                (new_id, self._session_id, self._project),
            )
            await self._conn.commit()
        return SQLiteContextStore(self._conn, new_id, self._project)

    async def set_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        async with self._transaction():
            await self._conn.execute(
                "INSERT INTO axio_context_tokens (session_id, project, input_tokens, output_tokens)"
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, project) DO UPDATE SET input_tokens=?, output_tokens=?",
                (self._session_id, self._project, input_tokens, output_tokens, input_tokens, output_tokens),
            )
            await self._conn.commit()

    async def add_context_tokens(self, input_tokens: int, output_tokens: int) -> None:
        async with self._transaction():
            await self._conn.execute(
                "INSERT INTO axio_context_tokens (session_id, project, input_tokens, output_tokens)"
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, project) DO UPDATE "
                "SET input_tokens = input_tokens + excluded.input_tokens, "
                "    output_tokens = output_tokens + excluded.output_tokens",
                (self._session_id, self._project, input_tokens, output_tokens),
            )
            await self._conn.commit()

    async def get_context_tokens(self) -> tuple[int, int]:
        async with self._conn.execute(
            "SELECT input_tokens, output_tokens FROM axio_context_tokens WHERE session_id = ? AND project = ?",
            (self._session_id, self._project),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return 0, 0
        return int(row[0]), int(row[1])

    async def close(self) -> None:
        """No-op: the caller owns the connection."""

    async def list_sessions(self) -> list[SessionInfo]:
        """List all sessions for a project, newest first."""
        async with self._conn.execute(
            "SELECT m.session_id, COUNT(*) as cnt, "
            "(SELECT decompress_payload(content) FROM axio_context_messages WHERE session_id = m.session_id "
            "AND role = 'user' ORDER BY position LIMIT 1) as first_content, "
            "MIN(m.created_at) as created, "
            "COALESCE(ct.input_tokens, 0), COALESCE(ct.output_tokens, 0) "
            "FROM axio_context_messages m "
            "LEFT JOIN axio_context_tokens ct ON ct.session_id = m.session_id AND ct.project = m.project "
            "WHERE m.project = ? "
            "GROUP BY m.session_id ORDER BY created DESC",
            (self._project,),
        ) as cursor:
            rows = await cursor.fetchall()
        result: list[SessionInfo] = []
        for session_id, count, first_content, created_at, in_tok, out_tok in rows:
            preview = _extract_preview(first_content) if first_content else "(no preview)"
            result.append(
                SessionInfo(
                    session_id=session_id,
                    message_count=count,
                    preview=preview,
                    created_at=created_at,
                    input_tokens=int(in_tok),
                    output_tokens=int(out_tok),
                )
            )
        return result
