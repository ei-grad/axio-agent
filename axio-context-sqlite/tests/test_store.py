"""Tests for SQLiteContextStore."""

import asyncio
import threading
from collections.abc import AsyncGenerator
from pathlib import Path

import aiosqlite
import pytest
from axio.blocks import TextBlock
from axio.messages import Message

from axio_context_sqlite import SQLiteContextStore, connect
from axio_context_sqlite.store import COMPRESS_THRESHOLD, compress_payload, decompress_payload


def _msg(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])  # type: ignore[arg-type]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
async def conn(db_path: Path) -> AsyncGenerator[aiosqlite.Connection, None]:
    c = await connect(db_path)
    yield c
    await c.close()


@pytest.fixture
async def store(conn: aiosqlite.Connection) -> SQLiteContextStore:
    return SQLiteContextStore(conn, "session-1", "test-project")


async def test_append_and_get_history(store: SQLiteContextStore) -> None:
    await store.append(_msg("user", "Hello"))
    await store.append(_msg("assistant", "Hi!"))
    history = await store.get_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


async def test_append_many_commits_ordered_batch(store: SQLiteContextStore) -> None:
    await store.append_many([_msg("assistant", "tool use"), _msg("user", "tool result")])

    history = await store.get_history()
    assert [(message.role, message.content[0].text) for message in history] == [  # type: ignore[attr-defined]
        ("assistant", "tool use"),
        ("user", "tool result"),
    ]


async def test_append_many_rolls_back_whole_batch_on_insert_failure(
    store: SQLiteContextStore,
    conn: aiosqlite.Connection,
) -> None:
    await conn.execute(
        "CREATE TRIGGER reject_user_message BEFORE INSERT ON axio_context_messages "
        "WHEN NEW.role = 'user' BEGIN SELECT RAISE(ABORT, 'rejected test row'); END"
    )
    await conn.commit()

    with pytest.raises(aiosqlite.DatabaseError, match="rejected test row"):
        await store.append_many([_msg("assistant", "tool use"), _msg("user", "tool result")])

    assert await store.get_history() == []


async def test_cancelled_batch_rollback_preserves_concurrent_fork_append(
    store: SQLiteContextStore,
    conn: aiosqlite.Connection,
) -> None:
    forked = await store.fork()
    compress_count = 0
    second_compress_started = threading.Event()
    release_compress = threading.Event()

    def blocking_compress(data: str) -> str:
        nonlocal compress_count
        compress_count += 1
        if compress_count == 2:
            second_compress_started.set()
            if not release_compress.wait(timeout=2):
                raise TimeoutError("test did not release the blocked compression")
        return compress_payload(data)

    await conn.create_function("compress_payload", 1, blocking_compress, deterministic=True)
    batch_task = asyncio.create_task(
        store.append_many([_msg("assistant", "batch tool use"), _msg("user", "batch tool result")])
    )
    independent_append_started = asyncio.Event()

    async def append_to_fork() -> None:
        independent_append_started.set()
        await forked.append(_msg("user", "independent append"))

    independent_task: asyncio.Task[None] | None = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(second_compress_started.wait, 1), timeout=2)
        independent_task = asyncio.create_task(append_to_fork())
        await asyncio.wait_for(independent_append_started.wait(), timeout=1)
        await asyncio.sleep(0)

        batch_task.cancel()
        release_compress.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(batch_task, timeout=1)
        await asyncio.wait_for(independent_task, timeout=1)
    finally:
        release_compress.set()
        if not batch_task.done():
            batch_task.cancel()
        if independent_task is not None and not independent_task.done():
            independent_task.cancel()
        await asyncio.gather(
            batch_task,
            *(task for task in [independent_task] if task is not None),
            return_exceptions=True,
        )

    assert await store.get_history() == []
    history = await forked.get_history()
    assert [(message.role, message.content[0].text) for message in history] == [  # type: ignore[attr-defined]
        ("user", "independent append"),
    ]


async def test_clear(store: SQLiteContextStore) -> None:
    await store.append(_msg("user", "Hello"))
    await store.clear()
    assert await store.get_history() == []


async def test_fork(store: SQLiteContextStore) -> None:
    await store.append(_msg("user", "Hello"))
    forked = await store.fork()
    await forked.append(_msg("assistant", "Hi!"))
    assert len(await store.get_history()) == 1
    assert len(await forked.get_history()) == 2


async def test_concurrent_parent_and_fork_mutations(store: SQLiteContextStore) -> None:
    await store.append(_msg("user", "shared history"))
    forked = await store.fork()

    await asyncio.gather(
        store.append(_msg("assistant", "parent only")),
        forked.append(_msg("assistant", "fork only")),
        store.add_context_tokens(10, 1),
        forked.add_context_tokens(20, 2),
    )

    parent_history = await store.get_history()
    fork_history = await forked.get_history()
    assert [message.content[0].text for message in parent_history] == [  # type: ignore[attr-defined]
        "shared history",
        "parent only",
    ]
    assert [message.content[0].text for message in fork_history] == [  # type: ignore[attr-defined]
        "shared history",
        "fork only",
    ]
    assert await store.get_context_tokens() == (10, 1)
    assert await forked.get_context_tokens() == (20, 2)


async def test_set_get_context_tokens(store: SQLiteContextStore) -> None:
    await store.set_context_tokens(100, 50)
    inp, out = await store.get_context_tokens()
    assert inp == 100
    assert out == 50


async def test_add_context_tokens(store: SQLiteContextStore) -> None:
    await store.set_context_tokens(100, 50)
    await store.add_context_tokens(20, 10)
    inp, out = await store.get_context_tokens()
    assert inp == 120
    assert out == 60


async def test_list_sessions(db_path: Path) -> None:
    c = await connect(db_path)
    try:
        s1 = SQLiteContextStore(c, "sess-a", "proj")
        s2 = SQLiteContextStore(c, "sess-b", "proj")
        await s1.append(_msg("user", "First session"))
        await s2.append(_msg("user", "Second session"))
        sessions = await s1.list_sessions()
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert ids == {"sess-a", "sess-b"}
        previews = {s.preview for s in sessions}
        assert "First session" in previews
        assert "Second session" in previews
    finally:
        await c.close()


class TestCompressPayload:
    def test_small_stays_plain(self) -> None:
        data = "hello"
        result = compress_payload(data)
        assert result.startswith("plain:")
        assert decompress_payload(result) == data

    def test_large_gets_compressed(self) -> None:
        data = "x" * COMPRESS_THRESHOLD
        result = compress_payload(data)
        assert result.startswith("gzip:")
        assert decompress_payload(result) == data

    def test_legacy_no_prefix(self) -> None:
        assert decompress_payload('{"foo": 1}') == '{"foo": 1}'

    def test_roundtrip(self) -> None:
        data = "[" + '{"type":"text","text":"a"},' * 50 + "]"
        assert decompress_payload(compress_payload(data)) == data


async def test_large_message_compressed_on_disk(db_path: Path) -> None:
    """Large content is stored compressed; get_history still returns original."""
    big_text = "word " * 200  # well above threshold

    c = await connect(db_path)
    try:
        s = SQLiteContextStore(c, "big-session", "proj")
        await s.append(_msg("user", big_text))
    finally:
        await c.close()

    # Inspect raw bytes on disk - should start with gzip:
    async with aiosqlite.connect(str(db_path)) as raw:
        async with raw.execute("SELECT content FROM axio_context_messages") as cur:
            row = await cur.fetchone()
    assert row is not None
    assert row[0].startswith("gzip:")

    # But get_history transparently decompresses
    c2 = await connect(db_path)
    try:
        s2 = SQLiteContextStore(c2, "big-session", "proj")
        history = await s2.get_history()
    finally:
        await c2.close()
    assert len(history) == 1
    block = history[0].content[0]
    assert isinstance(block, TextBlock)
    assert block.text == big_text


async def test_close_and_reopen(db_path: Path) -> None:
    """Data persists after close and reopen."""
    c = await connect(db_path)
    try:
        s = SQLiteContextStore(c, "persist-session", "proj")
        await s.append(_msg("user", "Persistent message"))
        await s.set_context_tokens(42, 7)
    finally:
        await c.close()

    c2 = await connect(db_path)
    try:
        s2 = SQLiteContextStore(c2, "persist-session", "proj")
        history = await s2.get_history()
        assert len(history) == 1
        assert history[0].role == "user"
        inp, out = await s2.get_context_tokens()
        assert inp == 42
        assert out == 7
    finally:
        await c2.close()
