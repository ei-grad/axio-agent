from __future__ import annotations

import asyncio
import json
import stat
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from axio.events import Error, ImageOutput, IterationEnd, TextDelta
from axio.types import StopReason, Usage

from axio_repl import _journal as journal_module
from axio_repl._journal import (
    JournalQueueFullError,
    SessionJournal,
    default_journal_root,
    session_directory,
)


def _read_records(events_path: Path) -> list[dict[str, Any]]:
    raw = events_path.read_bytes()
    assert raw.endswith(b"\n")
    lines = raw.splitlines()
    assert all(line for line in lines)
    return [json.loads(line.decode("utf-8")) for line in lines]


def test_default_root_and_session_directory_follow_xdg() -> None:
    configured = default_journal_root(environ={"XDG_STATE_HOME": "/var/tmp/state"})
    fallback = default_journal_root(environ={"XDG_STATE_HOME": "relative"}, home=Path("/home/tester"))

    assert configured == Path("/var/tmp/state/axio/sessions")
    assert fallback == Path("/home/tester/.local/state/axio/sessions")
    assert session_directory(
        "session-1",
        root=Path("/journal"),
        started_at=datetime(2026, 8, 14, 10, 30, tzinfo=UTC),
    ) == Path("/journal/2026/08/14/session-1")

    with pytest.raises(ValueError, match="session_id"):
        session_directory("../outside", root=Path("/journal"))
    with pytest.raises(ValueError, match="timezone-aware"):
        session_directory("session-1", root=Path("/journal"), started_at=datetime(2026, 8, 14))


async def test_open_creates_private_storage_and_lifecycle_records(tmp_path: Path) -> None:
    journal = await SessionJournal.open(
        session_id="private-session",
        root=tmp_path,
        started_at=datetime(2026, 8, 14, tzinfo=UTC),
        start_payload={"model": "test-model"},
    )

    assert stat.S_IMODE(journal.session_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(journal.events_path.stat().st_mode) == 0o600

    await journal.close({"status": "complete"})

    records = _read_records(journal.events_path)
    assert [record["kind"] for record in records] == ["session_start", "session_end"]
    assert records[0]["payload"] == {"model": "test-model"}
    assert records[1]["payload"] == {"status": "complete"}
    assert journal.closed


async def test_concurrent_publishers_keep_one_strict_global_order(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="concurrent", root=tmp_path, queue_size=256)

    async def produce(index: int) -> None:
        await asyncio.sleep(0)
        accepted = await journal.publish(
            "stream_event",
            TextDelta(index=index, delta=f"message {index} — λ"),
            agent_id=f"agent-{index % 3}",
            parent_agent_id="main",
            turn_id=f"turn-{index}",
            context_id="context-1",
            execution_mode="background",
            parent_tool_use_id=f"call-{index}",
        )
        assert accepted

    await asyncio.gather(*(produce(index) for index in range(100)))
    assert await journal.sync()
    await journal.close()

    records = _read_records(journal.events_path)
    assert [record["seq"] for record in records] == list(range(1, 103))
    assert records[0]["kind"] == "session_start"
    assert records[-1]["kind"] == "session_end"
    for record in records:
        assert record["schema_version"] == 1
        assert record["session_id"] == "concurrent"
        assert record["timestamp"].endswith("Z")
        assert isinstance(record["monotonic_ns"], int)
    sample = records[42]
    assert sample["agent_id"].startswith("agent-")
    assert sample["parent_agent_id"] == "main"
    assert sample["context_id"] == "context-1"
    assert sample["execution_mode"] == "background"
    assert sample["parent_tool_use_id"].startswith("call-")


@dataclass(frozen=True, slots=True)
class _Envelope:
    usage: Usage
    reason: StopReason
    event: object
    failure: BaseException


async def test_serializer_handles_events_dataclasses_enums_usage_and_exceptions(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="types", root=tmp_path)
    payload = _Envelope(
        usage=Usage(input_tokens=7, output_tokens=3),
        reason=StopReason.end_turn,
        event=IterationEnd(iteration=2, stop_reason=StopReason.tool_use, usage=Usage(5, 1)),
        failure=RuntimeError("failed with password=hunter2"),
    )

    assert await journal.publish("stream_event", payload)
    assert await journal.publish("stream_event", Error(ValueError("bad request")))
    await journal.close()

    records = _read_records(journal.events_path)
    serialized = records[1]["payload"]
    assert serialized["record_type"] == "_Envelope"
    assert serialized["usage"] == {"record_type": "Usage", "input_tokens": 7, "output_tokens": 3}
    assert serialized["reason"] == "end_turn"
    assert serialized["event"] == {
        "record_type": "IterationEnd",
        "iteration": 2,
        "stop_reason": "tool_use",
        "usage": {"record_type": "Usage", "input_tokens": 5, "output_tokens": 1},
    }
    assert serialized["failure"] == {
        "exception_type": "RuntimeError",
        "message": f"failed with password={journal_module.REDACTED}",
    }
    assert records[2]["payload"] == {
        "record_type": "Error",
        "exception": {"exception_type": "ValueError", "message": "bad request"},
    }
    assert "traceback" not in journal.events_path.read_text(encoding="utf-8").lower()


async def test_recursive_redaction_covers_keys_and_secret_patterns(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="redaction", root=tmp_path)
    openai_key = "sk-1234567890abcdef"
    github_key = "ghp_1234567890abcdef"
    jwt = "eyJabcdefghijk.abcdefghijkl.abcdefghijkl"
    payload = {
        "Authorization": "Bearer abcdefghijk",
        "nested": [
            {"apiKey": openai_key, "safe": f"header=Bearer abcdefghijk and {github_key}"},
            {"command": f"API_KEY={openai_key} JWT={jwt}"},
        ],
        "aws_secret_access_key": "aws-secret-value",
        "output_tokens": 42,
    }

    assert await journal.publish("tool_result", payload)
    await journal.close()

    raw = journal.events_path.read_text(encoding="utf-8")
    assert openai_key not in raw
    assert github_key not in raw
    assert jwt not in raw
    assert "aws-secret-value" not in raw
    serialized = _read_records(journal.events_path)[1]["payload"]
    assert serialized["Authorization"] == journal_module.REDACTED
    assert serialized["nested"][0]["apiKey"] == journal_module.REDACTED
    assert serialized["aws_secret_access_key"] == journal_module.REDACTED
    assert serialized["output_tokens"] == 42


async def test_bytes_are_externalized_with_media_type_and_deduplicated(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="attachments", root=tmp_path)
    image = b"\x89PNG\r\n\x1a\nimage-data"

    assert await journal.publish("stream_event", ImageOutput(index=0, data=image, media_type="image/png"))
    assert await journal.publish("tool_result", {"media_type": "image/png", "data": image})
    await journal.close()

    records = _read_records(journal.events_path)
    first_reference = records[1]["payload"]["data"]
    second_reference = records[2]["payload"]["data"]
    assert first_reference == second_reference
    assert first_reference["type"] == "attachment"
    assert first_reference["media_type"] == "image/png"
    attachment_path = journal.session_dir / first_reference["path"]
    assert attachment_path.read_bytes() == image
    assert stat.S_IMODE(journal.attachments_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(attachment_path.stat().st_mode) == 0o600
    assert [entry.name for entry in journal.attachments_dir.iterdir()] == [first_reference["sha256"]]


async def test_close_drains_all_accepted_records_without_an_explicit_sync(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="drain", root=tmp_path, queue_size=512)
    for index in range(250):
        assert await journal.publish("event", {"index": index})

    await journal.close()
    await journal.close()

    records = _read_records(journal.events_path)
    assert len(records) == 252
    assert [record["payload"]["index"] for record in records[1:-1]] == list(range(250))
    assert records[-1]["kind"] == "session_end"


async def test_writer_failure_degrades_once_without_failing_the_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[BaseException] = []
    notified = asyncio.Event()

    def on_degraded(error: BaseException) -> None:
        failures.append(error)
        notified.set()

    journal = await SessionJournal.open(session_id="write-failure", root=tmp_path, on_degraded=on_degraded)
    assert await journal.sync()

    def fail_append(file_descriptor: int, line: bytes) -> None:
        del file_descriptor, line
        raise OSError("disk unavailable")

    monkeypatch.setattr(journal_module, "_append_line", fail_append)
    assert await journal.publish("event", {"value": 1})
    assert not await journal.sync()
    await asyncio.wait_for(notified.wait(), timeout=1)

    assert journal.degraded
    assert isinstance(journal.degraded_reason, OSError)
    assert str(journal.degraded_reason) == "disk unavailable"
    assert len(failures) == 1
    assert not await journal.publish("event", {"value": 2})

    await journal.close()
    assert len(failures) == 1
    assert [record["kind"] for record in _read_records(journal.events_path)] == ["session_start"]


async def test_full_queue_degrades_instead_of_blocking_publishers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer_entered = threading.Event()
    release_writer = threading.Event()
    original_append = journal_module._append_line
    failures: list[BaseException] = []

    def blocked_append(file_descriptor: int, line: bytes) -> None:
        writer_entered.set()
        assert release_writer.wait(timeout=2)
        original_append(file_descriptor, line)

    monkeypatch.setattr(journal_module, "_append_line", blocked_append)
    journal = await SessionJournal.open(
        session_id="queue-full",
        root=tmp_path,
        queue_size=1,
        on_degraded=failures.append,
    )
    assert await asyncio.to_thread(writer_entered.wait, 1)
    try:
        assert await journal.publish("event", {"value": 1})
        assert not await journal.publish("event", {"value": 2})
        assert isinstance(journal.degraded_reason, JournalQueueFullError)
        await asyncio.sleep(0)
        assert len(failures) == 1
    finally:
        release_writer.set()
        await journal.close()
