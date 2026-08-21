from __future__ import annotations

import asyncio
import json
import os
import stat
import struct
import zlib
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import DummyInput
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput

from axio_repl import _replay as replay_module
from axio_repl._panel import make_session
from axio_repl._replay import MAGIC, RecordingInput, ReplayCorruptionError, ReplayLog, read_replay, recording_output

_START_PAYLOAD = {
    "application": "axio-repl",
    "version": "test",
    "cwd": "/tmp/project",
    "mode": "interactive",
}


def _encoded_record(seq: int, kind: str, payload: object) -> bytes:
    raw = json.dumps(
        {
            "schema_version": 1,
            "seq": seq,
            "offset_ns": seq - 1,
            "session_id": "crafted",
            "kind": kind,
            "payload": payload,
        },
        separators=(",", ":"),
    ).encode()
    compressed = zlib.compress(raw)
    return struct.pack(">II", len(compressed), len(raw)) + compressed


class _FixedInput(DummyInput):
    def __init__(self, keys: list[KeyPress]) -> None:
        self._keys = keys

    def read_keys(self) -> list[KeyPress]:
        keys, self._keys = self._keys, []
        return keys

    @property
    def closed(self) -> bool:
        return False


class _MutableOutput(DummyOutput):
    def __init__(self, rows: int, columns: int) -> None:
        self.size = Size(rows=rows, columns=columns)

    def get_size(self) -> Size:
        return self.size


async def test_replay_records_terminal_frames_keys_and_raw_editor_state(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    replay = await ReplayLog.open(
        session_dir=tmp_path,
        session_id="replay-session",
        start_payload=_START_PAYLOAD,
    )
    output = recording_output(DummyOutput(), replay)
    output.write_raw("visible output\n")
    output.cursor_up(1)
    output.flush()

    input_adapter = RecordingInput(_FixedInput([KeyPress("x", "x"), KeyPress(Keys.ControlC, "\x03")]), replay)
    assert [key.data for key in input_adapter.read_keys()] == ["x", "\x03"]
    assert replay.record(
        "editor_state",
        {"text": "api_key=not-redacted-by-design", "cursor_position": 30},
    )
    await replay.close({"status": "complete"})

    result = read_replay(replay.replay_path)
    assert result.discarded_tail_bytes == 0
    assert [record["seq"] for record in result.records] == list(range(1, len(result.records) + 1))
    assert [record["kind"] for record in result.records] == [
        "session_start",
        "terminal_geometry",
        "terminal_frame",
        "key_press",
        "key_press",
        "editor_state",
        "session_end",
    ]
    frame = result.records[2]["payload"]
    assert isinstance(frame, dict)
    assert frame["operations"] == [
        {"op": "write_raw", "args": ["visible output\n"]},
        {"op": "cursor_up", "args": [1]},
    ]
    assert result.records[5]["payload"] == {
        "text": "api_key=not-redacted-by-design",
        "cursor_position": 30,
    }
    offsets: list[int] = []
    for record in result.records:
        offset = record["offset_ns"]
        assert isinstance(offset, int) and offset >= 0
        offsets.append(offset)
    assert offsets == sorted(offsets)
    assert stat.S_IMODE(replay.replay_path.stat().st_mode) == 0o600


async def test_replay_reader_ignores_only_an_incomplete_final_frame(tmp_path: Path) -> None:
    replay = await ReplayLog.open(
        session_dir=tmp_path,
        session_id="tail",
        start_payload=_START_PAYLOAD,
    )
    assert replay.record("terminal_geometry", {"rows": 24, "columns": 80, "source": "initial"})
    assert replay.record("editor_state", {"text": "draft", "cursor_position": 5})
    await replay.close()
    valid = replay.replay_path.read_bytes()

    with replay.replay_path.open("ab") as replay_file:
        replay_file.write(b"\x00\x00\x00")
        replay_file.flush()
        os.fsync(replay_file.fileno())

    result = read_replay(replay.replay_path)
    assert len(result.records) == 4
    assert result.discarded_tail_bytes == 3
    assert replay.replay_path.read_bytes().startswith(valid)


def test_replay_reader_bounds_decompression_by_declared_size(tmp_path: Path) -> None:
    compressed = zlib.compress(b"x" * (1024 * 1024))
    replay_path = tmp_path / "bomb.axrp"
    replay_path.write_bytes(MAGIC + struct.pack(">II", len(compressed), 1) + compressed)

    with pytest.raises(ReplayCorruptionError, match="invalid uncompressed size"):
        read_replay(replay_path)


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                ("session_start", _START_PAYLOAD),
                ("session_start", _START_PAYLOAD),
            ],
            "second session_start",
        ),
        (
            [
                ("session_start", _START_PAYLOAD),
                ("session_end", {"status": "complete"}),
                (
                    "runtime_event",
                    {
                        "hub_seq": 1,
                        "run_id": "run",
                        "agent_id": "main",
                        "parent_agent_id": None,
                        "turn_id": None,
                        "context_id": None,
                        "execution_mode": "foreground",
                        "parent_tool_use_id": None,
                        "kind": "turn_started",
                        "payload": {"event": {}},
                    },
                ),
            ],
            "after session_end",
        ),
    ],
)
def test_replay_reader_rejects_invalid_lifecycle_order(
    tmp_path: Path,
    records: list[tuple[str, object]],
    message: str,
) -> None:
    replay_path = tmp_path / "lifecycle.axrp"
    replay_path.write_bytes(
        MAGIC + b"".join(_encoded_record(seq, kind, payload) for seq, (kind, payload) in enumerate(records, start=1))
    )

    with pytest.raises(ReplayCorruptionError, match=message):
        read_replay(replay_path)


async def test_prompt_toolkit_records_render_keys_and_editor_transitions(tmp_path: Path) -> None:
    replay = await ReplayLog.open(
        session_dir=tmp_path,
        session_id="prompt",
        start_payload=_START_PAYLOAD,
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        session = make_session(lambda: "status", replay=replay)
        prompt_task = asyncio.create_task(session.prompt_async("user> "))
        await asyncio.sleep(0)
        pipe.send_text("draft\r")
        assert await asyncio.wait_for(prompt_task, timeout=1) == "draft"
    await replay.close()

    records = read_replay(replay.replay_path).records
    kinds = [record["kind"] for record in records]
    assert "terminal_frame" in kinds
    assert kinds.count("key_press") >= len("draft") + 1
    editor_states = [record["payload"] for record in records if record["kind"] == "editor_state"]
    assert any(isinstance(state, dict) and state.get("text") == "draft" for state in editor_states)


async def test_replay_write_failure_degrades_without_failing_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[BaseException] = []
    replay = await ReplayLog.open(
        session_dir=tmp_path,
        session_id="write-failure",
        start_payload=_START_PAYLOAD,
        on_degraded=failures.append,
    )
    assert replay.record("terminal_geometry", {"rows": 24, "columns": 80, "source": "initial"})

    def fail_write(file_descriptor: int, data: bytes) -> None:
        del file_descriptor, data
        raise OSError("replay disk unavailable")

    monkeypatch.setattr(replay_module, "_write_all", fail_write)
    assert replay.record("editor_state", {"text": "draft", "cursor_position": 5})
    await replay.close()
    await asyncio.sleep(0)

    assert isinstance(replay.degraded_reason, OSError)
    assert str(replay.degraded_reason) == "replay disk unavailable"
    assert len(failures) == 1


async def test_partial_replay_write_stops_before_appending_later_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = await ReplayLog.open(
        session_dir=tmp_path,
        session_id="partial-write",
        start_payload=_START_PAYLOAD,
    )

    def write_partial_then_fail(file_descriptor: int, data: bytes) -> None:
        os.write(file_descriptor, data[:12])
        raise OSError("partial replay write")

    monkeypatch.setattr(replay_module, "_write_all", write_partial_then_fail)
    assert replay.record("terminal_geometry", {"rows": 24, "columns": 80, "source": "initial"})
    assert replay.record("editor_state", {"text": "draft", "cursor_position": 5})
    await replay.close()

    result = read_replay(replay.replay_path)
    assert [record["kind"] for record in result.records] == ["session_start"]
    assert result.discarded_tail_bytes == 12


async def test_replay_schema_round_trips_every_supported_frontend_kind(tmp_path: Path) -> None:
    replay = await ReplayLog.open(
        session_dir=tmp_path,
        session_id="schema",
        start_payload=_START_PAYLOAD,
    )
    records = (
        ("terminal_geometry", {"rows": 24, "columns": 80, "source": "initial"}),
        ("terminal_frame", {"operations": [{"op": "write_raw", "args": ["text"]}]}),
        ("terminal_fallback", {"content": "late", "stream": "stderr", "destination": "late"}),
        ("key_press", {"key": "x", "data": "x"}),
        ("editor_state", {"text": "draft", "cursor_position": 5}),
        (
            "input_submission",
            {
                "text": "draft",
                "target_agent_id": "main",
                "disposition": "pending",
                "input_id": "input-1",
                "arrival_seq": 1,
            },
        ),
        (
            "runtime_event",
            {
                "hub_seq": 1,
                "run_id": "run-1",
                "agent_id": "main",
                "parent_agent_id": None,
                "turn_id": None,
                "context_id": "context-1",
                "execution_mode": "foreground",
                "parent_tool_use_id": None,
                "kind": "configuration_changed",
                "payload": {"event": {"name": "model", "value": "stub"}},
            },
        ),
        ("terminal_geometry", {"rows": 40, "columns": 120, "source": "resize"}),
    )
    for kind, payload in records:
        assert replay.record(kind, payload)
    await replay.close({"status": "complete"})

    decoded = read_replay(replay.replay_path).records
    assert [record["kind"] for record in decoded] == [
        "session_start",
        *(kind for kind, _payload in records),
        "session_end",
    ]


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("unknown", {}),
        ("session_start", _START_PAYLOAD),
        ("session_end", {"status": "complete"}),
        ("editor_state", {"text": "draft"}),
        ("terminal_frame", {"operations": [{"op": "unknown"}]}),
        ("key_press", {"key": "x", "data": object()}),
    ],
)
async def test_replay_schema_rejects_unknown_or_malformed_records(
    tmp_path: Path,
    kind: str,
    payload: object,
) -> None:
    session_dir = tmp_path / kind
    session_dir.mkdir()
    replay = await ReplayLog.open(
        session_dir=session_dir,
        session_id=f"invalid-{kind}",
        start_payload=_START_PAYLOAD,
    )
    assert replay.record("terminal_geometry", {"rows": 24, "columns": 80, "source": "initial"})

    assert not replay.record(kind, payload)
    assert replay.degraded_reason is not None
    await replay.close()


async def test_recording_output_captures_initial_geometry_and_resizes_once(tmp_path: Path) -> None:
    replay = await ReplayLog.open(
        session_dir=tmp_path,
        session_id="geometry",
        start_payload=_START_PAYLOAD,
    )
    delegate = _MutableOutput(rows=24, columns=80)
    output = recording_output(delegate, replay)
    assert output.get_size() == Size(rows=24, columns=80)
    delegate.size = Size(rows=40, columns=120)
    assert output.get_size() == Size(rows=40, columns=120)
    assert output.get_size() == Size(rows=40, columns=120)
    await replay.close()

    geometries = [
        record["payload"]
        for record in read_replay(replay.replay_path).records
        if record["kind"] == "terminal_geometry"
    ]
    assert geometries == [
        {"rows": 24, "columns": 80, "source": "initial"},
        {"rows": 40, "columns": 120, "source": "resize"},
    ]


async def test_large_runtime_media_uses_bounded_reference_without_disabling_frames(tmp_path: Path) -> None:
    replay = await ReplayLog.open(
        session_dir=tmp_path,
        session_id="large-media",
        start_payload=_START_PAYLOAD,
    )
    assert replay.record("terminal_geometry", {"rows": 24, "columns": 80, "source": "initial"})
    media = b"x" * (4 * 1024 * 1024)
    assert replay.record(
        "runtime_event",
        {
            "hub_seq": 1,
            "run_id": "run-1",
            "agent_id": "main",
            "parent_agent_id": None,
            "turn_id": "turn-1",
            "context_id": "context-1",
            "execution_mode": "foreground",
            "parent_tool_use_id": None,
            "kind": "message_committed",
            "payload": {"message": {"content": [{"media_type": "image/png", "data": media}]}},
        },
    )
    assert replay.record("terminal_frame", {"operations": [{"op": "write_raw", "args": ["after media"]}]})
    await replay.close()

    decoded = read_replay(replay.replay_path).records
    runtime = next(record for record in decoded if record["kind"] == "runtime_event")
    payload = runtime["payload"]
    assert isinstance(payload, dict)
    runtime_payload = payload["payload"]
    assert isinstance(runtime_payload, dict)
    message = runtime_payload["message"]
    assert isinstance(message, dict)
    content = message["content"]
    assert isinstance(content, list)
    image = content[0]
    assert isinstance(image, dict)
    binary = image["data"]
    assert isinstance(binary, dict)
    assert binary["type"] == "binary_reference"
    assert binary["size"] == len(media)
    assert decoded[-2]["kind"] == "terminal_frame"
