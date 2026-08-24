from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from axio.agent import Agent
from axio.blocks import TextBlock
from axio.context import MemoryContextStore
from axio.events import (
    Error,
    ImageOutput,
    IterationEnd,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolFieldDelta,
    ToolFieldStart,
    ToolInputDelta,
    ToolUseStart,
)
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.provider_output import ProviderOutputPolicy
from axio.testing import make_text_response, make_tool_use_response
from axio.tool import Tool
from axio.types import CostSource, StopReason, Usage
from axio_tools_agents.runtime import (
    AgentEventEnvelope,
    ConfigurationChanged,
    EditorSnapshot,
    ExecutionMode,
    ObservedContextStore,
    OutcomeDelivered,
    RecoveryApplied,
    RuntimeEvent,
    SessionEventHub,
    ShutdownRecorded,
    TurnFinished,
    TurnStarted,
    TurnStatus,
    new_turn_identity,
    observe_agent_turn,
)

from axio_repl import _journal as journal_module
from axio_repl import _session_journal, _write_runtime_event, main
from axio_repl._journal import (
    JournalCorruptionError,
    JournalQueueFullError,
    SessionJournal,
    default_journal_root,
    read_journal,
    recover_journal_tail,
    session_directory,
)
from axio_repl._recovery import materialize_recovery
from axio_repl._replay import ReplayLog, ReplaySchemaError, read_replay


def _read_records(events_path: Path) -> list[dict[str, Any]]:
    raw = events_path.read_bytes()
    assert raw.endswith(b"\n")
    lines = raw.splitlines()
    assert all(line for line in lines)
    return [json.loads(line.decode("utf-8")) for line in lines]


def _only_events_path(root: Path) -> Path:
    paths = list(root.glob(f"*/*/*/*/{journal_module.SEMANTIC_FILENAME}"))
    assert len(paths) == 1
    return paths[0]


async def test_provider_circuit_breaker_keeps_journal_bounded_and_records_cause(tmp_path: Path) -> None:
    class AmplifyingTransport:
        def __init__(self) -> None:
            self.closed = False

        async def stream(
            self,
            messages: list[Message],
            tools: list[Tool[object]],
            system: str,
        ) -> AsyncIterator[StreamEvent]:
            del messages, tools, system
            try:
                yield TextDelta(index=0, delta="available partial\n")
                yield TextDelta(index=0, delta="REJECTED-JOURNAL-DATA" + "x" * 40_000)
                yield IterationEnd(1, StopReason.end_turn, Usage(1, 1))
            finally:
                self.closed = True

    journal = await SessionJournal.open(session_id="bounded-provider", root=tmp_path)
    hub = SessionEventHub(session_id="bounded-provider")

    async def persist(envelope: AgentEventEnvelope) -> None:
        await _write_runtime_event(journal, envelope)

    hub.subscribe(persist)
    identity = new_turn_identity(
        agent_id="main",
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        run_id="main-run",
        context_id="context",
    )
    transport = AmplifyingTransport()
    agent = Agent(
        system="test",
        transport=transport,
        provider_output_policy=ProviderOutputPolicy(
            max_response_bytes=32 * 1024,
            sustained_rate_bytes_per_second=None,
        ),
    )

    outcome = await observe_agent_turn(
        agent=agent,
        context=MemoryContextStore(),
        prompt="go",
        identity=identity,
        hub=hub,
    )
    await journal.close()

    raw = journal.semantic_path.read_text(encoding="utf-8")
    records = read_journal(journal.semantic_path).records
    checkpoint_parts: list[str] = []
    for record in records:
        if record["kind"] != "turn_checkpoint":
            continue
        payload = record["payload"]
        assert isinstance(payload, dict)
        checkpoint_parts.append(str(payload["text"]))
    checkpoint_text = "".join(checkpoint_parts)
    assert outcome.status is TurnStatus.FAILED
    assert transport.closed
    assert "available partial" in checkpoint_text
    assert "safety size limit" in checkpoint_text
    assert "REJECTED-JOURNAL-DATA" not in raw
    assert "ProviderOutputLimitError" in raw
    assert len(raw.encode()) < 16 * 1024


class _OneShotTransport:
    name = "stub"

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.model = ModelSpec(
            id="stub/model",
            capabilities=frozenset({Capability.text, Capability.tool_use}),
        )
        self.models = ModelRegistry([self.model])
        self.temperature: float | None = None
        self.max_output_tokens: int | None = None
        self.debug = False

    async def fetch_models(self) -> None:
        pass

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools, system
        yield TextDelta(index=0, delta="stub answer")
        yield IterationEnd(
            iteration=1,
            stop_reason=StopReason.end_turn,
            usage=Usage(input_tokens=2, output_tokens=3),
        )


async def _run_stub_one_shot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *extra_args: str,
) -> None:
    import axio_repl

    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    monkeypatch.setattr(axio_repl, "_select_transport", lambda _name: (_OneShotTransport, ""))
    monkeypatch.setattr(
        "sys.argv",
        ["axio-repl", "test prompt", "--sandbox", "none", *extra_args],
    )
    await main()


async def _run_agent_tool_one_shot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    tool_name: str,
    extra_args: tuple[str, ...] = (),
    observed_systems: list[str] | None = None,
    observed_histories: list[list[dict[str, Any]]] | None = None,
) -> None:
    import axio_repl

    class AgentToolTransport(_OneShotTransport):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.calls = 0

        async def stream(
            self,
            messages: list[Message],
            tools: list[Tool[object]],
            system: str,
        ) -> AsyncIterator[StreamEvent]:
            if observed_systems is not None:
                observed_systems.append(system)
            if observed_histories is not None:
                observed_histories.append([message.to_dict() for message in messages])
            self.calls += 1
            events = (
                make_tool_use_response(
                    tool_name,
                    tool_id=f"{tool_name}-call",
                    tool_input={"task": "complete child task", "name": "journal-child"},
                )
                if self.calls == 1
                else make_text_response(f"answer from call {self.calls}")
            )
            for event in events:
                yield event

    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    monkeypatch.setattr(axio_repl, "_select_transport", lambda _name: (AgentToolTransport, ""))
    monkeypatch.setattr(
        "sys.argv",
        [
            "axio-repl",
            "delegate work",
            "--sandbox",
            "none",
            "--session-log-dir",
            str(tmp_path / "journals"),
            *extra_args,
        ],
    )
    await main()


async def test_spawned_child_inherits_prompt_fallback_effort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    systems: list[str] = []

    await _run_agent_tool_one_shot(
        monkeypatch,
        tmp_path,
        tool_name="spawn_agent",
        extra_args=("--effort", "high"),
        observed_systems=systems,
    )

    assert len(systems) >= 2
    assert all("Effort guidance (high)" in system for system in systems)


@pytest.mark.parametrize("tool_name", ["run_agent", "spawn_agent"])
async def test_main_and_local_child_share_one_runtime_identity_without_history_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_name: str,
) -> None:
    import axio_repl

    systems: list[str] = []
    histories: list[list[dict[str, Any]]] = []
    resolver_calls = 0

    def resolve_username() -> str:
        nonlocal resolver_calls
        resolver_calls += 1
        return "nss-user"

    monkeypatch.setattr(axio_repl, "resolve_effective_username", resolve_username)

    await _run_agent_tool_one_shot(
        monkeypatch,
        tmp_path,
        tool_name=tool_name,
        observed_systems=systems,
        observed_histories=histories,
    )

    assert resolver_calls == 1
    assert len(systems) >= 2
    assert all(system.count('"effective_username":"nss-user"') == 1 for system in systems)
    assert all(system.count("axio_runtime_metadata") == 1 for system in systems)
    serialized_history = json.dumps(histories, ensure_ascii=False)
    assert "nss-user" not in serialized_history
    assert "axio_runtime_metadata" not in serialized_history


async def test_selected_agent_model_context_reaches_main_and_child_but_description_does_not(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    bundle = config_dir / "agents" / "local"
    bundle.mkdir(parents=True)
    (bundle / "agent.yaml").write_text(
        """\
version: 1
description: Catalog-only description
model_context: Trusted operator policy description.
""",
        encoding="utf-8",
    )
    systems: list[str] = []

    await _run_agent_tool_one_shot(
        monkeypatch,
        tmp_path,
        tool_name="spawn_agent",
        extra_args=("--config-dir", str(config_dir), "--agent", "local"),
        observed_systems=systems,
    )

    assert len(systems) >= 2
    assert all(system.count("Trusted operator policy description.") == 1 for system in systems)
    assert all(system.count("Operator model context") == 1 for system in systems)
    assert all("Catalog-only description" not in system for system in systems)


@pytest.mark.parametrize("agent_actions", ["off", "on"])
async def test_one_shot_background_report_is_displayed_exactly_once_without_prose_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    agent_actions: str,
) -> None:
    import axio_repl

    unique_report = "UNIQUE_BACKGROUND_ONE_SHOT_REPORT"

    class BackgroundReportTransport(_OneShotTransport):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.parent_calls = 0

        async def stream(
            self,
            messages: list[Message],
            tools: list[Tool[object]],
            system: str,
        ) -> AsyncIterator[StreamEvent]:
            del messages, system
            if not any(tool.name == "spawn_agent" for tool in tools):
                yield TextDelta(index=0, delta=unique_report)
                yield IterationEnd(
                    iteration=1,
                    stop_reason=StopReason.end_turn,
                    usage=Usage(input_tokens=1, output_tokens=1),
                )
                return
            self.parent_calls += 1
            if self.parent_calls == 1:
                for event in make_tool_use_response(
                    "spawn_agent",
                    tool_id="spawn-call",
                    tool_input={"task": "report once", "name": "one-shot-child"},
                ):
                    yield event
                return
            yield TextDelta(index=0, delta="parent response")
            yield IterationEnd(
                iteration=self.parent_calls,
                stop_reason=StopReason.end_turn,
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    monkeypatch.setattr(axio_repl, "_select_transport", lambda _name: (BackgroundReportTransport, ""))
    monkeypatch.setattr(
        "sys.argv",
        [
            "axio-repl",
            "delegate once",
            "--sandbox",
            "none",
            "--agent-actions",
            agent_actions,
            "--session-log-dir",
            str(tmp_path / "journals"),
        ],
    )

    await main()

    output = capsys.readouterr().out
    assert output.count(unique_report) == 1
    assert output.index("incoming") < output.index(unique_report)
    if agent_actions == "on":
        assert "agent one-shot-child" in output


async def test_one_shot_root_stdout_keeps_the_unlabelled_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _run_stub_one_shot(monkeypatch, tmp_path, "--no-session-log")

    output = capsys.readouterr().out
    assert "stub answer" in output
    assert "── agent axio-repl (main) ──" not in output


@pytest.mark.parametrize("no_color", [False, True], ids=("non-tty", "no-color"))
async def test_one_shot_pipe_emits_no_ansi_or_default_powerline_even_with_explicit_theme(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    no_color: bool,
) -> None:
    import axio_repl

    class StyledOneShotTransport(_OneShotTransport):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.calls = 0

        async def stream(
            self,
            messages: list[Message],
            tools: list[Tool[object]],
            system: str,
        ) -> AsyncIterator[StreamEvent]:
            del messages, tools, system
            self.calls += 1
            if self.calls == 1:
                yield ToolUseStart(index=0, tool_use_id="styled-call", name="styled")
                yield ToolInputDelta(index=0, tool_use_id="styled-call", partial_json="{}")
                yield IterationEnd(
                    iteration=1,
                    stop_reason=StopReason.tool_use,
                    usage=Usage(input_tokens=1, output_tokens=1),
                )
                return
            yield ReasoningDelta(index=0, delta="visible reasoning")
            yield TextDelta(index=0, delta="visible answer")
            yield IterationEnd(
                iteration=2,
                stop_reason=StopReason.end_turn,
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    async def styled() -> str:
        return "visible tool result"

    async def build_tools(*args: object, **kwargs: object) -> tuple[list[Tool[object]], str, Path, str]:
        del args, kwargs
        return [Tool(name="styled", handler=styled)], "test", tmp_path, ""

    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(axio_repl, "_select_transport", lambda _name: (StyledOneShotTransport, ""))
    monkeypatch.setattr(axio_repl._sandbox, "build_tools", build_tools)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axio-repl",
            "test prompt",
            "--sandbox",
            "none",
            "--no-session-log",
            "--theme",
            "monochrome",
        ],
    )

    await main()

    output = capsys.readouterr().out
    assert "visible reasoning" in output
    assert "visible tool result" in output
    assert "visible answer" in output
    assert "\x1b[" not in output
    assert "\ue0b0" not in output


@pytest.mark.parametrize("agent_actions", ["off", "on"])
async def test_one_shot_background_failure_reason_has_one_visible_delivery_and_reaches_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    agent_actions: str,
) -> None:
    import axio_repl

    unique_failure = "Stopped after 2 iterations without finishing"
    parent_prompts: list[str] = []

    class BackgroundFailureTransport(_OneShotTransport):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.parent_calls = 0

        async def stream(
            self,
            messages: list[Message],
            tools: list[Tool[object]],
            system: str,
        ) -> AsyncIterator[StreamEvent]:
            del system
            if not any(tool.name == "spawn_agent" for tool in tools):
                yield IterationEnd(
                    iteration=1,
                    stop_reason=StopReason.tool_use,
                    usage=Usage(input_tokens=1, output_tokens=0),
                )
                return
            self.parent_calls += 1
            parent_prompts.extend(
                block.text
                for message in messages
                if message.role == "user"
                for block in message.content
                if isinstance(block, TextBlock)
            )
            if self.parent_calls == 1:
                for event in make_tool_use_response(
                    "spawn_agent",
                    tool_id="spawn-call",
                    tool_input={"task": "fail once", "name": "failing-child"},
                ):
                    yield event
                return
            yield TextDelta(index=0, delta="parent handled failure")
            yield IterationEnd(
                iteration=self.parent_calls,
                stop_reason=StopReason.end_turn,
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    monkeypatch.setattr(axio_repl, "_select_transport", lambda _name: (BackgroundFailureTransport, ""))
    monkeypatch.setattr(
        "sys.argv",
        [
            "axio-repl",
            "delegate failure",
            "--sandbox",
            "none",
            "--agent-actions",
            agent_actions,
            "--max-iterations",
            "2",
            "--no-session-log",
        ],
    )

    await main()

    captured = capsys.readouterr()
    assert captured.out.count(unique_failure) == 1
    assert unique_failure not in captured.err
    assert any(unique_failure in prompt for prompt in parent_prompts)


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
    opened = read_journal(journal.events_path)
    assert [record["kind"] for record in opened.records] == ["session_start"]
    assert opened.discarded_tail_bytes == 0

    await journal.close({"status": "complete"})

    records = _read_records(journal.events_path)
    assert [record["kind"] for record in records] == ["session_start", "session_end"]
    assert records[0]["payload"] == {"model": "test-model"}
    assert records[1]["payload"] == {"status": "complete"}
    assert journal.closed


def test_crash_keeps_last_synced_prefix_after_more_publishes_are_accepted(tmp_path: Path) -> None:
    script = """
import asyncio
import os
import sys
from pathlib import Path

from axio_repl._journal import SessionJournal


async def run() -> None:
    journal = await SessionJournal.open(session_id="crash", root=Path(sys.argv[1]))
    for index in range(25):
        assert await journal.publish("before_boundary", {"index": index})
    assert await journal.sync()
    for index in range(100):
        assert await journal.publish("after_boundary", {"index": index})
    os._exit(23)


asyncio.run(run())
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 23, completed.stderr
    result = read_journal(_only_events_path(tmp_path))
    assert len(result.records) == 26
    assert result.discarded_tail_bytes == 0
    assert [record["seq"] for record in result.records] == list(range(1, len(result.records) + 1))
    assert result.records[0]["kind"] == "session_start"
    boundary_payloads = [record["payload"] for record in result.records[1:26]]
    assert all(isinstance(payload, dict) for payload in boundary_payloads)
    assert [payload["index"] for payload in boundary_payloads if isinstance(payload, dict)] == list(range(25))
    assert all(record["kind"] != "session_end" for record in result.records)


async def test_default_one_shot_writes_complete_main_session_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    await _run_stub_one_shot(monkeypatch, tmp_path)

    assert not (state_home / "axio" / "history").exists()
    events_path = _only_events_path(state_home / "axio" / "sessions")
    records = _read_records(events_path)
    kinds = [record["kind"] for record in records]
    assert kinds[0] == "session_start"
    assert kinds[-1] == "session_end"
    assert "agent_started" in kinds
    assert "input_received" in kinds
    assert "turn_started" in kinds
    assert "turn_checkpoint" in kinds
    assert "iteration_end" in kinds
    assert "turn_finished" in kinds
    assert "agent_stopped" in kinds
    input_record = next(record for record in records if record["kind"] == "input_received")
    assert input_record["payload"]["event"]["text"] == "test prompt"
    assert input_record["payload"]["event"]["source"] == "one-shot"
    configurations = {
        record["payload"]["event"]["name"]: record["payload"]["event"]["value"]
        for record in records
        if record["kind"] == "configuration_changed"
    }
    assert configurations["transport"] == "stub"
    assert configurations["model"] == "stub/model"
    assert configurations["sandbox"] == "host — tools run directly on this machine"
    assert configurations["effort"]["requested"] == "default"
    assert configurations["effort"]["mechanism"] == "prompt-fallback"

    committed = [record for record in records if record["kind"] == "message_committed"]
    assert [record["payload"]["message"]["role"] for record in committed] == ["user", "assistant"]
    assert all(record["agent_id"] == "main" for record in committed)
    assert all(record["context_id"] for record in committed)
    assert committed[0]["payload"]["message"]["content"][0]["text"].endswith("] test prompt")
    assert committed[1]["payload"]["message"]["content"][0]["text"] == "stub answer"

    captured = capsys.readouterr()
    assert "Semantic log:" not in captured.out
    assert str(events_path) in captured.err


async def test_session_journal_can_add_an_independent_opt_in_replay(tmp_path: Path) -> None:
    journal = await SessionJournal.open(
        session_id="with-replay",
        root=tmp_path,
        start_payload={
            "application": "axio-repl",
            "version": "test",
            "cwd": tmp_path,
            "mode": "interactive",
        },
        replay=True,
    )
    assert journal.semantic_path.name == journal_module.SEMANTIC_FILENAME
    assert journal.replay_path is not None
    assert journal.record_replay("terminal_geometry", {"rows": 24, "columns": 80, "source": "initial"})
    assert journal.record_replay("editor_state", {"text": "draft", "cursor_position": 5})
    await journal.close({"status": "complete"})

    assert [record["kind"] for record in read_journal(journal.semantic_path).records] == [
        "session_start",
        "session_end",
    ]
    assert journal.replay_path is not None
    assert [record["kind"] for record in read_replay(journal.replay_path).records] == [
        "session_start",
        "terminal_geometry",
        "editor_state",
        "session_end",
    ]


async def test_replay_start_failure_does_not_disable_semantic_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[BaseException] = []

    async def fail_replay_open(**kwargs: object) -> object:
        del kwargs
        raise OSError("replay storage unavailable")

    monkeypatch.setattr(ReplayLog, "open", fail_replay_open)
    journal = await SessionJournal.open(
        session_id="semantic-survives",
        root=tmp_path,
        replay=True,
        on_replay_degraded=failures.append,
    )
    assert journal.replay_path is None
    assert isinstance(journal.replay_degraded_reason, OSError)
    assert await journal.publish("configuration_changed", {"name": "model", "value": "stub"})
    await journal.close()

    assert len(failures) == 1
    assert [record["kind"] for record in read_journal(journal.semantic_path).records] == [
        "session_start",
        "configuration_changed",
        "session_end",
    ]


async def test_replay_start_notification_failure_does_not_leak_semantic_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_replay_open(**kwargs: object) -> object:
        del kwargs
        raise OSError("replay storage unavailable")

    def fail_notification(error: BaseException) -> None:
        del error
        raise RuntimeError("notification failed")

    monkeypatch.setattr(ReplayLog, "open", fail_replay_open)
    journal = await SessionJournal.open(
        session_id="callback-failure",
        root=tmp_path,
        replay=True,
        on_replay_degraded=fail_notification,
    )
    assert await journal.publish("configuration_changed", {"name": "model", "value": "stub"})
    await journal.close()
    await asyncio.sleep(0)

    assert [record["kind"] for record in read_journal(journal.semantic_path).records] == [
        "session_start",
        "configuration_changed",
        "session_end",
    ]
    assert not any(task.get_name() == "axio-journal-callback-failure" for task in asyncio.all_tasks())


async def test_arbitrary_semantic_close_payload_only_degrades_replay(
    tmp_path: Path,
) -> None:
    journal = await SessionJournal.open(
        session_id="arbitrary-close-payload",
        root=tmp_path,
        start_payload={
            "application": "axio-repl",
            "version": "test",
            "cwd": tmp_path,
            "mode": "interactive",
        },
        replay=True,
    )
    assert journal.record_replay("terminal_geometry", {"rows": 24, "columns": 80, "source": "initial"})

    await journal.close("done")
    await asyncio.sleep(0)

    assert journal.closed
    assert isinstance(journal.replay_degraded_reason, ReplaySchemaError)
    assert [record["kind"] for record in read_journal(journal.semantic_path).records] == [
        "session_start",
        "session_end",
    ]
    assert not any(task.get_name() == "axio-replay-arbitrary-close-payload" for task in asyncio.all_tasks())


async def test_replay_accepts_real_runtime_envelopes_before_and_after_frontend_start(tmp_path: Path) -> None:
    hub = SessionEventHub(session_id="runtime-replay")
    async with _session_journal(
        hub,
        disabled=False,
        root=tmp_path,
        one_shot=False,
        cwd=tmp_path,
        replay=True,
    ) as journal:
        assert journal is not None
        await hub.publish(
            ConfigurationChanged(name="model", value="stub/model", source="startup"),
            run_id="main-run",
            agent_id="main",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.FOREGROUND,
            context_id="context-1",
        )
        assert journal.record_replay("terminal_geometry", {"rows": 24, "columns": 80, "source": "initial"})
        await hub.publish(
            TurnStarted("inspect"),
            run_id="main-run",
            agent_id="main",
            parent_agent_id=None,
            turn_id="turn-1",
            execution_mode=ExecutionMode.FOREGROUND,
            context_id="context-1",
        )
        replay_path = journal.replay_path

    assert replay_path is not None
    runtime = [record for record in read_replay(replay_path).records if record["kind"] == "runtime_event"]
    assert [record["payload"]["kind"] for record in runtime if isinstance(record["payload"], dict)] == [
        "configuration_changed",
        "turn_started",
    ]


async def test_cli_effort_is_recorded_as_configuration_without_extra_history_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    await _run_stub_one_shot(monkeypatch, tmp_path, "--effort", "high")

    events_path = _only_events_path(state_home / "axio" / "sessions")
    records = _read_records(events_path)
    effort_change = next(
        record
        for record in records
        if record["kind"] == "configuration_changed" and record["payload"]["event"]["name"] == "effort"
    )
    assert effort_change["payload"]["event"]["value"]["requested"] == "high"
    assert effort_change["payload"]["event"]["value"]["mechanism"] == "prompt-fallback"
    committed = [record for record in records if record["kind"] == "message_committed"]
    assert [record["payload"]["message"]["role"] for record in committed] == ["user", "assistant"]


async def test_one_shot_session_log_can_be_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    await _run_stub_one_shot(monkeypatch, tmp_path, "--no-session-log")

    assert not (state_home / "axio" / "sessions").exists()
    assert "Semantic log:" not in capsys.readouterr().err


async def test_exact_replay_rejects_one_shot_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        await _run_stub_one_shot(monkeypatch, tmp_path, "--session-replay")


async def test_one_shot_session_log_honours_custom_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_root = tmp_path / "custom-journals"

    await _run_stub_one_shot(
        monkeypatch,
        tmp_path,
        "--session-log-dir",
        str(custom_root),
    )

    records = _read_records(_only_events_path(custom_root))
    assert records[0]["kind"] == "session_start"
    assert records[-1]["kind"] == "session_end"


async def test_journal_open_failure_warns_once_and_does_not_abort_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail_open(**kwargs: object) -> SessionJournal:
        del kwargs
        raise PermissionError("journal root is read-only")

    monkeypatch.setattr(journal_module.SessionJournal, "open", fail_open)
    hub = SessionEventHub(session_id="unwritable")

    async with _session_journal(
        hub,
        disabled=False,
        root=tmp_path,
        one_shot=True,
        cwd=tmp_path,
    ) as journal:
        assert journal is None
        await hub.publish(
            TextDelta(index=0, delta="session continues"),
            run_id="main-run",
            agent_id="main",
            parent_agent_id=None,
            turn_id="turn",
            execution_mode=ExecutionMode.FOREGROUND,
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("Session journal degraded") == 1
    assert "PermissionError: journal root is read-only" in captured.err


async def test_start_sync_failure_warns_once_and_disables_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_sync(_file_descriptor: int) -> None:
        raise OSError("fsync unavailable")

    monkeypatch.setattr(journal_module, "_sync_file", fail_sync)
    hub = SessionEventHub(session_id="start-sync-failure")

    async with _session_journal(
        hub,
        disabled=False,
        root=tmp_path,
        one_shot=True,
        cwd=tmp_path,
    ) as journal:
        assert journal is None
    await asyncio.sleep(0)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("Session journal degraded") == 1
    assert "OSError: fsync unavailable" in captured.err


async def test_journal_records_hidden_agents_and_delivery_before_display_filtering(tmp_path: Path) -> None:
    hub = SessionEventHub(session_id="all-agents")
    displayed: list[str] = []

    async with _session_journal(
        hub,
        disabled=False,
        root=tmp_path,
        one_shot=True,
        cwd=tmp_path,
    ) as journal:
        assert journal is not None

        async def active_only(envelope: AgentEventEnvelope) -> None:
            if envelope.execution_mode is ExecutionMode.FOREGROUND:
                displayed.append(envelope.agent_id)

        hub.subscribe(active_only)
        await hub.publish(
            TextDelta(index=0, delta="hidden output"),
            run_id="background-run",
            agent_id="hidden-child",
            parent_agent_id="main",
            turn_id="background-turn",
            execution_mode=ExecutionMode.BACKGROUND,
            parent_tool_use_id="spawn-call",
            context_id="background-context",
        )
        foreground = new_turn_identity(
            agent_id="foreground-child",
            parent_agent_id="main",
            execution_mode=ExecutionMode.FOREGROUND,
            parent_tool_use_id="run-call",
            run_id="foreground-run",
            context_id="foreground-context",
        )
        await hub.publish_for(foreground, TextDelta(index=0, delta="visible output"))
        await hub.publish_for(
            foreground,
            OutcomeDelivered(recipient_agent_id="main", route="parent_tool_result"),
        )
        events_path = journal.events_path

    records = _read_records(events_path)
    checkpoints = [record for record in records if record["kind"] == "turn_checkpoint"]
    assert [record["agent_id"] for record in checkpoints] == ["hidden-child", "foreground-child"]
    assert [record["payload"]["text"] for record in checkpoints] == ["hidden output", "visible output"]
    assert displayed == ["foreground-child", "foreground-child"]
    delivered = next(record for record in records if record["kind"] == "outcome_delivered")
    assert delivered["parent_agent_id"] == "main"
    assert delivered["parent_tool_use_id"] == "run-call"
    assert delivered["payload"]["event"]["route"] == "parent_tool_result"


async def test_completed_turn_is_a_durable_boundary_without_syncing_stream_deltas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_calls = 0
    original_sync = journal_module._sync_file

    def count_sync(file_descriptor: int) -> None:
        nonlocal sync_calls
        sync_calls += 1
        original_sync(file_descriptor)

    monkeypatch.setattr(journal_module, "_sync_file", count_sync)
    journal = await SessionJournal.open(session_id="boundaries", root=tmp_path)
    assert sync_calls == 1

    await _write_runtime_event(
        journal,
        AgentEventEnvelope(
            seq=1,
            session_id="boundaries",
            run_id="main-run",
            agent_id="main",
            parent_agent_id=None,
            turn_id="turn-1",
            execution_mode=ExecutionMode.FOREGROUND,
            parent_tool_use_id=None,
            event=TextDelta(index=0, delta="live"),
            context_id="context-1",
        ),
    )
    assert sync_calls == 1

    await _write_runtime_event(
        journal,
        AgentEventEnvelope(
            seq=2,
            session_id="boundaries",
            run_id="main-run",
            agent_id="main",
            parent_agent_id=None,
            turn_id="turn-1",
            execution_mode=ExecutionMode.FOREGROUND,
            parent_tool_use_id=None,
            event=TurnFinished(status=TurnStatus.SUCCEEDED, stop_reason=StopReason.end_turn),
            context_id="context-1",
        ),
    )
    assert sync_calls == 2
    durable = read_journal(journal.events_path)
    assert [record["kind"] for record in durable.records] == [
        "session_start",
        "turn_checkpoint",
        "turn_finished",
    ]
    await journal.close()


async def test_semantic_journal_omits_reasoning_and_resumes_sparse_turn_checkpoints(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="semantic", root=tmp_path)
    identity = AgentEventEnvelope(
        seq=1,
        session_id="semantic",
        run_id="main-run",
        agent_id="main",
        parent_agent_id=None,
        turn_id="unfinished",
        execution_mode=ExecutionMode.FOREGROUND,
        parent_tool_use_id=None,
        event=TurnStarted("inspect"),
        context_id="context-1",
    )
    await _write_runtime_event(journal, identity)
    events: tuple[RuntimeEvent, ...] = (
        ReasoningDelta(index=0, delta="private reasoning that must not enter semantic JSONL"),
        TextDelta(index=0, delta="available partial"),
        ToolUseStart(index=1, tool_use_id="call-1", name="write_file"),
        ToolFieldStart(index=1, tool_use_id="call-1", key="path"),
        ToolFieldDelta(index=1, tool_use_id="call-1", key="path", text="demo.py"),
    )
    for seq, event in enumerate(events, start=2):
        await _write_runtime_event(
            journal,
            AgentEventEnvelope(
                seq=seq,
                session_id="semantic",
                run_id="main-run",
                agent_id="main",
                parent_agent_id=None,
                turn_id="unfinished",
                execution_mode=ExecutionMode.FOREGROUND,
                parent_tool_use_id=None,
                event=event,
                context_id="context-1",
            ),
        )
    await _write_runtime_event(
        journal,
        AgentEventEnvelope(
            seq=7,
            session_id="semantic",
            run_id="main-run",
            agent_id="main",
            parent_agent_id=None,
            turn_id="unfinished",
            execution_mode=ExecutionMode.FOREGROUND,
            parent_tool_use_id=None,
            event=ShutdownRecorded(reason="sigterm", pending_input_ids=(), deferred_tool_use_ids=()),
            context_id="context-1",
        ),
    )
    await journal.close()

    raw = journal.semantic_path.read_text(encoding="utf-8")
    assert "private reasoning" not in raw
    records = read_journal(journal.semantic_path).records
    assert not any(record["kind"] == "stream_event" for record in records)
    checkpoints = [record for record in records if record["kind"] == "turn_checkpoint"]
    checkpoint_text: list[str] = []
    for record in checkpoints:
        payload = record["payload"]
        assert isinstance(payload, dict)
        checkpoint_text.append(str(payload["text"]))
    assert "".join(checkpoint_text) == "available partial"

    recovered = materialize_recovery(journal.semantic_path)
    notice = recovered.messages[-1].content[0]
    partial = recovered.messages[-2].content[0]
    assert isinstance(notice, TextBlock)
    assert isinstance(partial, TextBlock)
    assert "available partial" in partial.text
    assert "write_file" in notice.text
    assert "path: demo.py" in notice.text
    assert "private reasoning" not in notice.text


async def test_semantic_journal_preserves_streamed_tool_argument_whitespace(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="tool-whitespace", root=tmp_path)
    base = AgentEventEnvelope(
        seq=1,
        session_id="tool-whitespace",
        run_id="main-run",
        agent_id="main",
        parent_agent_id=None,
        turn_id="turn-1",
        execution_mode=ExecutionMode.FOREGROUND,
        parent_tool_use_id=None,
        event=TurnStarted("patch"),
        context_id="context-1",
    )
    await _write_runtime_event(journal, base)
    raw_parts = ('{"content":"        first line\\n', '\\tsecond line"}')
    events: tuple[RuntimeEvent, ...] = (
        ToolUseStart(index=0, tool_use_id="call-1", name="patch_file"),
        ToolInputDelta(index=0, tool_use_id="call-1", partial_json=raw_parts[0]),
        ToolInputDelta(index=0, tool_use_id="call-1", partial_json=raw_parts[1]),
    )
    for seq, event in enumerate(events, start=2):
        await _write_runtime_event(journal, replace(base, seq=seq, event=event))
    await _write_runtime_event(
        journal,
        replace(
            base,
            seq=5,
            event=TurnFinished(status=TurnStatus.SUCCEEDED, stop_reason=StopReason.tool_use),
        ),
    )
    await journal.close()

    records = read_journal(journal.semantic_path).records
    fragments: list[str] = []
    for record in records:
        if record["kind"] != "turn_checkpoint":
            continue
        payload = record["payload"]
        assert isinstance(payload, dict)
        tool_arguments = payload["tool_arguments"]
        assert isinstance(tool_arguments, dict)
        fragment = tool_arguments.get("call-1", "")
        assert isinstance(fragment, str)
        fragments.append(fragment)
    assert "".join(fragments) == "".join(raw_parts)


@pytest.mark.parametrize(
    ("event", "kind"),
    [
        (EditorSnapshot("draft"), "editor_snapshot"),
        (
            ShutdownRecorded(
                reason="eof",
                pending_input_ids=("input",),
                deferred_tool_use_ids=("tool",),
            ),
            "shutdown_recorded",
        ),
        (
            RecoveryApplied(source_session_id="source", recovery_ids=("source:turn",)),
            "recovery_applied",
        ),
    ],
)
async def test_recovery_lifecycle_records_are_durable_boundaries(
    tmp_path: Path,
    event: RuntimeEvent,
    kind: str,
) -> None:
    journal = await SessionJournal.open(session_id=f"boundary-{kind}", root=tmp_path)
    await _write_runtime_event(
        journal,
        AgentEventEnvelope(
            seq=1,
            session_id=journal.session_id,
            run_id="main-run",
            agent_id="main",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.FOREGROUND,
            parent_tool_use_id=None,
            event=event,
            context_id="context-1",
        ),
    )

    assert [record["kind"] for record in read_journal(journal.events_path).records] == [
        "session_start",
        kind,
    ]
    await journal.close()


@pytest.mark.parametrize(
    ("tool_name", "execution_mode", "delivery_route"),
    [
        ("run_agent", "foreground", "parent_tool_result"),
        ("spawn_agent", "background", "background_outcome_handler"),
    ],
)
async def test_one_shot_journal_captures_actual_local_subagent_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    execution_mode: str,
    delivery_route: str,
) -> None:
    await _run_agent_tool_one_shot(monkeypatch, tmp_path, tool_name=tool_name)

    records = _read_records(_only_events_path(tmp_path / "journals"))
    child_started = next(
        record for record in records if record["kind"] == "agent_started" and record["agent_id"] != "main"
    )
    child_id = child_started["agent_id"]
    assert child_started["parent_agent_id"] == "main"
    child_records = [record for record in records if record["agent_id"] == child_id]
    assert child_records
    assert all(record["execution_mode"] == execution_mode for record in child_records)
    assert all(record["parent_tool_use_id"] == f"{tool_name}-call" for record in child_records)
    assert any(record["kind"] == "turn_checkpoint" for record in child_records)
    committed = [record for record in child_records if record["kind"] == "message_committed"]
    assert [record["payload"]["message"]["role"] for record in committed] == ["user", "assistant"]
    assert len({record["context_id"] for record in child_records if record["context_id"]}) == 1
    if tool_name == "run_agent":
        assert not any(record["kind"] == "outcome_delivered" for record in child_records)
        parent_result_records = [
            record
            for record in records
            if record["agent_id"] == "main"
            and record["kind"] == "message_committed"
            and any(
                block.get("record_type") == "ToolResultBlock" and block.get("tool_use_id") == f"{tool_name}-call"
                for block in record["payload"]["message"]["content"]
            )
        ]
        assert len(parent_result_records) == 1
        child_stopped_seq = max(record["seq"] for record in child_records if record["kind"] == "agent_stopped")
        assert parent_result_records[0]["seq"] > child_stopped_seq
    else:
        delivered = next(record for record in child_records if record["kind"] == "outcome_delivered")
        assert delivered["payload"]["event"]["route"] == delivery_route
    stopped_index = max(index for index, record in enumerate(records) if record["kind"] == "agent_stopped")
    assert stopped_index < len(records) - 1
    assert records[-1]["kind"] == "session_end"


async def test_partial_message_commit_after_cancelled_turn_is_journalled(tmp_path: Path) -> None:
    hub = SessionEventHub(session_id="cancelled")
    context = ObservedContextStore(MemoryContextStore(), hub)
    identity = new_turn_identity(
        agent_id="main",
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        run_id="main-run",
        context_id=context.session_id,
    )
    context.bind_identity(identity)

    async with _session_journal(
        hub,
        disabled=False,
        root=tmp_path,
        one_shot=True,
        cwd=tmp_path,
    ) as journal:
        assert journal is not None
        await hub.publish_for(
            identity,
            TurnFinished(status=TurnStatus.CANCELLED, stop_reason=None, error="turn cancelled"),
        )
        await context.append(Message(role="assistant", content=[TextBlock(text="partial answer")]))
        events_path = journal.events_path

    records = _read_records(events_path)
    finished_index = next(index for index, record in enumerate(records) if record["kind"] == "turn_finished")
    committed_index = next(index for index, record in enumerate(records) if record["kind"] == "message_committed")
    assert committed_index > finished_index
    committed = records[committed_index]
    assert committed["turn_id"] == identity.turn_id
    assert committed["context_id"] == context.session_id
    assert committed["payload"]["message"]["content"][0]["text"] == "partial answer"


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
        assert record["schema_version"] == journal_module.SCHEMA_VERSION
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
        usage=Usage(input_tokens=7, output_tokens=3, cost_usd=0.001, cost_source=CostSource.provider),
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
    assert serialized["usage"] == {
        "record_type": "Usage",
        "input_tokens": 7,
        "output_tokens": 3,
        "cost_usd": 0.001,
        "cost_source": "provider",
    }
    assert serialized["reason"] == "end_turn"
    assert serialized["event"] == {
        "record_type": "IterationEnd",
        "iteration": 2,
        "stop_reason": "tool_use",
        "usage": {
            "record_type": "Usage",
            "input_tokens": 5,
            "output_tokens": 1,
            "cost_usd": None,
            "cost_source": None,
        },
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


async def test_partial_write_leaves_one_recoverable_unterminated_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = await SessionJournal.open(session_id="torn-write", root=tmp_path)

    def write_partial_then_fail(file_descriptor: int, line: bytes) -> None:
        os.write(file_descriptor, line[: max(1, len(line) // 2)])
        raise OSError("simulated crash during write")

    monkeypatch.setattr(journal_module, "_append_line", write_partial_then_fail)
    assert await journal.publish("event", {"value": "not durable"})
    assert not await journal.sync()
    await journal.close()

    before_recovery = read_journal(journal.events_path)
    assert [record["kind"] for record in before_recovery.records] == ["session_start"]
    assert before_recovery.discarded_tail_bytes > 0

    recovered = recover_journal_tail(journal.events_path)
    assert recovered == before_recovery
    assert journal.events_path.read_bytes().endswith(b"\n")
    after_recovery = read_journal(journal.events_path)
    assert after_recovery.records == before_recovery.records
    assert after_recovery.discarded_tail_bytes == 0


async def test_reader_rejects_corruption_except_for_final_unterminated_line(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="corruption", root=tmp_path)
    await journal.close()
    valid = journal.events_path.read_bytes()

    journal.events_path.write_bytes(valid + b'{"partial":')
    result = read_journal(journal.events_path)
    assert len(result.records) == 2
    assert result.discarded_tail_bytes == len(b'{"partial":')

    complete_without_delimiter = b'{"complete":true}'
    journal.events_path.write_bytes(valid + complete_without_delimiter)
    result = read_journal(journal.events_path)
    assert len(result.records) == 2
    assert result.discarded_tail_bytes == len(complete_without_delimiter)

    journal.events_path.write_bytes(valid + b"{not-json}\n")
    with pytest.raises(JournalCorruptionError, match="invalid JSON"):
        read_journal(journal.events_path)

    journal.events_path.write_bytes(valid + b"{not-json}\n{}\n")
    with pytest.raises(JournalCorruptionError, match="invalid JSON"):
        read_journal(journal.events_path)


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

    journal = await SessionJournal.open(
        session_id="queue-full",
        root=tmp_path,
        queue_size=1,
        on_degraded=failures.append,
    )
    monkeypatch.setattr(journal_module, "_append_line", blocked_append)
    assert await journal.publish("event", {"value": 0})
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
