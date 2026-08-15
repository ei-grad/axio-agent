from __future__ import annotations

import asyncio
import sys
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
from axio.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from axio.events import IterationEnd, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.types import StopReason, Usage
from axio_tools_agents.runtime import ConfigurationChanged, EditorSnapshot, RuntimeEvent

from axio_repl import (
    ReplRenderer,
    _build_argument_parser,
    _cancel_and_settle_tasks,
    _handle_agent_actions,
    _IncomingPrompt,
    _pending_prompt_count,
    main,
)
from axio_repl._journal import SessionJournal, read_journal
from axio_repl._multiplexer import ActionMultiplexer, DisplayMode
from axio_repl._recovery import materialize_recovery


def test_agent_actions_default_to_off() -> None:
    args = _build_argument_parser().parse_args([])

    assert args.agent_actions == "off"


def test_agent_actions_can_be_enabled() -> None:
    args = _build_argument_parser().parse_args(["--agent-actions", "on", "inspect"])

    assert args.agent_actions == "on"
    assert args.prompt == "inspect"


def test_agent_actions_rejects_unknown_modes() -> None:
    with pytest.raises(SystemExit):
        _build_argument_parser().parse_args(["--agent-actions", "verbose"])


def test_sandbox_networking_defaults_to_fail_closed() -> None:
    args = _build_argument_parser().parse_args([])

    assert args.sandbox_image == "axio-agent-sandbox:standard"
    assert args.sandbox_network is None
    assert args.sandbox_proxy is None
    assert args.sandbox_memory == "256m"
    assert args.sandbox_cpus == "1.0"


def test_sandbox_restricted_network_flags_are_parsed() -> None:
    args = _build_argument_parser().parse_args(
        [
            "--sandbox-network",
            "agent-egress",
            "--sandbox-proxy",
            "http://mitmania:8080",
            "--sandbox-pypi-index",
            "http://nexus:8081/repository/pypi/simple",
            "--sandbox-datasets",
            "/srv/datasets",
        ]
    )

    assert args.sandbox_network == "agent-egress"
    assert args.sandbox_proxy == "http://mitmania:8080"
    assert args.sandbox_pypi_index.endswith("/simple")
    assert args.sandbox_datasets.as_posix() == "/srv/datasets"


async def test_sandbox_none_reports_restricted_options_as_cli_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--sandbox", "none", "--sandbox-memory", "1g"])

    with pytest.raises(SystemExit, match="2"):
        await main()

    assert "restricted sandbox settings require Docker" in capsys.readouterr().err


async def test_agent_actions_command_toggles_and_publishes_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = ReplRenderer()
    published: list[RuntimeEvent] = []

    async def publish(event: RuntimeEvent) -> None:
        published.append(event)

    assert await _handle_agent_actions(renderer, "on", publish)

    assert renderer.display_mode is DisplayMode.ALL_ACTIONS
    assert published == [ConfigurationChanged(name="agent_actions", value="on", source="interactive")]
    assert "Agent actions" in capsys.readouterr().out


async def test_agent_actions_command_reports_invalid_value_without_changing_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    renderer = ReplRenderer()

    assert not await _handle_agent_actions(renderer, "verbose")

    assert renderer.display_mode is DisplayMode.ACTIVE_ONLY
    assert "must be 'on' or 'off'" in capsys.readouterr().out


async def test_agent_actions_command_reports_discarded_incomplete_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mux = ActionMultiplexer(DisplayMode.ALL_ACTIONS)
    renderer = ReplRenderer(action_multiplexer=mux)
    mux.observe("child", ToolUseStart(index=0, tool_use_id="call", name="shell"))
    mux.observe("child", ToolInputDelta(index=0, tool_use_id="call", partial_json="incomplete"))
    retained = mux.retained_bytes

    assert await _handle_agent_actions(renderer, "off")

    output = capsys.readouterr().out
    assert "discarded 0 queued frame(s)" in output
    assert f"({retained} bytes)" in output


async def test_pending_prompt_count_includes_claimed_and_buffered_prompts() -> None:
    peer_queue: asyncio.Queue[_IncomingPrompt] = asyncio.Queue()
    buffered_prompts: deque[_IncomingPrompt] = deque()
    queued = _IncomingPrompt(text="queued")
    await peer_queue.put(queued)

    inbox_task: asyncio.Task[_IncomingPrompt] | None = asyncio.create_task(peer_queue.get())
    await asyncio.sleep(0)

    assert peer_queue.qsize() == 0
    assert _pending_prompt_count(peer_queue, buffered_prompts, inbox_task) == 1

    assert inbox_task is not None
    buffered_prompts.append(inbox_task.result())
    inbox_task = None

    assert _pending_prompt_count(peer_queue, buffered_prompts, inbox_task) == 1


async def test_cancel_and_settle_tasks_retrieves_done_sibling_failure() -> None:
    async def fail() -> None:
        raise RuntimeError("foreground failed")

    async def wait_forever() -> None:
        await asyncio.Future()

    failed_task = asyncio.create_task(fail())
    pending_task = asyncio.create_task(wait_forever())
    await asyncio.sleep(0)

    outcomes = await _cancel_and_settle_tasks(failed_task, pending_task, None)

    assert isinstance(outcomes[0], RuntimeError)
    assert isinstance(outcomes[1], asyncio.CancelledError)
    assert pending_task.cancelled()


async def test_interactive_input_is_arbitrated_while_a_turn_is_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import axio_repl

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    waiting_after_queued = asyncio.Event()
    inputs: asyncio.Queue[str] = asyncio.Queue()
    observed_calls: list[list[str]] = []
    interrupt_callbacks: list[Callable[[], None]] = []
    active_calls = 0
    max_active_calls = 0

    class BlockingTransport:
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
            self.calls = 0

        async def fetch_models(self) -> None:
            pass

        async def stream(
            self,
            messages: list[Message],
            tools: list[Tool[object]],
            system: str,
        ) -> AsyncIterator[StreamEvent]:
            nonlocal active_calls, max_active_calls
            del tools, system
            self.calls += 1
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            observed_calls.append(
                [
                    "".join(block.text for block in message.content if isinstance(block, TextBlock))
                    for message in messages
                ]
            )
            try:
                if self.calls == 1:
                    first_started.set()
                    await release_first.wait()
                else:
                    second_started.set()
                yield TextDelta(index=0, delta=f"answer {self.calls}")
                yield IterationEnd(
                    iteration=self.calls,
                    stop_reason=StopReason.end_turn,
                    usage=Usage(input_tokens=1, output_tokens=1),
                )
            finally:
                active_calls -= 1

    class PromptSession:
        previous: str | None = None

        async def prompt_async(self, prompt: str) -> str:
            assert prompt == "repl> "
            if self.previous == "queued request 2":
                waiting_after_queued.set()
            result = await inputs.get()
            self.previous = result
            return result

    class InertTerminal:
        def __init__(self, session: PromptSession) -> None:
            del session

        async def start(self) -> None:
            pass

        async def wait_failed(self) -> None:
            await asyncio.Future()

        async def close(self) -> None:
            pass

    def make_prompt_session(
        status: object,
        *,
        on_interrupt: Callable[[], None],
        on_shutdown: Callable[[], None],
        recall_pending: Callable[[], Awaitable[str | None]],
        on_empty_eof: Callable[[float], bool],
        capture_target: Callable[[], str],
        reserve_sequence: Callable[[], int],
    ) -> PromptSession:
        del on_empty_eof, on_shutdown, recall_pending, reserve_sequence, status, capture_target
        interrupt_callbacks.append(on_interrupt)
        return PromptSession()

    async def wait_for_output(fragment: str) -> str:
        output = ""
        for _ in range(100):
            output += capsys.readouterr().out
            if fragment in output:
                return output
            await asyncio.sleep(0.01)
        pytest.fail(f"output did not contain {fragment!r}: {output!r}")

    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    monkeypatch.setattr(axio_repl, "MAX_PENDING_INPUTS", 4)
    monkeypatch.setattr(axio_repl, "_select_transport", lambda _name: (BlockingTransport, ""))
    monkeypatch.setattr(axio_repl._panel, "make_session", make_prompt_session)
    monkeypatch.setattr(axio_repl._panel, "commit_history", lambda _session, _texts: None)
    monkeypatch.setattr(axio_repl, "TerminalUI", InertTerminal)
    resolve_local_agent_id = axio_repl._resolve_local_agent_id
    monkeypatch.setattr(
        axio_repl,
        "_resolve_local_agent_id",
        lambda value: "child" if value == "child" else resolve_local_agent_id(value),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["axio-repl", "--sandbox", "none", "--no-session-log"],
    )

    await inputs.put("first request")
    repl_task = asyncio.create_task(main())
    try:
        await asyncio.wait_for(first_started.wait(), timeout=1)
        await inputs.put("/agent-focus child")
        await inputs.put("/agent-focus main")
        await inputs.put("/help")
        output = await wait_for_output("Type your request. Tools:")
        assert "answer 1" not in output
        assert "Focused agent: \x1b[1mchild" not in output

        await inputs.put("queued request 1")
        await inputs.put("queued request 2")
        await asyncio.wait_for(waiting_after_queued.wait(), timeout=1)
        interrupt_callbacks[0]()
        interrupt_callbacks[0]()
        await asyncio.wait_for(second_started.wait(), timeout=1)
        second_call = observed_calls[1]
        first_index = second_call.index("queued request 1")
        assert second_call[first_index : first_index + 2] == ["queued request 1", "queued request 2"]
        assert second_call[-1].startswith("[Turn ")
        assert max_active_calls == 1

        await inputs.put("/quit")
        await asyncio.sleep(0)
        assert not repl_task.done()
        await asyncio.wait_for(repl_task, timeout=1)
    finally:
        release_first.set()
        if not repl_task.done():
            await inputs.put("/quit")
            repl_task.cancel()
            await asyncio.gather(repl_task, return_exceptions=True)


async def test_input_preempts_blocking_tool_and_actual_result_arrives_later(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import axio_repl

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    third_started = asyncio.Event()
    inputs: asyncio.Queue[str] = asyncio.Queue()
    calls: list[list[Message]] = []

    async def slow_tool() -> str:
        tool_started.set()
        await release_tool.wait()
        return "actual tool result"

    class ToolTransport:
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
            del tools, system
            calls.append(list(messages))
            call_number = len(calls)
            if call_number == 1:
                yield ToolUseStart(index=0, tool_use_id="slow-call", name="slow")
                yield ToolInputDelta(index=0, tool_use_id="slow-call", partial_json="{}")
                yield IterationEnd(
                    iteration=1,
                    stop_reason=StopReason.tool_use,
                    usage=Usage(input_tokens=1, output_tokens=1),
                )
                return
            if call_number == 2:
                second_started.set()
                await release_second.wait()
            else:
                third_started.set()
            yield TextDelta(index=0, delta=f"answer {call_number}")
            yield IterationEnd(
                iteration=call_number,
                stop_reason=StopReason.end_turn,
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    class PromptSession:
        async def prompt_async(self, prompt: str) -> str:
            assert prompt == "repl> "
            return await inputs.get()

    class InertTerminal:
        def __init__(self, session: PromptSession) -> None:
            del session

        async def start(self) -> None:
            pass

        async def wait_failed(self) -> None:
            await asyncio.Future()

        async def close(self) -> None:
            pass

    def make_prompt_session(
        status: object,
        *,
        on_interrupt: Callable[[], None],
        on_shutdown: Callable[[], None],
        recall_pending: Callable[[], Awaitable[str | None]],
        on_empty_eof: Callable[[float], bool],
        capture_target: Callable[[], str],
        reserve_sequence: Callable[[], int],
    ) -> PromptSession:
        del capture_target, on_empty_eof, on_interrupt, on_shutdown, recall_pending, reserve_sequence, status
        return PromptSession()

    async def build_tools(*args: object, **kwargs: object) -> tuple[list[Tool[object]], str, Path, str]:
        del args, kwargs
        return [Tool(name="slow", handler=slow_tool)], "test", tmp_path, ""

    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    monkeypatch.setattr(axio_repl, "_select_transport", lambda _name: (ToolTransport, ""))
    monkeypatch.setattr(axio_repl._sandbox, "build_tools", build_tools)
    monkeypatch.setattr(axio_repl._panel, "make_session", make_prompt_session)
    monkeypatch.setattr(axio_repl._panel, "commit_history", lambda _session, _texts: None)
    monkeypatch.setattr(axio_repl, "TerminalUI", InertTerminal)
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--sandbox", "none", "--no-session-log"])

    await inputs.put("start slow tool")
    repl_task = asyncio.create_task(main())
    try:
        await asyncio.wait_for(tool_started.wait(), timeout=1)
        await inputs.put("message before tool result")
        await asyncio.wait_for(second_started.wait(), timeout=1)

        tool_use_index = next(
            index
            for index, message in enumerate(calls[1])
            if any(isinstance(block, ToolUseBlock) for block in message.content)
        )
        placeholder = calls[1][tool_use_index + 1]
        assert any(
            isinstance(block, ToolResultBlock)
            and block.tool_use_id == "slow-call"
            and "continues after interruption" in str(block.content)
            for block in placeholder.content
        )
        assert any(
            any(
                isinstance(block, TextBlock) and block.text == "message before tool result"
                for block in message.content
            )
            for message in calls[1][tool_use_index + 2 :]
        )

        release_tool.set()
        await asyncio.sleep(0)
        assert len(calls) == 2
        release_second.set()
        await asyncio.wait_for(third_started.wait(), timeout=1)
        assert any(
            any(
                isinstance(block, TextBlock) and "Deferred tool completed: name=slow, call_id=slow-call" in block.text
                for block in message.content
            )
            for message in calls[2]
        )
        assert (
            sum(
                isinstance(block, ToolResultBlock) and block.tool_use_id == "slow-call"
                for message in calls[2]
                for block in message.content
            )
            == 1
        )

        await inputs.put("/quit")
        await asyncio.wait_for(repl_task, timeout=1)
    finally:
        release_tool.set()
        release_second.set()
        if not repl_task.done():
            repl_task.cancel()
            await asyncio.gather(repl_task, return_exceptions=True)


async def test_double_eof_shutdown_recovers_active_turn_pending_input_and_editor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import axio_repl

    turn_started = asyncio.Event()
    turn_cancelled = asyncio.Event()
    inputs: asyncio.Queue[str | BaseException] = asyncio.Queue()
    prompt_calls = 0

    class BlockingTransport:
        name = "stub"

        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.model = ModelSpec(id="stub/model", capabilities=frozenset({Capability.text}))
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
            turn_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                turn_cancelled.set()
                raise
            if False:
                yield TextDelta(index=0, delta="unreachable")

    class Buffer:
        text = ""

    class PromptSession:
        default_buffer = Buffer()

        async def prompt_async(self, prompt: str, **kwargs: object) -> str:
            nonlocal prompt_calls
            assert prompt == "repl> "
            prompt_calls += 1
            if default := kwargs.get("default"):
                self.default_buffer.text = str(default)
            value = await inputs.get()
            if isinstance(value, BaseException):
                raise value
            self.default_buffer.text = ""
            return value

    class InertTerminal:
        def __init__(self, session: PromptSession) -> None:
            del session

        async def start(self) -> None:
            pass

        async def wait_failed(self) -> None:
            await asyncio.Future()

        async def close(self) -> None:
            pass

    prompt_session = PromptSession()

    def make_prompt_session(
        status: object,
        *,
        on_interrupt: Callable[[], None],
        on_shutdown: Callable[[], None],
        recall_pending: Callable[[], Awaitable[str | None]],
        on_empty_eof: Callable[[float], bool],
        capture_target: Callable[[], str],
        reserve_sequence: Callable[[], int],
    ) -> PromptSession:
        del capture_target, status, on_interrupt, on_shutdown, recall_pending, on_empty_eof, reserve_sequence
        return prompt_session

    journal_root = tmp_path / "journals"
    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    monkeypatch.setattr(axio_repl, "_select_transport", lambda _name: (BlockingTransport, ""))
    monkeypatch.setattr(axio_repl._panel, "make_session", make_prompt_session)
    monkeypatch.setattr(axio_repl._panel, "commit_history", lambda _session, _texts: None)
    monkeypatch.setattr(axio_repl, "TerminalUI", InertTerminal)
    monkeypatch.setattr(
        sys,
        "argv",
        ["axio-repl", "--sandbox", "none", "--session-log-dir", str(journal_root)],
    )

    await inputs.put("active request")
    repl_task = asyncio.create_task(main())
    await asyncio.wait_for(turn_started.wait(), timeout=1)
    await inputs.put("pending request")
    for _ in range(100):
        if prompt_calls >= 3:
            break
        await asyncio.sleep(0.01)
    assert prompt_calls >= 3
    prompt_session.default_buffer.text = "unsent editor"
    await inputs.put(EOFError())

    await asyncio.wait_for(repl_task, timeout=2)
    await asyncio.wait_for(turn_cancelled.wait(), timeout=1)

    events_paths = list(journal_root.rglob("events.jsonl"))
    assert len(events_paths) == 1
    recovered = materialize_recovery(events_paths[0])
    assert [entry.text for entry in recovered.pending_inputs] == ["pending request"]
    assert recovered.editor_text == "unsent editor"
    assert any(
        isinstance(block, TextBlock) and "Recorded reason: double_eof" in block.text
        for message in recovered.messages
        for block in message.content
    )


async def test_resume_copies_context_and_restores_editor_before_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import axio_repl

    source = await SessionJournal.open(session_id="source", root=tmp_path / "source")
    assert await source.publish(
        "message_committed",
        {
            "hub_seq": 1,
            "run_id": "source-run",
            "message": Message(role="user", content=[TextBlock(text="source history")]),
        },
        agent_id="main",
        turn_id="source-turn",
        context_id="source-context",
        execution_mode="foreground",
    )
    assert await source.publish(
        "editor_snapshot",
        {
            "hub_seq": 2,
            "run_id": "source-run",
            "event": EditorSnapshot("restored editor"),
        },
        agent_id="main",
        context_id="source-context",
        execution_mode="foreground",
    )
    await source.close()

    class IdleTransport:
        name = "stub"

        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.model = ModelSpec(id="stub/model", capabilities=frozenset({Capability.text}))
            self.models = ModelRegistry([self.model])
            self.temperature: float | None = None
            self.max_output_tokens: int | None = None
            self.debug = False

        async def fetch_models(self) -> None:
            pass

    class Buffer:
        text = ""

    class PromptSession:
        default_buffer = Buffer()

        async def prompt_async(self, prompt: str, **kwargs: object) -> str:
            assert prompt == "repl> "
            assert kwargs == {"default": "restored editor"}
            self.default_buffer.text = "restored editor"
            raise EOFError

    class InertTerminal:
        def __init__(self, session: PromptSession) -> None:
            del session

        async def start(self) -> None:
            pass

        async def wait_failed(self) -> None:
            await asyncio.Future()

        async def close(self) -> None:
            pass

    prompt_session = PromptSession()

    def make_prompt_session(
        status: object,
        *,
        on_interrupt: Callable[[], None],
        on_shutdown: Callable[[], None],
        recall_pending: Callable[[], Awaitable[str | None]],
        on_empty_eof: Callable[[float], bool],
        capture_target: Callable[[], str],
        reserve_sequence: Callable[[], int],
    ) -> PromptSession:
        del capture_target, status, on_interrupt, on_shutdown, recall_pending, on_empty_eof, reserve_sequence
        return prompt_session

    resumed_root = tmp_path / "resumed"
    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    monkeypatch.setattr(axio_repl, "_select_transport", lambda _name: (IdleTransport, ""))
    monkeypatch.setattr(axio_repl._panel, "make_session", make_prompt_session)
    monkeypatch.setattr(axio_repl, "TerminalUI", InertTerminal)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axio-repl",
            "--sandbox",
            "none",
            "--resume",
            str(source.events_path),
            "--session-log-dir",
            str(resumed_root),
        ],
    )

    await asyncio.wait_for(main(), timeout=2)

    resumed_paths = list(resumed_root.rglob("events.jsonl"))
    assert len(resumed_paths) == 1
    records = read_journal(resumed_paths[0]).records
    assert any(record.get("kind") == "recovery_applied" for record in records)
    recovered = materialize_recovery(resumed_paths[0])
    assert recovered.messages[0].content == [TextBlock(text="source history")]
    assert recovered.editor_text == "restored editor"
