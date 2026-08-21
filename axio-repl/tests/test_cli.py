from __future__ import annotations

import asyncio
import sys
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import pytest
from axio.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from axio.events import IterationEnd, StreamEvent, TextDelta, ToolInputDelta, ToolOutputDelta, ToolResult, ToolUseStart
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.types import StopReason, Usage
from axio_tools_agents.runtime import ConfigurationChanged, EditorSnapshot, RuntimeEvent
from prompt_toolkit.formatted_text import to_plain_text

from axio_repl import (
    TOOLS,
    ReplRenderer,
    _build_argument_parser,
    _cancel_and_settle_tasks,
    _capture_command_output,
    _handle_agent_actions,
    _IncomingPrompt,
    _pending_prompt_count,
    _retain_interrupted_partial,
    _select_configured_tools,
    _show_agents,
    _show_model,
    main,
)
from axio_repl._coordinator import PendingInputCoordinator
from axio_repl._journal import SEMANTIC_FILENAME, SessionJournal, read_journal
from axio_repl._multiplexer import ActionMultiplexer, DisplayMode
from axio_repl._recovery import materialize_recovery
from axio_repl._theme import DEFAULT_THEME, MONOCHROME_THEME, NO_COLOR_THEME, TerminalTheme


def test_agent_actions_default_to_off() -> None:
    args = _build_argument_parser().parse_args([])

    assert args.agent_actions == "off"


def test_powerline_defaults_to_off_and_can_be_enabled() -> None:
    parser = _build_argument_parser()

    assert parser.parse_args([]).powerline is False
    assert parser.parse_args(["--powerline", "inspect"]).powerline is True
    assert parser.parse_args(["--no-powerline"]).powerline is False


def test_theme_defaults_and_builtin_selection() -> None:
    parser = _build_argument_parser()

    assert parser.parse_args([]).theme == "default"
    assert parser.parse_args(["--theme", "monochrome"]).theme == "monochrome"


def test_theme_rejects_unknown_name_with_available_choices(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _build_argument_parser().parse_args(["--theme", "unknown"])

    error = capsys.readouterr().err
    assert "invalid choice: 'unknown'" in error
    assert "default" in error
    assert "monochrome" in error


def test_repeated_tool_preemption_does_not_retain_unconsumable_partials() -> None:
    partials: dict[str, str] = {}
    pending_interrupt_turns: set[str] = set()

    for index in range(1000):
        _retain_interrupted_partial(
            partials,
            pending_interrupt_turns,
            turn_id=f"preempted-{index}",
            partial=f"partial {index}",
            preemption_reason="queued user input",
        )

    assert partials == {}

    pending_interrupt_turns.add("escape-turn")
    _retain_interrupted_partial(
        partials,
        pending_interrupt_turns,
        turn_id="escape-turn",
        partial="visible partial",
        preemption_reason="queued user input",
    )
    assert partials.pop("escape-turn") == "visible partial"
    pending_interrupt_turns.discard("escape-turn")
    assert partials == {}


@pytest.mark.parametrize(
    "theme",
    [DEFAULT_THEME, MONOCHROME_THEME, NO_COLOR_THEME],
    ids=("default", "monochrome", "no-color"),
)
def test_command_feedback_uses_the_active_presentation(
    monkeypatch: pytest.MonkeyPatch,
    theme: TerminalTheme,
) -> None:
    model = ModelSpec(id="stub/model", capabilities=frozenset({Capability.text}))
    transport = type("Transport", (), {"model": model, "models": ModelRegistry([model])})()
    renderer = ReplRenderer(theme=theme)
    monkeypatch.setattr("axio_repl.local_background_agent_records", lambda: [])

    with _capture_command_output(theme) as output:
        _show_model(transport)
        _show_agents(renderer)

    feedback = "".join(output)
    assert "Current model:" in feedback
    assert "Focused agent:" in feedback
    if theme is NO_COLOR_THEME:
        assert "\x1b[" not in feedback
    else:
        assert f"{theme.command.ansi}stub/model{theme.reset}" in feedback
        assert f"{theme.command.ansi}main{theme.reset}" in feedback
        assert theme.command.ansi == DEFAULT_THEME.command.ansi


async def test_invalid_config_theme_fails_before_transport_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import axio_repl

    (tmp_path / "config.yaml").write_text(
        "version: 1\ndefaults:\n  runtime:\n    theme: unknown\n",
        encoding="utf-8",
    )
    transport_selected = False

    def select_transport(_name: str | None) -> object:
        nonlocal transport_selected
        transport_selected = True
        raise AssertionError("transport must not be selected")

    monkeypatch.setattr(axio_repl, "_select_transport", select_transport)
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--config-dir", str(tmp_path)])

    with pytest.raises(SystemExit):
        await main()

    assert transport_selected is False


def test_agent_actions_can_be_enabled() -> None:
    args = _build_argument_parser().parse_args(["--agent-actions", "on", "inspect"])

    assert args.agent_actions == "on"
    assert args.prompt == "inspect"


def test_agent_actions_rejects_unknown_modes() -> None:
    with pytest.raises(SystemExit):
        _build_argument_parser().parse_args(["--agent-actions", "verbose"])


def test_session_replay_is_disabled_by_default_and_has_explicit_toggle() -> None:
    assert _build_argument_parser().parse_args([]).session_replay is False
    assert _build_argument_parser().parse_args(["--session-replay"]).session_replay is True
    assert _build_argument_parser().parse_args(["--no-session-replay"]).session_replay is False


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


def test_agent_configuration_flags_are_parsed_without_abbreviations(tmp_path: Path) -> None:
    args = _build_argument_parser().parse_args(
        [
            "--agent",
            "local",
            "--config-dir",
            str(tmp_path),
            "--transport-base-url",
            "http://127.0.0.1:18080/v1",
            "--transport-api-key-env",
            "LOCAL_TOKEN",
            "--tools",
            "read_file,shell",
            "--no-debug",
            "--session-log",
        ]
    )

    assert args.agent == "local"
    assert args.config_dir == tmp_path
    assert args.transport_base_url == "http://127.0.0.1:18080/v1"
    assert args.transport_api_key_env == "LOCAL_TOKEN"
    assert args.tools == "read_file,shell"
    assert args.debug is False
    assert args.no_session_log is False
    with pytest.raises(SystemExit):
        _build_argument_parser().parse_args(["--transp", "llama-cpp"])


def test_configured_tools_support_all_none_and_named_subsets() -> None:
    assert _select_configured_tools(None) == TOOLS
    assert _select_configured_tools("all") == TOOLS
    assert _select_configured_tools("none") == []
    assert [tool.name for tool in _select_configured_tools(("shell", "read_file"))] == ["read_file", "shell"]

    with pytest.raises(ValueError, match="unknown tool"):
        _select_configured_tools("missing")
    with pytest.raises(ValueError, match="cannot be combined"):
        _select_configured_tools("all,shell")


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
    status_callbacks: list[Callable[[], str]] = []
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

    class Buffer:
        text = ""

        def reset(self) -> None:
            self.text = ""

    class PromptSession:
        def __init__(self, capture_target: Callable[[], str], reserve_sequence: Callable[[], int]) -> None:
            self.previous: str | None = None
            self.default_buffer = Buffer()
            self._capture_target = capture_target
            self._reserve_sequence = reserve_sequence

        async def prompt_async(self, prompt: object) -> str:
            assert to_plain_text(cast(Any, prompt)).endswith("> ")
            if self.previous == "queued request 2":
                waiting_after_queued.set()
            result = await inputs.get()
            self.previous = result
            self.default_buffer.text = result
            setattr(self.default_buffer, "_axio_accepted_target", self._capture_target())
            if result.strip():
                setattr(self.default_buffer, "_axio_accepted_seq", self._reserve_sequence())
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
        theme: object,
    ) -> PromptSession:
        del on_empty_eof, on_shutdown, recall_pending, theme
        status_callbacks.append(cast(Callable[[], str], status))
        interrupt_callbacks.append(on_interrupt)
        return PromptSession(capture_target, reserve_sequence)

    async def wait_for_panel(fragment: str) -> str:
        for _ in range(100):
            panel = status_callbacks[0]() if status_callbacks else ""
            if fragment in panel:
                return panel
            await asyncio.sleep(0.01)
        pytest.fail(f"panel did not contain {fragment!r}")

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
        ["axio-repl", "--sandbox", "none", "--session-log-dir", str(tmp_path / "journals")],
    )

    await inputs.put("first request")
    repl_task = asyncio.create_task(main())
    try:
        await asyncio.wait_for(first_started.wait(), timeout=1)
        await inputs.put("/agent-focus child")
        await inputs.put("/agent-focus main")
        await inputs.put("/help")
        panel = await wait_for_panel("Type your request. Tools:")
        output = capsys.readouterr().out
        assert "Commands: /help" in panel
        assert "Type your request. Tools:" not in output
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
        assert not any(text.startswith("/") for call in observed_calls for text in call)

        await inputs.put("/quit")
        await asyncio.sleep(0)
        assert not repl_task.done()
        await asyncio.wait_for(repl_task, timeout=1)
        journal_paths = list((tmp_path / "journals").rglob(SEMANTIC_FILENAME))
        assert len(journal_paths) == 1
        records = read_journal(journal_paths[0]).records
        assert all(
            command not in repr(record)
            for record in records
            for command in ("/help", "/agent-focus child", "/agent-focus main")
        )
        for record in records:
            if record["kind"] != "input_received":
                continue
            payload = record["payload"]
            assert isinstance(payload, dict)
            event = payload.get("event")
            assert isinstance(event, dict)
            assert event.get("text") != "/help"
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
    status_callbacks: list[Callable[[], str]] = []

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
        async def prompt_async(self, prompt: object) -> str:
            assert to_plain_text(cast(Any, prompt)).endswith("> ")
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
        theme: object,
    ) -> PromptSession:
        del capture_target, on_empty_eof, on_interrupt, on_shutdown, recall_pending, reserve_sequence, theme
        status_callbacks.append(cast(Callable[[], str], status))
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
        await inputs.put("/help")
        for _ in range(100):
            if status_callbacks and "Type your request. Tools:" in status_callbacks[0]():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("help feedback did not reach the panel")
        assert not second_started.is_set()

        await inputs.put("message before tool result")
        await asyncio.wait_for(second_started.wait(), timeout=1)
        assert "Main turn preempted for queued user input" in status_callbacks[0]()

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


async def test_queued_input_cancels_unstarted_dispatch_then_reissued_shell_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import axio_repl

    first_tool_ready = asyncio.Event()
    queued_input_admitted = asyncio.Event()
    allow_first_iteration_end = asyncio.Event()
    shell_stream_started = asyncio.Event()
    shell_output_rendered = asyncio.Event()
    release_shell = asyncio.Event()
    final_response_started = asyncio.Event()
    inputs: asyncio.Queue[str] = asyncio.Queue()
    calls: list[list[Message]] = []
    rendered_tool_results: list[ToolResult] = []
    observed_renderer: ReplRenderer | None = None
    shell_stream_invocations = 0

    async def shell_handler() -> str:
        raise AssertionError("streaming shell must use its stream handler")

    async def shell_stream() -> AsyncIterator[tuple[str, str]]:
        nonlocal shell_stream_invocations
        shell_stream_invocations += 1
        shell_stream_started.set()
        yield "stdout", "early shell output\n"
        await release_shell.wait()
        yield "stdout", "late shell output\n"

    shell_handler.stream = shell_stream  # type: ignore[attr-defined]

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
                yield ToolUseStart(index=0, tool_use_id="cancelled-shell-a", name="shell")
                yield ToolInputDelta(index=0, tool_use_id="cancelled-shell-a", partial_json="{}")
                yield ToolUseStart(index=1, tool_use_id="cancelled-shell-b", name="shell")
                yield ToolInputDelta(index=1, tool_use_id="cancelled-shell-b", partial_json="{}")
                first_tool_ready.set()
                await allow_first_iteration_end.wait()
                yield IterationEnd(
                    iteration=1,
                    stop_reason=StopReason.tool_use,
                    usage=Usage(input_tokens=1, output_tokens=1),
                )
                return
            if call_number == 2:
                yield ToolUseStart(index=0, tool_use_id="reissued-shell", name="shell")
                yield ToolInputDelta(index=0, tool_use_id="reissued-shell", partial_json="{}")
                yield IterationEnd(
                    iteration=2,
                    stop_reason=StopReason.tool_use,
                    usage=Usage(input_tokens=1, output_tokens=1),
                )
                return
            final_response_started.set()
            yield TextDelta(index=0, delta="done")
            yield IterationEnd(
                iteration=3,
                stop_reason=StopReason.end_turn,
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    class PromptSession:
        async def prompt_async(self, prompt: object) -> str:
            assert to_plain_text(cast(Any, prompt)).endswith("> ")
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
        theme: object,
    ) -> PromptSession:
        del capture_target, on_empty_eof, on_interrupt, on_shutdown, recall_pending, reserve_sequence, status, theme
        return PromptSession()

    async def build_tools(*args: object, **kwargs: object) -> tuple[list[Tool[object]], str, Path, str]:
        del args, kwargs
        return [Tool(name="shell", handler=shell_handler)], "test", tmp_path, ""

    original_admit = PendingInputCoordinator.admit

    async def tracked_admit(
        coordinator: object,
        text: str,
        target_agent_id: str,
        *,
        reserved_seq: int | None = None,
    ) -> object:
        entry = await original_admit(
            cast(Any, coordinator),
            text,
            target_agent_id,
            reserved_seq=reserved_seq,
        )
        if text == "queued while model is generating":
            queued_input_admitted.set()
        return entry

    original_render_runtime_event = axio_repl.render_runtime_event

    async def tracked_render_runtime_event(renderer: ReplRenderer, envelope: object) -> None:
        nonlocal observed_renderer
        observed_renderer = renderer
        await original_render_runtime_event(renderer, cast(Any, envelope))
        event = cast(Any, envelope).event
        if isinstance(event, ToolResult):
            rendered_tool_results.append(event)
        if isinstance(event, ToolOutputDelta) and event.tool_use_id == "reissued-shell":
            shell_output_rendered.set()

    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    monkeypatch.setattr(axio_repl, "_select_transport", lambda _name: (ToolTransport, ""))
    monkeypatch.setattr(axio_repl._sandbox, "build_tools", build_tools)
    monkeypatch.setattr(axio_repl._panel, "make_session", make_prompt_session)
    monkeypatch.setattr(axio_repl._panel, "commit_history", lambda _session, _texts: None)
    monkeypatch.setattr(axio_repl, "TerminalUI", InertTerminal)
    monkeypatch.setattr(PendingInputCoordinator, "admit", tracked_admit)
    monkeypatch.setattr(axio_repl, "render_runtime_event", tracked_render_runtime_event)
    monkeypatch.setattr(sys, "argv", ["axio-repl", "--sandbox", "none", "--no-session-log"])

    await inputs.put("start")
    repl_task = asyncio.create_task(main())
    try:
        await asyncio.wait_for(first_tool_ready.wait(), timeout=1)
        await inputs.put("queued while model is generating")
        await asyncio.wait_for(queued_input_admitted.wait(), timeout=1)
        allow_first_iteration_end.set()

        await asyncio.wait_for(shell_stream_started.wait(), timeout=1)
        await asyncio.wait_for(shell_output_rendered.wait(), timeout=1)
        assert shell_stream_invocations == 1
        first_output = capsys.readouterr().out
        assert "early shell output" in first_output
        assert first_output.count("▶ shell #001") == 1
        assert first_output.count("✗ shell #001") == 1
        assert first_output.count("▶ shell #002") == 1
        assert first_output.count("✗ shell #002") == 1
        assert first_output.index("▶ shell #001") < first_output.index("✗ shell #001")
        assert first_output.index("▶ shell #002") < first_output.index("✗ shell #002")
        assert first_output.count("[interrupted by user]") == 2
        assert observed_renderer is not None
        assert observed_renderer.active_tool_call_count == 1

        first_cancelled_results = [
            block
            for message in calls[1]
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.tool_use_id.startswith("cancelled-shell-")
        ]
        assert [(block.tool_use_id, block.content, block.is_error) for block in first_cancelled_results] == [
            ("cancelled-shell-a", "[interrupted by user]", True),
            ("cancelled-shell-b", "[interrupted by user]", True),
        ]
        assert [
            (event.tool_use_id, event.name, event.content, event.is_error)
            for event in rendered_tool_results
            if event.tool_use_id.startswith("cancelled-shell-")
        ] == [
            ("cancelled-shell-a", "shell", "[interrupted by user]", True),
            ("cancelled-shell-b", "shell", "[interrupted by user]", True),
        ]
        assert not any("continues after interruption" in str(block.content) for block in first_cancelled_results)
        assert any(
            isinstance(block, TextBlock) and block.text == "queued while model is generating"
            for message in calls[1]
            for block in message.content
        )

        release_shell.set()
        await asyncio.wait_for(final_response_started.wait(), timeout=1)
        final_output = capsys.readouterr().out
        assert final_output.count("✓ shell #003") == 1
        assert observed_renderer.active_tool_call_count == 0
        assert shell_stream_invocations == 1
        assert [event.tool_use_id for event in rendered_tool_results].count("cancelled-shell-a") == 1
        assert [event.tool_use_id for event in rendered_tool_results].count("cancelled-shell-b") == 1
        assert not any(
            isinstance(block, TextBlock) and "Deferred tool completed" in block.text
            for call in calls
            for message in call
            for block in message.content
        )

        await inputs.put("/quit")
        await asyncio.wait_for(repl_task, timeout=1)
    finally:
        allow_first_iteration_end.set()
        release_shell.set()
        if not repl_task.done():
            repl_task.cancel()
            await asyncio.gather(repl_task, return_exceptions=True)


async def test_double_eof_drains_active_and_pending_turns_before_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import axio_repl

    turn_started = asyncio.Event()
    release_active = asyncio.Event()
    pending_started = asyncio.Event()
    release_pending_tool = asyncio.Event()
    inputs: asyncio.Queue[str | BaseException] = asyncio.Queue()
    prompt_calls = 0
    calls: list[list[Message]] = []

    class DrainingTransport:
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
            del tools, system
            calls.append(list(messages))
            if len(calls) == 1:
                turn_started.set()
                await release_active.wait()
            elif len(calls) == 2:
                yield ToolUseStart(index=0, tool_use_id="drain-call", name="drain_tool")
                yield ToolInputDelta(index=0, tool_use_id="drain-call", partial_json="{}")
                yield IterationEnd(
                    iteration=2,
                    stop_reason=StopReason.tool_use,
                    usage=Usage(input_tokens=1, output_tokens=1),
                )
                return
            yield TextDelta(index=0, delta=f"answer {len(calls)}")
            yield IterationEnd(
                iteration=len(calls),
                stop_reason=StopReason.end_turn,
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    class Buffer:
        text = ""

    class PromptSession:
        default_buffer = Buffer()

        async def prompt_async(self, prompt: object, **kwargs: object) -> str:
            nonlocal prompt_calls
            assert to_plain_text(cast(Any, prompt)).endswith("> ")
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

    async def drain_tool() -> str:
        pending_started.set()
        await release_pending_tool.wait()
        return "drained tool result"

    async def build_tools(*args: object, **kwargs: object) -> tuple[list[Tool[object]], str, Path, str]:
        del args, kwargs
        return [Tool(name="drain_tool", handler=drain_tool)], "test", tmp_path, ""

    def make_prompt_session(
        status: object,
        *,
        on_interrupt: Callable[[], None],
        on_shutdown: Callable[[], None],
        recall_pending: Callable[[], Awaitable[str | None]],
        on_empty_eof: Callable[[float], bool],
        capture_target: Callable[[], str],
        reserve_sequence: Callable[[], int],
        theme: object,
    ) -> PromptSession:
        del capture_target, on_empty_eof, on_interrupt, on_shutdown
        del recall_pending, reserve_sequence, status, theme
        return prompt_session

    journal_root = tmp_path / "journals"
    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    monkeypatch.setattr(axio_repl, "_select_transport", lambda _name: (DrainingTransport, ""))
    monkeypatch.setattr(axio_repl._sandbox, "build_tools", build_tools)
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
    prompt_calls_at_eof = prompt_calls

    await asyncio.sleep(0)
    assert not repl_task.done()
    release_active.set()
    await asyncio.wait_for(pending_started.wait(), timeout=1)
    assert not repl_task.done()
    release_pending_tool.set()

    await asyncio.wait_for(repl_task, timeout=2)
    assert prompt_calls == prompt_calls_at_eof
    assert len(calls) == 3
    assert any(
        isinstance(block, TextBlock) and block.text == "pending request"
        for message in calls[1]
        for block in message.content
    )

    events_paths = list(journal_root.rglob(SEMANTIC_FILENAME))
    assert len(events_paths) == 1
    recovered = materialize_recovery(events_paths[0])
    assert recovered.pending_inputs == ()
    assert recovered.editor_text == "unsent editor"
    shutdown = next(
        record for record in read_journal(events_paths[0]).records if record["kind"] == "shutdown_recorded"
    )
    shutdown_payload = shutdown["payload"]
    assert isinstance(shutdown_payload, dict)
    shutdown_event = shutdown_payload["event"]
    assert isinstance(shutdown_event, dict)
    assert shutdown_event["reason"] == "double_eof"


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

    prompt_calls = 0

    class PromptSession:
        default_buffer = Buffer()

        async def prompt_async(self, prompt: object, **kwargs: object) -> str:
            nonlocal prompt_calls
            prompt_calls += 1
            assert to_plain_text(cast(Any, prompt)).endswith("> ")
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
        theme: object,
    ) -> PromptSession:
        del capture_target, on_empty_eof, on_interrupt, on_shutdown
        del recall_pending, reserve_sequence, status, theme
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

    assert prompt_calls == 1
    resumed_paths = list(resumed_root.rglob(SEMANTIC_FILENAME))
    assert len(resumed_paths) == 1
    records = read_journal(resumed_paths[0]).records
    assert any(record.get("kind") == "recovery_applied" for record in records)
    recovered = materialize_recovery(resumed_paths[0])
    assert recovered.messages[0].content == [TextBlock(text="source history")]
    assert recovered.editor_text == "restored editor"
