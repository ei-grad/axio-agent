import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from axio import background
from axio.models import ModelSpec
from axio.types import Usage

from axio_repl import _panel
from axio_repl._theme import MONOCHROME_THEME, NO_COLOR_THEME


class _Record:
    def __init__(self, agent_id: str, name: str) -> None:
        self.id = agent_id
        self.name = name


@pytest.fixture(autouse=True)
async def clean_registry() -> AsyncGenerator[None, None]:
    yield
    await background.cancel_all()


def test_silent_when_nothing_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", list)
    # A status line that always says something wastes a row of terminal.
    assert _panel.agent_summary() == ""


def test_counts_agents_by_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _panel,
        "local_background_agent_records",
        lambda: [_Record("a", "one"), _Record("b", "two"), _Record("c", "three")],
    )
    states = {"a": ("running", None), "b": ("running", None), "c": ("idle", None)}
    monkeypatch.setattr(_panel, "background_agent_state", lambda i: states[i])
    assert _panel.agent_summary() == "agents: 1 idle, 2 running"


def test_names_the_failed_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", lambda: [_Record("a", "analyst")])
    monkeypatch.setattr(_panel, "background_agent_state", lambda i: ("idle", "StreamError: boom"))
    summary = _panel.agent_summary()
    assert "failed: analyst" in summary


@pytest.mark.asyncio
async def test_reports_detached_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", list)

    started = asyncio.Event()

    async def slow() -> str:
        started.set()
        await asyncio.sleep(60)
        return "never"

    background.start("shell", slow())
    await started.wait()
    assert "tasks: 1 running" in _panel.agent_summary()


@pytest.mark.asyncio
async def test_finished_calls_are_flagged_until_collected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_panel, "local_background_agent_records", list)

    async def quick() -> str:
        return "done"

    handle = background.start("shell", quick())
    await background.get(handle).task  # type: ignore[union-attr]
    assert "1 ready to collect" in _panel.agent_summary()

    background.describe(handle)  # reading it marks it collected
    assert "ready to collect" not in _panel.agent_summary()


_M3 = ModelSpec(id="MiniMaxAI/MiniMax-M3", context_window=1_000_000, input_cost=0.3, output_cost=1.2)


def test_a_fresh_session_still_says_which_model() -> None:
    line = _panel.status_line(_M3, _panel.SessionStats())

    assert "MiniMaxAI/MiniMax-M3" in line
    assert "ctx 0/1M" in line
    assert "$0" in line


def test_tokens_of_every_agent_are_counted() -> None:
    stats = _panel.SessionStats()

    stats.record("main", Usage(88_622, 5_169), _M3)
    stats.record("child-1", Usage(19_666, 1_908), _M3)

    assert stats.input_tokens == 108_288
    assert stats.output_tokens == 7_077
    assert "108.3k in / 7.1k out" in _panel.status_line(_M3, stats)


def test_only_the_main_agent_sets_how_full_the_window_is() -> None:
    # A child's prompt is its own context, not the one the user is filling.
    stats = _panel.SessionStats()

    stats.record("main", Usage(30_000, 10), _M3)
    stats.record("child-1", Usage(900_000, 10), _M3)

    assert stats.context_tokens == 30_000


def test_the_window_shrinks_when_the_context_does() -> None:
    # Compaction is the point: the latest prompt is what occupies the window,
    # not the largest one ever sent.
    stats = _panel.SessionStats()

    stats.record("main", Usage(500_000, 10), _M3)
    stats.record("main", Usage(20_000, 10), _M3)

    assert stats.context_tokens == 20_000


def test_cost_follows_the_model_that_was_charged() -> None:
    stats = _panel.SessionStats()
    cheap = ModelSpec(id="cheap", input_cost=0.1, output_cost=0.2)

    stats.record("main", Usage(1_000_000, 1_000_000), _M3)
    stats.record("child", Usage(1_000_000, 1_000_000), cheap)

    assert stats.cost == pytest.approx(0.3 + 1.2 + 0.1 + 0.2)
    assert stats.cost_is_complete is True
    assert stats.per_model["cheap"] == Usage(1_000_000, 1_000_000)


def test_unavailable_pricing_permanently_hides_incomplete_cost_across_model_switches() -> None:
    stats = _panel.SessionStats()
    stats.record("main", Usage(1_000_000, 1_000_000), _M3)
    unpriced_models = (
        ModelSpec(id="local/one", input_cost=99.0, output_cost=99.0, pricing_available=False),
        ModelSpec(id="local/two", pricing_available=False),
    )

    for model in unpriced_models:
        stats.record("main", Usage(10, 20), model)
        line = _panel.status_line(model, stats)

        assert model.id in line
        assert "$" not in line
        assert stats.cost_is_complete is False

    assert (stats.input_tokens, stats.output_tokens) == (1_000_020, 1_000_040)
    assert stats.cost == pytest.approx(1.5)
    priced_line = _panel.status_line(_M3, stats)
    assert "$" not in priced_line


def test_known_free_model_still_displays_zero_cost() -> None:
    known_free = ModelSpec(id="known-free")
    stats = _panel.SessionStats()
    stats.record("main", Usage(1_000, 2_000), known_free)
    line = _panel.status_line(known_free, stats)

    assert stats.cost_is_complete is True
    assert "$0" in line


def test_usage_without_a_model_still_counts_tokens() -> None:
    stats = _panel.SessionStats()

    stats.record("main", Usage(10, 20), None)

    assert (stats.input_tokens, stats.output_tokens, stats.cost) == (10, 20, 0.0)
    assert stats.cost_is_complete is False


def test_counts_stay_short_enough_for_one_line() -> None:
    assert _panel.compact(0) == "0"
    assert _panel.compact(999) == "999"
    assert _panel.compact(1_000) == "1k"
    assert _panel.compact(8_587) == "8.6k"
    assert _panel.compact(1_000_000) == "1M"
    assert _panel.compact(10_123_456) == "10.12M"


def test_a_cost_too_small_to_show_is_still_shown() -> None:
    assert _panel.format_cost(0) == "$0"
    assert _panel.format_cost(0.0026) == "$0.0026"
    assert _panel.format_cost(4.0067) == "$4.007"
    assert _panel.format_cost(123.456) == "$123.46"


def test_the_toolbar_is_not_reverse_video() -> None:
    # Its default is, which paints a solid white band across the terminal.
    from prompt_toolkit.styles import default_ui_style, merge_styles

    session = _panel.make_session(lambda: "x")
    merged = merge_styles([default_ui_style(), session.style])

    assert default_ui_style().get_attrs_for_style_str("class:bottom-toolbar").reverse is True
    for cls in ("class:bottom-toolbar", "class:bottom-toolbar.text"):
        attrs = merged.get_attrs_for_style_str(cls)
        assert attrs.reverse is False, cls
        assert attrs.bgcolor == "default", cls


def test_prompt_is_named_and_visually_distinct_from_agent_text() -> None:
    from prompt_toolkit.formatted_text import to_formatted_text
    from prompt_toolkit.styles import default_ui_style, merge_styles

    session = _panel.make_session(lambda: "x")
    merged = merge_styles([default_ui_style(), session.style])

    assert to_formatted_text(_panel.PROMPT_MESSAGE) == [("class:repl-prompt", "axio-repl> ")]
    prompt_attrs = merged.get_attrs_for_style_str("class:repl-prompt")
    default_attrs = merged.get_attrs_for_style_str("")
    assert prompt_attrs.color == "ansiwhite"
    assert prompt_attrs.bold is True
    assert prompt_attrs != default_attrs


def test_powerline_prompt_uses_a_coloured_segment_and_separator() -> None:
    from prompt_toolkit.formatted_text import to_formatted_text

    assert to_formatted_text(_panel.prompt_message(powerline=True)) == [
        ("bold fg:ansiwhite bg:ansicyan", " axio-repl "),
        ("fg:ansicyan bg:default", "\ue0b0"),
        ("", " "),
    ]


def test_prompt_factory_uses_only_the_stable_effective_username() -> None:
    from prompt_toolkit.formatted_text import to_formatted_text

    plain = _panel.make_prompt_factory("alice")

    assert to_formatted_text(plain()) == [("class:repl-prompt", "alice> ")]
    assert to_formatted_text(plain()) == [("class:repl-prompt", "alice> ")]

    powerline = _panel.make_prompt_factory(
        "alice",
        powerline=True,
    )
    assert to_formatted_text(powerline()) == [
        ("bold fg:ansiwhite bg:ansicyan", " alice "),
        ("fg:ansicyan bg:default", "\ue0b0"),
        ("", " "),
    ]

    monochrome = _panel.make_prompt_factory(
        "alice",
        powerline=True,
        theme=MONOCHROME_THEME,
    )
    assert to_formatted_text(monochrome()) == [
        ("bold fg:ansiblack bg:ansiwhite", " alice "),
        ("fg:ansiwhite bg:default", "\ue0b0"),
        ("", " "),
    ]


def test_submitted_message_uses_accept_time_without_changing_message_text() -> None:
    submitted_at = datetime(2026, 8, 20, 12, 41, tzinfo=UTC)

    assert _panel.submitted_message("first\nsecond", "alice", submitted_at) == (
        "\x1b[1;97m12:41 alice>\x1b[0m first\nsecond"
    )
    assert _panel.submitted_message("ping", "alice", submitted_at, powerline=True) == (
        "\x1b[1;97;46m 12:41 alice \x1b[22;36;49m\ue0b0\x1b[0m ping"
    )
    assert _panel.submitted_message("ping", "alice", submitted_at, theme=NO_COLOR_THEME) == "12:41 alice> ping"


def test_status_line_reports_action_visibility_and_backlog() -> None:
    from axio.models import ModelSpec

    status = _panel.status_line(ModelSpec(id="test"), _panel.SessionStats(), "actions: on (3 queued)")

    assert "actions: on (3 queued)" in status


def test_status_line_reports_agent_phase_and_temporary_panel_feedback() -> None:
    from axio.models import ModelSpec

    status = _panel.status_line(
        ModelSpec(id="test"),
        _panel.SessionStats(),
        "actions: off",
        agent_status="main: reasoning",
        panel_message="Commands: /help, /quit\nTool results remain in the dialog",
    )

    assert "main: reasoning │ 0 in / 0 out" in status
    assert status.endswith("Commands: /help, /quit\nTool results remain in the dialog")


def test_panel_feedback_is_bounded() -> None:
    message = "\n".join(f"line {index}" for index in range(_panel.MAX_PANEL_MESSAGE_LINES + 3))

    bounded = _panel.bounded_panel_message(message)

    assert bounded.splitlines()[:-1] == [f"line {index}" for index in range(_panel.MAX_PANEL_MESSAGE_LINES)]
    assert bounded.splitlines()[-1] == "… 3 more line(s)"


async def test_ctrl_c_at_the_prompt_interrupts_instead_of_ending_the_read() -> None:
    # The prompt is up for the whole session now, and it puts the terminal in
    # raw mode - so Ctrl+C arrives as a keypress, not as SIGINT, and the handler
    # that used to stop a running turn never fires.
    from prompt_toolkit.formatted_text import to_formatted_text

    from axio_repl import ReplRenderer, _read_input_async
    from axio_repl._input import InputSubmitted, SubmissionDisposition

    interrupts: list[int] = []
    answers: list[object] = [KeyboardInterrupt(), KeyboardInterrupt(), "carry on"]
    prompt_factory = _panel.make_prompt_factory("alice", powerline=True)
    observed_prompts: list[object] = []

    class _Session:
        async def prompt_async(self, prompt: object) -> str:
            observed_prompts.append(prompt)
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return str(answer)

    renderer = ReplRenderer()
    renderer.set_focus("child")

    async def admit(text: str, target_agent_id: str, reserved_seq: int | None) -> InputSubmitted:
        del reserved_seq
        return InputSubmitted(
            text=text,
            target_agent_id=target_agent_id,
            disposition=SubmissionDisposition.PENDING,
            input_id="input-1",
            arrival_seq=1,
        )

    result = await _read_input_async(
        _Session(),
        renderer,
        lambda: interrupts.append(1),
        admit,
        prompt_factory=prompt_factory,
    )

    assert result.text == "carry on"
    assert result.target_agent_id == "child"
    assert len(interrupts) == 2
    assert [to_formatted_text(cast(Any, prompt))[0][1] for prompt in observed_prompts] == [
        " alice ",
        " alice ",
        " alice ",
    ]


async def test_initial_editor_text_survives_prompt_retry_but_not_a_completed_empty_attempt() -> None:
    from axio_repl import ReplRenderer, _read_input_async
    from axio_repl._input import InputSubmitted, SubmissionDisposition

    answers: list[object] = [KeyboardInterrupt(), "", "submit"]
    observed_defaults: list[object] = []

    class Session:
        async def prompt_async(self, prompt: object, **kwargs: object) -> str:
            del prompt
            observed_defaults.append(kwargs.get("default"))
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return str(answer)

    async def admit(text: str, target_agent_id: str, reserved_seq: int | None) -> InputSubmitted:
        del reserved_seq
        return InputSubmitted(
            text=text,
            target_agent_id=target_agent_id,
            disposition=SubmissionDisposition.PENDING,
            input_id="input-1",
            arrival_seq=1,
        )

    result = await _read_input_async(Session(), ReplRenderer(), lambda: None, admit, "restored draft")

    assert result.text == "submit"
    assert observed_defaults == ["restored draft", "restored draft", None]


async def test_enter_time_is_captured_before_delayed_admission_and_rendered_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from axio_repl import ReplRenderer, _read_input_async
    from axio_repl._input import InputSubmitted, SubmissionDisposition

    submitted_at = datetime(2026, 8, 20, 12, 41, tzinfo=UTC)
    provider_calls: list[datetime] = []
    admission_started = asyncio.Event()
    release_admission = asyncio.Event()

    def capture_time() -> datetime:
        provider_calls.append(submitted_at)
        return submitted_at

    async def admit(text: str, target_agent_id: str, reserved_seq: int | None) -> InputSubmitted:
        assert text == "queued message"
        assert target_agent_id == "main"
        assert reserved_seq == 1
        admission_started.set()
        await release_admission.wait()
        return InputSubmitted(
            text=text,
            target_agent_id=target_agent_id,
            disposition=SubmissionDisposition.PENDING,
            input_id="input-1",
            arrival_seq=reserved_seq,
        )

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(
                lambda: "status",
                reserve_sequence=lambda: 1,
                accepted_at_provider=capture_time,
            )
            reader = asyncio.create_task(
                _read_input_async(
                    session,
                    ReplRenderer(effective_username="alice"),
                    lambda: None,
                    admit,
                    prompt_factory=_panel.make_prompt_factory("alice"),
                )
            )
            pipe.send_text("queued message\r")
            await asyncio.wait_for(admission_started.wait(), timeout=1)

            assert provider_calls == [submitted_at]
            assert _panel.accepted_at(session) == submitted_at
            assert not reader.done()

            release_admission.set()
            submitted = await asyncio.wait_for(reader, timeout=1)

    assert submitted.text == "queued message"
    output = capsys.readouterr().out
    assert output.count("12:41 alice>") == 1
    assert output.endswith("12:41 alice>\x1b[0m queued message\n")
    assert _panel.accepted_at(session) is None


async def test_admission_failure_clears_accept_metadata_without_retrying() -> None:
    from axio_repl import ReplRenderer, _read_input_async
    from axio_repl._input import InputSubmitted

    submitted_at = datetime(2026, 8, 20, 12, 41, tzinfo=UTC)
    calls = 0

    class Buffer:
        def __init__(self) -> None:
            self.text = "accepted text"
            self._axio_accepted_target = "main"
            self._axio_accepted_seq = 1
            self._axio_accepted_at = submitted_at

        def reset(self) -> None:
            self.text = ""

    class Session:
        default_buffer = Buffer()

        async def prompt_async(self, prompt: object) -> str:
            del prompt
            return self.default_buffer.text

    async def admit(text: str, target_agent_id: str, reserved_seq: int | None) -> InputSubmitted:
        nonlocal calls
        del text, target_agent_id, reserved_seq
        calls += 1
        raise RuntimeError("admission failed")

    session = Session()
    with pytest.raises(RuntimeError, match="admission failed"):
        await _read_input_async(session, ReplRenderer(), lambda: None, admit)

    assert calls == 1
    assert session.default_buffer.text == "accepted text"
    assert _panel.accepted_sequence(session) is None
    assert _panel.accepted_at(session) is None


async def test_enter_admission_completes_before_editor_clear_even_when_reader_is_cancelled() -> None:
    from axio_tools_agents.runtime import AgentEventEnvelope, ExecutionMode, RuntimeEvent

    from axio_repl import ReplRenderer, _read_input_async
    from axio_repl._coordinator import PendingInputCoordinator
    from axio_repl._input import InputSubmitted, SubmissionDisposition

    published = asyncio.Event()
    release = asyncio.Event()

    async def publish(event: RuntimeEvent) -> AgentEventEnvelope:
        published.set()
        await release.wait()
        return AgentEventEnvelope(
            seq=1,
            session_id="session",
            run_id="run",
            agent_id="main",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.FOREGROUND,
            parent_tool_use_id=None,
            event=event,
        )

    coordinator = PendingInputCoordinator(publish)

    class Buffer:
        text = "queued text"
        _axio_accepted_target = "child-at-enter"

        def reset(self) -> None:
            self.text = ""

    class Session:
        default_buffer = Buffer()

        async def prompt_async(self, prompt: object) -> str:
            assert prompt == _panel.PROMPT_MESSAGE
            return self.default_buffer.text

    async def admit(text: str, target_agent_id: str, reserved_seq: int | None) -> InputSubmitted:
        entry = await coordinator.admit(text, target_agent_id, reserved_seq=reserved_seq)
        return InputSubmitted(
            text=text,
            target_agent_id=target_agent_id,
            disposition=SubmissionDisposition.PENDING,
            input_id=entry.id,
            arrival_seq=entry.arrival_seq,
        )

    session = Session()
    renderer = ReplRenderer()
    renderer.set_focus("main")
    reader = asyncio.create_task(_read_input_async(session, renderer, lambda: None, admit))
    await asyncio.wait_for(published.wait(), timeout=1)

    assert session.default_buffer.text == "queued text"
    reader.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await reader

    assert session.default_buffer.text == ""
    assert coordinator.pending_count == 1
    assert coordinator.state.pending[0].intended_target_agent_id == "child-at-enter"


async def test_reader_cancellation_after_accept_cannot_abandon_reserved_sequence() -> None:
    from axio_tools_agents.runtime import (
        AgentEventEnvelope,
        ExecutionMode,
        InputReceived,
        RuntimeEvent,
        SessionEventHub,
    )

    from axio_repl import ReplRenderer, _read_input_async
    from axio_repl._coordinator import PendingInputCoordinator
    from axio_repl._input import InputSubmitted, SubmissionDisposition

    hub = SessionEventHub(session_id="session")
    accepted = asyncio.Event()

    async def publish_main(event: RuntimeEvent, reserved_seq: int | None = None) -> AgentEventEnvelope:
        return await hub.publish(
            event,
            run_id="run",
            agent_id="main",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.FOREGROUND,
            reserved_seq=reserved_seq,
        )

    coordinator = PendingInputCoordinator(
        publish_main,
        lambda event, sequence: publish_main(event, sequence),
    )

    class Buffer:
        text = "accepted draft"
        _axio_accepted_target = ""
        _axio_accepted_seq = 0

        def reset(self) -> None:
            self.text = ""

    class Session:
        default_buffer = Buffer()

        async def prompt_async(self, prompt: object) -> str:
            assert prompt == _panel.PROMPT_MESSAGE
            self.default_buffer._axio_accepted_target = "main"
            self.default_buffer._axio_accepted_seq = hub.reserve_sequence()
            accepted.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

    async def admit(text: str, target_agent_id: str, reserved_seq: int | None) -> InputSubmitted:
        entry = await coordinator.admit(text, target_agent_id, reserved_seq=reserved_seq)
        return InputSubmitted(
            text=text,
            target_agent_id=target_agent_id,
            disposition=SubmissionDisposition.PENDING,
            input_id=entry.id,
            arrival_seq=entry.arrival_seq,
        )

    reader = asyncio.create_task(_read_input_async(Session(), ReplRenderer(), lambda: None, admit))
    await asyncio.wait_for(accepted.wait(), timeout=1)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader

    peer = await asyncio.wait_for(
        hub.publish(
            InputReceived(text="later", source="peer"),
            run_id="peer",
            agent_id="peer",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.BACKGROUND,
        ),
        timeout=1,
    )
    assert coordinator.state.pending[0].arrival_seq == 1
    assert peer.seq == 2


async def test_prompt_failure_after_accept_completes_reservation_before_reraising() -> None:
    from axio_tools_agents.runtime import (
        AgentEventEnvelope,
        ExecutionMode,
        InputReceived,
        RuntimeEvent,
        SessionEventHub,
    )

    from axio_repl import ReplRenderer, _read_input_async
    from axio_repl._coordinator import PendingInputCoordinator
    from axio_repl._input import InputSubmitted, SubmissionDisposition

    hub = SessionEventHub(session_id="session")

    async def publish_main(event: RuntimeEvent, reserved_seq: int | None = None) -> AgentEventEnvelope:
        return await hub.publish(
            event,
            run_id="run",
            agent_id="main",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.FOREGROUND,
            reserved_seq=reserved_seq,
        )

    coordinator = PendingInputCoordinator(
        publish_main,
        lambda event, sequence: publish_main(event, sequence),
    )

    class Buffer:
        text = "accepted before failure"
        _axio_accepted_target = "main"
        _axio_accepted_seq = hub.reserve_sequence()

        def reset(self) -> None:
            self.text = ""

    class Session:
        default_buffer = Buffer()

        async def prompt_async(self, prompt: object) -> str:
            assert prompt == _panel.PROMPT_MESSAGE
            raise OSError("prompt teardown failed")

    async def admit(text: str, target_agent_id: str, reserved_seq: int | None) -> InputSubmitted:
        entry = await coordinator.admit(text, target_agent_id, reserved_seq=reserved_seq)
        return InputSubmitted(
            text=text,
            target_agent_id=target_agent_id,
            disposition=SubmissionDisposition.PENDING,
            input_id=entry.id,
            arrival_seq=entry.arrival_seq,
        )

    with pytest.raises(OSError, match="prompt teardown failed"):
        await _read_input_async(Session(), ReplRenderer(), lambda: None, admit)

    peer = await asyncio.wait_for(
        hub.publish(
            InputReceived(text="later", source="peer"),
            run_id="peer",
            agent_id="peer",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.BACKGROUND,
        ),
        timeout=1,
    )
    assert coordinator.state.pending[0].arrival_seq == 1
    assert peer.seq == 2


async def test_escape_interrupts_without_submitting_or_changing_editor(
    tmp_path: Path,
    repl_history_path: Path,
) -> None:
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    interrupts: list[int] = []
    interrupted = asyncio.Event()

    def interrupt() -> None:
        interrupts.append(1)
        interrupted.set()

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status", on_interrupt=interrupt)

            pipe.send_text("sent with enter\r")
            assert await session.prompt_async(_panel.PROMPT_MESSAGE) == "sent with enter"
            _panel.commit_history(session, ("sent with enter",))
            assert interrupts == []

            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            pipe.send_text("unfinished editor\x1b")
            await asyncio.wait_for(interrupted.wait(), timeout=1)
            assert prompt.done() is False
            assert len(interrupts) == 1

            interrupted.clear()
            pipe.send_text("\x1b")
            await asyncio.wait_for(interrupted.wait(), timeout=1)
            assert prompt.done() is False
            assert len(interrupts) == 2

            pipe.send_text(" completed\r")
            assert await prompt == "unfinished editor completed"
            _panel.commit_history(session, ("unfinished editor completed",))

    assert repl_history_path == tmp_path / "history"
    history = repl_history_path.read_text()
    assert "sent with enter" in history
    assert "unfinished editor completed" in history


async def test_repeated_escape_stress_never_submits_or_mutates_editor() -> None:
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    interrupt_count = 0

    def interrupt() -> None:
        nonlocal interrupt_count
        interrupt_count += 1

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status", on_interrupt=interrupt)
            session.app.ttimeoutlen = 0.001
            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            await asyncio.sleep(0.05)
            pipe.send_text("editor stays")

            pipe.send_text("\x1b" * 500)
            for _ in range(1_000):
                if interrupt_count == 500:
                    break
                await asyncio.sleep(0.001)
            assert interrupt_count == 500
            assert prompt.done() is False

            pipe.send_text(" intact\r")
            assert await prompt == "editor stays intact"


async def test_up_recalls_all_pending_messages_as_one_editor_value(repl_history_path: Path) -> None:
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    pending = ["first pending", "second pending"]
    recalled = asyncio.Event()

    async def recall_pending() -> str | None:
        recalled.set()
        if not pending:
            return None
        result = "\n\n".join(pending)
        pending.clear()
        return result

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status", recall_pending=recall_pending)

            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            pipe.send_text("\x1b[A")
            await asyncio.wait_for(recalled.wait(), timeout=1)
            await asyncio.sleep(0)
            pipe.send_text(" and more\r")
            assert await prompt == "first pending\n\nsecond pending and more"
            _panel.commit_history(session, ("first pending\n\nsecond pending and more",))

    history = repl_history_path.read_text()
    assert "first pending" in history
    assert "second pending and more" in history


async def test_up_falls_back_to_prompt_history_when_pending_is_empty(repl_history_path: Path) -> None:
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    async def no_pending() -> str | None:
        return None

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status", recall_pending=no_pending)

            pipe.send_text("history item\r")
            assert await session.prompt_async(_panel.PROMPT_MESSAGE) == "history item"
            _panel.commit_history(session, ("history item",))

            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            pipe.send_text("\x1b[A")
            await asyncio.sleep(0.05)
            pipe.send_text(" edited\r")
            assert await prompt == "history item edited"
            _panel.commit_history(session, ("history item edited",))

    assert "history item edited" in repl_history_path.read_text()


async def test_unclaimed_enter_is_not_exposed_by_persistent_history() -> None:
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status")

            pipe.send_text("still pending\r")
            assert await session.prompt_async(_panel.PROMPT_MESSAGE) == "still pending"

            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            pipe.send_text("\x1b[Anew text\r")
            assert await prompt == "new text"


async def test_enter_captures_focus_before_prompt_teardown() -> None:
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    focused = {"agent_id": "child-at-enter"}

    def capture_target() -> str:
        accepted = focused["agent_id"]
        focused["agent_id"] = "main-after-accept"
        return accepted

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status", capture_target=capture_target)
            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            pipe.send_text("targeted input\r")

            assert await prompt == "targeted input"
            assert focused["agent_id"] == "main-after-accept"
            assert _panel.accepted_target(session, focused["agent_id"]) == "child-at-enter"


async def test_enter_reservation_precedes_peer_published_after_accept_handler() -> None:
    from typing import Any

    from axio_tools_agents.runtime import (
        AgentEventEnvelope,
        ExecutionMode,
        InputBuffered,
        InputReceived,
        RuntimeEvent,
        SessionEventHub,
    )
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from axio_repl import ReplRenderer, _read_input_async
    from axio_repl._coordinator import PendingInputCoordinator
    from axio_repl._input import InputSubmitted, SubmissionDisposition

    hub = SessionEventHub(session_id="session")
    observed: list[AgentEventEnvelope] = []
    peer_tasks: list[asyncio.Task[AgentEventEnvelope]] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        observed.append(envelope)

    hub.subscribe(collect)

    async def publish_main(event: RuntimeEvent, reserved_seq: int | None = None) -> AgentEventEnvelope:
        return await hub.publish(
            event,
            run_id="main-run",
            agent_id="main",
            parent_agent_id=None,
            turn_id=None,
            execution_mode=ExecutionMode.FOREGROUND,
            reserved_seq=reserved_seq,
        )

    coordinator = PendingInputCoordinator(
        publish_main,
        lambda event, sequence: publish_main(event, sequence),
    )

    def schedule_peer_after_accept() -> str:
        def publish_peer() -> None:
            peer_tasks.append(
                asyncio.create_task(
                    hub.publish(
                        InputReceived(text="peer", source="peer"),
                        run_id="peer-run",
                        agent_id="peer",
                        parent_agent_id=None,
                        turn_id=None,
                        execution_mode=ExecutionMode.BACKGROUND,
                    )
                )
            )

        asyncio.get_running_loop().call_soon(publish_peer)
        return "main"

    async def admit(text: str, target_agent_id: str, reserved_seq: int | None) -> InputSubmitted:
        entry = await coordinator.admit(text, target_agent_id, reserved_seq=reserved_seq)
        return InputSubmitted(
            text=text,
            target_agent_id=target_agent_id,
            disposition=SubmissionDisposition.PENDING,
            input_id=entry.id,
            arrival_seq=entry.arrival_seq,
        )

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(
                lambda: "status",
                capture_target=schedule_peer_after_accept,
                reserve_sequence=hub.reserve_sequence,
            )
            reader = asyncio.create_task(_read_input_async(session, ReplRenderer(), lambda: None, admit))
            pipe.send_text("queued before peer\r")
            submitted = await asyncio.wait_for(reader, timeout=1)
            await asyncio.gather(*peer_tasks)

    assert submitted.arrival_seq == 1
    assert isinstance(observed[0].event, InputBuffered)
    assert isinstance(observed[1].event, InputReceived)
    assert [envelope.seq for envelope in observed] == [1, 2]


async def test_empty_ctrl_d_requires_two_presses_but_nonempty_ctrl_d_deletes_forward() -> None:
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    presses: list[float] = []

    def arm_exit(now: float) -> bool:
        presses.append(now)
        return len(presses) == 2

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status", on_empty_eof=arm_exit)

            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            pipe.send_text("ab\x1b[D\x04\r")
            assert await prompt == "a"
            assert presses == []

            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            pipe.send_text("\x04")
            await asyncio.sleep(0.05)
            assert prompt.done() is False
            assert len(presses) == 1

            pipe.send_text("\x04")
            with pytest.raises(EOFError):
                await prompt
            assert len(presses) == 2


def test_editor_text_reads_without_mutating_default_buffer() -> None:
    class Buffer:
        text = "unsent editor"

    class Session:
        default_buffer = Buffer()

    session = Session()

    assert _panel.editor_text(session) == "unsent editor"
    assert session.default_buffer.text == "unsent editor"


async def test_cancelling_prompt_keeps_unsent_editor_available_for_shutdown_snapshot() -> None:
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status")
            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            pipe.send_text("unsent editor")
            for _ in range(100):
                if _panel.editor_text(session) == "unsent editor":
                    break
                await asyncio.sleep(0.01)
            assert _panel.editor_text(session) == "unsent editor"

            prompt.cancel()
            await asyncio.gather(prompt, return_exceptions=True)

            assert _panel.editor_text(session) == "unsent editor"


async def test_ctrl_c_requests_shutdown_without_exiting_or_clearing_editor() -> None:
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    shutdown = asyncio.Event()
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status", on_shutdown=shutdown.set)
            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            pipe.send_text("unsaved draft\x03")
            await asyncio.wait_for(shutdown.wait(), timeout=1)

            assert not prompt.done()
            assert _panel.editor_text(session) == "unsaved draft"

            prompt.cancel()
            await asyncio.gather(prompt, return_exceptions=True)
            assert _panel.editor_text(session) == "unsaved draft"


async def test_prompt_cancellation_cannot_split_recall_from_editor_update() -> None:
    from typing import Any

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    recall_started = asyncio.Event()
    release_recall = asyncio.Event()

    async def recall_pending() -> str:
        recall_started.set()
        await release_recall.wait()
        return "first\n\nsecond"

    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            session: Any = _panel.make_session(lambda: "status", recall_pending=recall_pending)
            prompt = asyncio.create_task(session.prompt_async(_panel.PROMPT_MESSAGE))
            pipe.send_text("\x1b[A")
            await asyncio.wait_for(recall_started.wait(), timeout=1)

            prompt.cancel()
            await asyncio.sleep(0)
            release_recall.set()
            await asyncio.gather(prompt, return_exceptions=True)

            assert _panel.editor_text(session) == "first\n\nsecond"
