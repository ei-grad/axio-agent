"""A status line pinned to the bottom of an otherwise ordinary terminal.

Deliberately not a full-screen UI. Output stays on the primary buffer, so it
scrolls into the terminal's own history. prompt_toolkit owns only the lines it
draws; the REPL terminal sink temporarily removes and redraws them around
serialized background output.

Everything the line reports is a pure function of state passed in, so what it
says can be tested without a terminal.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from axio import background
from axio.models import ModelSpec
from axio.types import CostSource, Usage
from axio_tools_agents.peers import background_agent_state, local_background_agent_records
from prompt_toolkit.formatted_text import FormattedText

from axio_repl import _replay
from axio_repl._multiplexer import sanitize_identity_component, sanitize_terminal_text
from axio_repl._powerline import prompt_badge, submitted_prompt_badge
from axio_repl._theme import DEFAULT_THEME, TerminalTheme

HISTORY_PATH = Path.home() / ".axio_repl_history"

SEPARATOR = " │ "
MAX_PANEL_MESSAGE_LINES = 8
MAX_PANEL_MESSAGE_CHARS = 4096
MAIN_AGENT = "main"


def prompt_message(
    label: str = "axio-repl",
    *,
    powerline: bool = False,
    theme: TerminalTheme = DEFAULT_THEME,
) -> FormattedText:
    """Build the input prompt in the selected presentation style."""

    safe_label = sanitize_identity_component(label) or "unknown"
    if powerline:
        return prompt_badge(safe_label, theme)
    return FormattedText([("class:repl-prompt", f"{safe_label}> ")])


def make_prompt_factory(
    effective_username: str,
    *,
    powerline: bool = False,
    theme: TerminalTheme = DEFAULT_THEME,
) -> Callable[[], FormattedText]:
    """Capture stable identity for every active editor prompt."""

    username = sanitize_identity_component(effective_username) or "unknown"

    def build() -> FormattedText:
        return prompt_message(username, powerline=powerline, theme=theme)

    return build


def submitted_message(
    text: str,
    effective_username: str,
    submitted_at: datetime,
    *,
    powerline: bool = False,
    theme: TerminalTheme = DEFAULT_THEME,
) -> str:
    """Format one accepted user message for persistent terminal scrollback."""

    username = sanitize_identity_component(effective_username) or "unknown"
    safe_text = sanitize_terminal_text(text)
    label = f"{submitted_at:%H:%M} {username}"
    if powerline:
        return f"{submitted_prompt_badge(label, theme)} {safe_text}"
    return f"{theme.prompt.ansi}{label}>{theme.reset} {safe_text}"


PROMPT_MESSAGE = prompt_message()


@dataclass
class SessionStats:
    """What the session has spent, and how full its context is.

    Usage is recorded per iteration rather than per turn so the line moves while
    an answer is still being written; a turn's total is the sum of its
    iterations, so counting both would double everything.

    Provider-reported cost is authoritative and complete without ModelSpec
    pricing. Otherwise the tokens are estimated at the model in use, which is
    exact for model attribution only when a spawned agent shares the parent's
    model. The status distinguishes reported, estimated, and mixed totals. A
    total becomes incomplete only when an operation has neither reported cost
    nor an available model-price estimate; missing model pricing alone does not
    invalidate an operation that already has reported cost.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    cost_source: CostSource | None = None
    cost_is_complete: bool = True
    context_tokens: int = 0
    per_model: dict[str, Usage] = field(default_factory=dict)

    def record(self, agent_id: str, usage: Usage, model: ModelSpec | None) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        if agent_id == MAIN_AGENT:
            # The prompt of the latest iteration is what occupies the window
            # now: it grows with the conversation and drops when it is compacted.
            self.context_tokens = usage.input_tokens
        if model is not None:
            self.per_model[model.id] = self.per_model.get(model.id, Usage(0, 0)) + usage
        if usage.cost_usd is not None:
            assert usage.cost_source is not None
            self.cost += usage.cost_usd
            self.cost_source = _combine_cost_sources(self.cost_source, usage.cost_source)
            return
        if model is None or not model.pricing_available:
            self.cost_is_complete = False
            return
        self.cost += (usage.input_tokens * model.input_cost + usage.output_tokens * model.output_cost) / 1_000_000
        self.cost_source = _combine_cost_sources(self.cost_source, CostSource.estimated)


def _combine_cost_sources(current: CostSource | None, added: CostSource) -> CostSource:
    if current is None:
        return added
    if current == added and current is not CostSource.mixed:
        return current
    return CostSource.mixed


def compact(value: int) -> str:
    """A token count short enough to sit in a status line."""
    if value < 1_000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1_000:.1f}k".replace(".0k", "k")
    return f"{value / 1_000_000:.2f}M".replace(".00M", "M")


def format_cost(cost: float) -> str:
    if cost == 0:
        return "$0"
    if cost < 0.01:
        return f"${cost:.4f}"
    if cost < 10:
        return f"${cost:.3f}"
    return f"${cost:.2f}"


def format_cost_with_source(cost: float, source: CostSource) -> str:
    prefix = {
        CostSource.provider: "reported",
        CostSource.estimated: "est.",
        CostSource.mixed: "mixed",
    }[source]
    return f"{prefix} {format_cost(cost)}"


def agent_summary(now: float | None = None) -> str:
    """Background agents and detached tool calls, when there are any.

    Empty when there is nothing running — the line should not cost space until
    it has something to say.
    """
    parts: list[str] = []

    states: dict[str, int] = {}
    failed: list[str] = []
    for record in local_background_agent_records():
        state, error = background_agent_state(record.id)
        states[state] = states.get(state, 0) + 1
        if error:
            failed.append(record.name)
    if states:
        counts = ", ".join(f"{count} {state}" for state, count in sorted(states.items()))
        parts.append(f"agents: {counts}")
    if failed:
        parts.append(f"failed: {', '.join(sorted(set(failed))[:3])}")

    calls = background.snapshot()
    running = [c for c in calls if c.state == "running"]
    done = [c for c in calls if c.state != "running" and not c.collected]
    if running:
        parts.append(f"tasks: {len(running)} running")
    if done:
        parts.append(f"{len(done)} ready to collect")

    return SEPARATOR.join(parts)


def bounded_panel_message(text: str) -> str:
    """Bound temporary UI feedback without moving normal output into scrollback."""

    normalized = text.strip()
    if not normalized:
        return ""
    lines = normalized.splitlines()
    omitted_lines = max(0, len(lines) - MAX_PANEL_MESSAGE_LINES)
    retained = lines[:MAX_PANEL_MESSAGE_LINES]
    value = "\n".join(retained)
    if len(value) > MAX_PANEL_MESSAGE_CHARS:
        value = value[:MAX_PANEL_MESSAGE_CHARS].rstrip()
        omitted_lines = max(1, omitted_lines)
    if omitted_lines:
        value += f"\n… {omitted_lines} more line(s)"
    return value


def status_line(
    model: ModelSpec | None,
    stats: SessionStats,
    action_status: str = "",
    *,
    agent_status: str = "",
    panel_message: str = "",
) -> str:
    """The whole line: model, context, usage, applicable cost, and activity."""
    parts: list[str] = []
    if model is not None:
        parts.append(model.id)
        if model.context_window:
            parts.append(f"ctx {compact(stats.context_tokens)}/{compact(model.context_window)}")
    if agent_status:
        parts.append(agent_status)
    parts.append(f"{compact(stats.input_tokens)} in / {compact(stats.output_tokens)} out")
    if stats.cost_is_complete:
        if stats.cost_source is not None:
            parts.append(format_cost_with_source(stats.cost, stats.cost_source))
        elif model is not None and model.pricing_available:
            parts.append(format_cost_with_source(stats.cost, CostSource.estimated))
    agents = agent_summary()
    if agents:
        parts.append(agents)
    if action_status:
        parts.append(action_status)
    status = SEPARATOR.join(parts)
    message = bounded_panel_message(panel_message)
    return f"{status}\n{message}" if message else status


ESCAPE_FLUSH_SECONDS = 0.2
"""How long a lone Escape byte waits to be told apart from a sequence.

An arrow key arrives as three bytes beginning with the same one, so the parser
holds an Escape until either the rest arrives or this elapses. Two hundred
milliseconds keeps a lone interrupt responsive while still allowing an arrow
sequence split across nearby reads to complete. Over a slow link the bytes can
arrive farther apart, so this remains an explicit compatibility tradeoff.
"""


def input_bindings(
    on_interrupt: Callable[[], None],
    on_shutdown: Callable[[], None],
    recall_pending: Callable[[], Awaitable[str | None]],
    on_empty_eof: Callable[[float], bool],
) -> Any:
    """Bind interruption and pending recall without submitting the editor.

    Enter remains prompt_toolkit's only submit operation. Escape records an
    interruption while leaving the current editor untouched. Up first asks the
    coordinator for every still-pending user message; only when there is no
    such input does it fall back to prompt history or multiline navigation.

    Bound eagerly, because forty of the default bindings begin with Escape - all
    the Alt+key ones - and a lone press is ambiguous against every one of them,
    so the processor waits a further second before deciding. That is the cost:
    Alt+b, Alt+f and their relatives no longer reach the buffer. Word motions
    are worth less here than a key that answers when pressed.
    """
    from prompt_toolkit.document import Document
    from prompt_toolkit.key_binding import KeyBindings

    bindings = KeyBindings()
    recall_in_flight = False

    @bindings.add("escape", eager=True)
    def _interrupt(event: Any) -> None:
        on_interrupt()
        event.app.invalidate()

    @bindings.add("c-c", eager=True)
    def _shutdown(event: Any) -> None:
        on_shutdown()
        event.app.invalidate()

    @bindings.add("up")
    def _recall_or_history(event: Any) -> None:
        nonlocal recall_in_flight
        if recall_in_flight:
            return
        recall_in_flight = True

        async def apply_recall() -> None:
            nonlocal recall_in_flight
            cancellation: asyncio.CancelledError | None = None
            try:
                recall_task: asyncio.Future[str | None] = asyncio.ensure_future(recall_pending())
                while True:
                    try:
                        text = await asyncio.shield(recall_task)
                        break
                    except asyncio.CancelledError as exc:
                        if recall_task.done():
                            text = recall_task.result()
                            cancellation = cancellation or exc
                            break
                        cancellation = cancellation or exc
                buffer = event.current_buffer
                if text is None:
                    buffer.auto_up(count=event.arg)
                else:
                    buffer.set_document(Document(text, cursor_position=len(text)))
                event.app.invalidate()
                if cancellation is not None:
                    raise cancellation
            finally:
                recall_in_flight = False

        event.app.create_background_task(apply_recall())

    @bindings.add("c-d", eager=True)
    def _forward_delete_or_exit(event: Any) -> None:
        buffer = event.current_buffer
        if buffer.text:
            buffer.delete()
        elif on_empty_eof(time.monotonic()):
            event.app.exit(exception=EOFError())

    return bindings


def make_session(
    status: Any = None,
    on_interrupt: Callable[[], None] | None = None,
    on_shutdown: Callable[[], None] | None = None,
    recall_pending: Callable[[], Awaitable[str | None]] | None = None,
    on_empty_eof: Callable[[float], bool] | None = None,
    capture_target: Callable[[], str] | None = None,
    reserve_sequence: Callable[[], int] | None = None,
    accepted_at_provider: Callable[[], datetime] | None = None,
    replay: _replay.ReplayLog | None = None,
    *,
    theme: TerminalTheme = DEFAULT_THEME,
) -> Any:
    """A prompt session with history, a status line, and explicit controls.

    The default toolbar style is reverse video, which paints a solid white band
    across the bottom of the terminal. Dim text on the terminal's own background
    keeps the line readable without turning it into a wall.

    Enter submits the editor. Escape never submits or modifies it. Up recalls
    pending input before walking through persistent prompt history.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application.current import get_app_session
    from prompt_toolkit.filters import to_filter
    from prompt_toolkit.history import FileHistory, History
    from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
    from prompt_toolkit.styles import Style

    class ClaimedHistory(History):
        def __init__(self, filename: Path) -> None:
            super().__init__()
            self._stored = FileHistory(str(filename))

        def load_history_strings(self) -> Iterable[str]:
            return self._stored.load_history_strings()

        def store_string(self, string: str) -> None:
            self._stored.store_string(string)

        def append_string(self, string: str) -> None:
            del string

        def commit(self, strings: tuple[str, ...]) -> None:
            for string in strings:
                if string:
                    History.append_string(self, string)

    async def no_pending_input() -> str | None:
        return None

    history = ClaimedHistory(HISTORY_PATH)
    app_session = get_app_session()
    session: Any = PromptSession(
        history=history,
        bottom_toolbar=status or (lambda: agent_summary() or None),
        style=Style.from_dict(
            {
                "bottom-toolbar": theme.panel,
                "repl-prompt": theme.prompt.prompt_toolkit,
            }
        ),
        key_bindings=input_bindings(
            on_interrupt or (lambda: None),
            on_shutdown or (lambda: None),
            recall_pending or no_pending_input,
            on_empty_eof or (lambda _now: True),
        ),
        # Redraw while idle so finished agents show up without a keypress.
        refresh_interval=0.5,
        erase_when_done=True,
        input=_replay.recording_input(app_session.input, replay),
        output=_replay.recording_output(app_session.output, replay),
    )
    if replay is not None:

        def record_editor_state(buffer: Any) -> None:
            replay.record(
                "editor_state",
                {
                    "text": str(buffer.text),
                    "cursor_position": int(buffer.cursor_position),
                },
            )

        session.default_buffer.on_text_changed += record_editor_state
        record_editor_state(session.default_buffer)
    input_window = session.app.layout.current_window
    if input_window is None or input_window.content is not session.app.layout.current_control:
        raise RuntimeError("prompt session does not expose its input window")
    input_window.dont_extend_height = to_filter(True)
    root_container = session.app.layout.container
    if not isinstance(root_container, HSplit) or not root_container.children:
        raise RuntimeError("prompt session does not expose its vertical layout")
    toolbar = root_container.children[-1]
    if (
        not isinstance(toolbar, ConditionalContainer)
        or not isinstance(toolbar.content, Window)
        or toolbar.content.style != "class:bottom-toolbar"
    ):
        raise RuntimeError("prompt session does not expose its bottom toolbar")
    root_container.children.insert(-1, Window(char=" ", style=""))
    session._axio_claimed_history = history
    original_accept = session.default_buffer.accept_handler
    if original_accept is None:
        raise RuntimeError("prompt session does not expose an accept handler")

    def capture_accept(buffer: Any) -> bool:
        target_agent_id = (capture_target or (lambda: "main"))()
        if not target_agent_id:
            raise RuntimeError("focused input target must not be empty")
        keep_text = bool(original_accept(buffer))
        buffer._axio_accepted_target = target_agent_id
        if str(buffer.text).strip() and reserve_sequence is not None:
            buffer._axio_accepted_seq = reserve_sequence()
        if str(buffer.text).strip():
            submitted_at = (accepted_at_provider or (lambda: datetime.now().astimezone()))()
            if submitted_at.tzinfo is None or submitted_at.utcoffset() is None:
                raise ValueError("accepted submission time must be timezone-aware")
            buffer._axio_accepted_at = submitted_at
        return keep_text

    session.default_buffer.accept_handler = capture_accept
    session.app.ttimeoutlen = ESCAPE_FLUSH_SECONDS
    return session


def commit_history(session: Any, texts: tuple[str, ...]) -> None:
    """Make claimed editor submissions available to persistent history."""

    history = getattr(session, "_axio_claimed_history", None)
    if history is None or not hasattr(history, "commit"):
        raise RuntimeError("prompt session does not expose claimed-input history")
    history.commit(texts)


def editor_text(session: Any) -> str:
    """Return the current editor without submitting or mutating it."""

    buffer = getattr(session, "default_buffer", None)
    text = getattr(buffer, "text", "")
    return text if isinstance(text, str) else ""


def accepted_target(session: Any, fallback: str) -> str:
    """Return the focus captured by the accept handler for the last Enter."""

    buffer = getattr(session, "default_buffer", None)
    value = getattr(buffer, "_axio_accepted_target", fallback)
    return value if isinstance(value, str) and value else fallback


def accepted_sequence(session: Any) -> int | None:
    """Return the logical sequence reserved by the last non-empty Enter."""

    buffer = getattr(session, "default_buffer", None)
    value = getattr(buffer, "_axio_accepted_seq", None)
    return value if isinstance(value, int) and value > 0 else None


def accepted_at(session: Any) -> datetime | None:
    """Return the local wall clock captured by the last non-empty Enter."""

    buffer = getattr(session, "default_buffer", None)
    value = getattr(buffer, "_axio_accepted_at", None)
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value


def complete_submission(session: Any, text: str, *, clear_editor: bool) -> None:
    """Finish one accepted Enter after its coordinator transaction settles."""

    buffer = getattr(session, "default_buffer", None)
    if buffer is None:
        return
    if clear_editor and getattr(buffer, "text", None) == text:
        buffer.reset()
    with suppress(AttributeError):
        del buffer._axio_accepted_target
    with suppress(AttributeError):
        del buffer._axio_accepted_seq
    with suppress(AttributeError):
        del buffer._axio_accepted_at
