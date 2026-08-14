"""Interactive REPL coding assistant powered by axio agent framework.

Auto-detects transport from available API keys (OPENAI_API_KEY, NEBIUS_API_KEY,
OPENROUTER_API_KEY), or use --transport to pick explicitly.

Run:
    axio-repl
    axio-repl "your prompt here"
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import os
import shutil
import signal
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, NamedTuple, cast
from uuid import uuid4

import aiohttp
from axio import notify
from axio.agent import Agent
from axio.blocks import TextBlock
from axio.context import ContextStore, MemoryContextStore
from axio.events import (
    AudioOutput,
    Error,
    ImageOutput,
    IterationEnd,
    ReasoningDelta,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    ToolFieldDelta,
    ToolFieldEnd,
    ToolFieldStart,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
    VideoOutput,
)
from axio.exceptions import StreamError
from axio.field import StrictStr
from axio.messages import Message
from axio.models import Capability, ModelSpec
from axio.tool import Tool
from axio.tool_args import ToolArgStream
from axio_tools_agents.monitoring import monitor
from axio_tools_agents.peers import (
    PeerMessage,
    PeerServer,
    enqueue_local_agent_prompt,
    format_message_for_dialog,
    interrupt_agent,
    is_local_background_agent,
    list_peers,
    local_background_agent_records,
    run_agent,
    send_message,
    set_background_outcome_handler,
    set_pending_message_probe,
    set_run_agent_factory,
    set_session_event_hub,
    set_spawn_agent_factory,
    spawn_agent,
    stop_agent,
    stop_local_background_agents,
    wait_local_background_agents_idle,
)
from axio_tools_agents.runtime import (
    AgentEventEnvelope,
    AgentStarted,
    AgentStopped,
    ConfigurationChanged,
    ContextCleared,
    ContextForked,
    ExecutionMode,
    ForegroundEntered,
    ForegroundExited,
    InputReceived,
    MessageCommitted,
    ObservedContextStore,
    OutcomeDelivered,
    RuntimeEvent,
    SessionEventHub,
    TurnFinished,
    TurnOutcome,
    TurnStarted,
    TurnStatus,
    new_turn_identity,
    observe_agent_turn,
)
from axio_tools_local.list_files import list_files
from axio_tools_local.patch_file import patch_file
from axio_tools_local.read_file import read_file
from axio_tools_local.shell import shell
from axio_tools_local.write_file import write_file

from axio_repl import _journal, _panel, _sandbox, _search
from axio_repl._multiplexer import ActionMultiplexer, DisplayMode, DisplayModeChange

LAST_ITERATION_HINT = Message(
    role="system",
    content=[
        TextBlock(
            text=(
                "This is your final iteration. A tool you call now will still run, but you will "
                "never see its result - the run ends before you are asked again. Answer with what "
                "you already have, and say plainly what you could not finish."
            )
        )
    ],
)
"""Delivered on the last iteration an agent is allowed.

Without it, running out of iterations produces nothing at all: the agent spends
its last turn on a tool call that is never dispatched, and the caller waiting on
it gets silence where the report should be.
"""

AGENT_NAME = "axio-repl"
AGENT_VERSION = "0.2.3"

# ── ANSI helpers ─────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


# ── Custom search tool ───────────────────────────────────────────────


async def search_files(
    query: StrictStr,
    path: StrictStr = ".",
    regex: bool = False,
    max_results: int = 100,
) -> str:
    """Search for text or regex patterns in files under a directory.
    Returns matching lines with file paths and line numbers."""
    return await asyncio.to_thread(_search.search, query, path, regex, max_results)


# ── Tools ────────────────────────────────────────────────────────────

TOOLS: list[Tool[Any]] = [
    Tool(name="read_file", handler=read_file),
    Tool(name="write_file", handler=write_file),
    Tool(name="patch_file", handler=patch_file),
    Tool(name="list_files", handler=list_files),
    Tool(name="search_files", handler=search_files),
    Tool(name="shell", handler=shell),
    Tool(name="interrupt_agent", handler=interrupt_agent),
    Tool(name="list_peers", handler=list_peers),
    Tool(name="monitor", handler=monitor),
    Tool(name="send_message", handler=send_message),
    Tool(name="run_agent", handler=run_agent, concurrency=1, detachable=False),
    Tool(name="spawn_agent", handler=spawn_agent, concurrency=3, detachable=False),
    Tool(name="stop_agent", handler=stop_agent),
]


# ── Transport auto-detection ─────────────────────────────────────────


def _discover_transports() -> dict[str, Callable[..., Any]]:
    result: dict[str, Callable[..., Any]] = {}
    for ep in entry_points(group="axio.transport"):
        try:
            result[ep.name] = ep.load()
        except Exception:
            pass
    return result


_TRANSPORT_ENV_VARS: dict[str, list[str]] = {
    "google": ["GEMINI_API_KEY"],
    "google-vertex": ["GOOGLE_GENAI_USE_VERTEXAI"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "nebius": ["NEBIUS_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}


def _transport_has_credentials(name: str) -> bool:
    env_vars = _TRANSPORT_ENV_VARS.get(name, [])
    return any(os.environ.get(v, "") for v in env_vars)


def _select_transport(name: str | None) -> tuple[Callable[..., Any], str]:
    available = _discover_transports()
    if name:
        if name not in available:
            print(
                f"Unknown transport {name!r}. Available: {', '.join(sorted(available))}",
                file=sys.stderr,
            )
            sys.exit(1)
        # Auto-detection picks a transport because its key is set; naming one
        # skipped the question entirely, and an unset key surfaced as a stack
        # trace from the first API call instead of the answer "set this".
        env_vars = _TRANSPORT_ENV_VARS.get(name, [])
        if env_vars and not _transport_has_credentials(name):
            print(f"No API key for {name}. Set {' or '.join(env_vars)}.", file=sys.stderr)
            sys.exit(1)
        return available[name], ""

    for transport_name, cls in available.items():
        if _transport_has_credentials(transport_name):
            return cls, ""

    print("No API key found. Set one of:", file=sys.stderr)
    for transport_name in available:
        env_vars = _TRANSPORT_ENV_VARS.get(transport_name, [])
        if env_vars:
            print(f"  {', '.join(env_vars)}  ({transport_name})", file=sys.stderr)
    sys.exit(1)


# ── AGENTS.md & system prompt ────────────────────────────────────────


def load_agents_instructions(root: Path) -> str:
    agents_file = root / "AGENTS.md"
    if not agents_file.exists():
        return ""
    try:
        return agents_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def build_system_prompt(
    root: Path,
    model: ModelSpec,
    tools: list[Tool[Any]],
    agents_text: str = "",
    parent_peer_id: str | None = None,
) -> str:
    caps = model.capabilities
    ctx_k = model.context_window // 1000
    out_k = model.max_output_tokens // 1000
    tool_names = ", ".join(t.name for t in tools)

    has_tools = Capability.tool_use in caps

    lines = [
        f"You are {AGENT_NAME} (v{AGENT_VERSION}) — a terminal coding assistant.",
        f"Model: {model.id} ({ctx_k}K context, {out_k}K max output)",
        f"Current directory: {root} (perform actions here unless specified otherwise)",
    ]
    if has_tools:
        lines.append(f"Tools: {tool_names}")
    lines.append("")

    # Capability-aware guidance
    cap_notes: list[str] = []
    if Capability.vision in caps:
        cap_notes.append("You can see images via read_file (screenshots, diagrams, photos).")
    if Capability.audio in caps:
        cap_notes.append("You can listen to audio files via read_file (speech, music, podcasts).")
    if Capability.video in caps:
        cap_notes.append("You can see video files via read_file.")
    if Capability.image_generation in caps:
        cap_notes.append("You can generate images inline — describe what to draw in your response.")
    if Capability.reasoning in caps:
        cap_notes.append("Extended thinking is available for complex reasoning.")
    if cap_notes:
        lines += cap_notes + [""]

    lines.append("Rules:")
    if has_tools:
        lines += [
            "- Start every task by listing the current directory to understand the project.",
            "- Read files before editing. Use line_numbers=True before patch_file.",
            "- Keep edits minimal and targeted — don't reformat surrounding code.",
            "- Ground answers on project context gathered through tools.",
        ]
    lines += [
        "- Write idiomatic code — follow the conventions and best practices of the "
        "languages and frameworks used in the project.",
        "- When the user asks about a file they provided or you read, base your answer "
        "strictly on the actual file contents. Do not guess, assume, or fill in details "
        "from general knowledge — only state what the file actually contains.",
        f"- Your max output is {out_k}K tokens. Use as many as the task requires — "
        "do not stop early. If the user asks for a full transcript, detailed analysis, "
        f"or comprehensive review, produce the complete output up to the {out_k}K limit.",
        "- Never refuse safe requests or claim inability.",
    ]
    if has_tools:
        lines += [
            "- If a tool call fails, analyze the error and try a different approach. "
            "If stuck after 3 attempts at the same sub-problem, "
            "explain what you tried and ask for guidance.",
            "- Do not return a final answer until all necessary work is done or you are stuck.",
            "- For compound requests, build a checklist of all items and verify each is addressed before finishing.",
            "- Don't narrate your tool calls — the user sees their full output.",
            "- After completing work, summarize what changed briefly.",
            "- Not tested — not done. Always run tests or builds to verify your changes. "
            "Re-read edited files, observe actual results — don't assume success "
            "from exit codes alone.",
            "- After any test or build that produces images or video, you MUST read_file "
            "every output file to actually see the results. Never describe visual output "
            "you haven't viewed. 'Tests passed' is not the same as 'I looked at the "
            "screenshots and they look correct'.",
            "- To verify UI, use browser automation (Playwright, Puppeteer) to capture "
            "real screenshots at multiple viewport sizes (desktop 1280×800, tablet 768×1024, "
            "mobile 375×667), then read_file every screenshot.",
            "- When you read a screenshot, you MUST critically analyze it. List every "
            "visual defect you notice: broken layout, text overflow, misaligned elements, "
            "poor contrast, missing images, clipped content, wrong spacing, responsive "
            "issues. Do NOT say 'looks good' unless you can specifically confirm each "
            "aspect is correct.",
            "- UI review is iterative: screenshot → list issues → fix code → re-screenshot "
            "→ verify fixes. Repeat until zero defects. Never declare UI done after a "
            "single screenshot pass.",
            "- Never use generate_image as a substitute for real UI testing.",
            "- Never run destructive shell commands (rm -rf, git reset --hard) without user confirmation.",
            "- For large files, read specific line ranges instead of the entire file.",
            "- Tools that advertise background=true return a handle at once and keep running. The result is "
            "delivered automatically — injected into your turn if you are still working, or as your next prompt "
            "if you have already finished — so you never need to poll for it. Use monitor(tasks=[handle]) only "
            "when you cannot proceed without the result right now. Use background=true for calls slow enough to "
            "be worth doing while you carry on — a test suite, a build, a long download. Do not use it when you "
            "need the result to decide your next step, and never for quick reads: the handle costs an extra "
            "round trip.",
        ]
        if any(t.name == "run_agent" for t in tools):
            lines += [
                "- Use run_agent for one bounded delegation whose result you need before continuing. The parent "
                "waits, the child streams live in the foreground, and its final answer returns as this tool result.",
                "- A run_agent child is one-shot and has no peer messaging or orchestration tools. Give it a complete "
                "task; do not ask it to send_message, spawn more agents, or wait for later instructions.",
                "- Prefer spawn_agent instead when independent work should continue in parallel while you do "
                "other work.",
            ]
        if any(t.name == "spawn_agent" for t in tools):
            lines += [
                "- Use spawn_agent for independent parallel work that can run in the background. It returns "
                "immediately with a global agent_id; it does not return the child agent's final answer.",
                "- Spawned agents start with empty context by default. Set inherit_context=true only when the child "
                "must see the current conversation; otherwise include all required instructions in task.",
                "- Write the task you give a spawned agent in English, whatever language this conversation is in. "
                "Models follow instructions more reliably in English, and the child has none of the context that "
                "would let it recover from an ambiguous phrasing. Answer the user in their own language as usual.",
                "- A spawned child's final answer is delivered to the parent automatically after each turn. "
                "send_message is only for additional peer communication, not for repeating the final answer.",
                "- Use list_peers() to discover running agents in this project, or list_peers(all_projects=true) "
                "to inspect all local agent ids. Use send_message(agent_id=..., message=...) for IPC by global id.",
                "- A spawned child's completion or failure is announced to you automatically — injected into your "
                "dialog as your next prompt after the current response — so you never need to poll or monitor just "
                "to learn a child is done. The announcement tells you the "
                "child is done and its completed answer is delivered with the background report. Call "
                "monitor(agents=[...], "
                "wait_all=true) only when you must join a swarm within this turn before proceeding; it reports a "
                "crashed child as finished, with its error. monitor(messages=true) waits for a child's PEER "
                "MESSAGE (sent via send_message) — those are still delivered only as your next prompt, never "
                "injected mid-turn. Use paths=/pids= to wait on files or processes. A timeout returns what is "
                "still outstanding rather than failing — so decide from that report whether to wait again.",
                "- An idle notification is information, not a request for a reply: do not send_message back to a "
                "child just because it went idle. Only message it again when you actually need something from "
                "it — otherwise a reply wakes its next turn, which can notify you again, and so on.",
                "- Use interrupt_agent(agent_id=...) to cancel a spawned agent's current response while keeping it "
                "alive. Use stop_agent(agent_id=...) only when the child should exit. A parent may interrupt or "
                "stop its own children by id.",
                "- In axio-repl, the user can switch the active local agent with /agent-focus, list them with "
                "/agents, interrupt with /agent-interrupt, and stop with /agent-stop. Only the focused agent "
                "streams fully. /agent-actions on shows framed tool and lifecycle actions from every other agent "
                "between complete paragraphs or tool calls without exposing their prose or reasoning.",
            ]
        if parent_peer_id is not None and any(t.name == "send_message" for t in tools):
            lines.append(
                f"- This REPL session is registered as peer {parent_peer_id!r}. When reporting back by IPC, use "
                f"send_message(agent_id={parent_peer_id!r}, message=<report>)."
            )
    lines.append("")

    if agents_text:
        lines += ["AGENTS.md instructions:", agents_text, ""]

    return "\n".join(lines)


# ── Readline history ─────────────────────────────────────────────────


# ── Event rendering ──────────────────────────────────────────────────


class _AgentRenderState:
    def __init__(self) -> None:
        self.in_text = False
        self.in_reasoning = False
        self.arg_streams: dict[str, ToolArgStream] = {}
        self.active_tool_ids: set[str] = set()
        self.streamed_tool_ids: set[str] = set()
        self.field_first_delta = True
        self.field_key: str | None = None
        self.background_text: list[str] = []
        self.background_reported_chars = 0
        self.pending_text: list[str] = []
        self.background_tools: list[str] = []
        self.background_errors: list[str] = []
        self.background_events: list[StreamEvent] = []
        self.paragraph_newline_pending = False


class ReplRenderer:
    def __init__(
        self,
        *,
        buffer_background_events: bool = False,
        stats: _panel.SessionStats | None = None,
        current_model: Callable[[], ModelSpec | None] | None = None,
        display_mode: DisplayMode = DisplayMode.ACTIVE_ONLY,
        action_multiplexer: ActionMultiplexer | None = None,
        action_boundary_frames: int = 4,
        action_boundary_bytes: int = 16 * 1024,
    ) -> None:
        self._lock = asyncio.Lock()
        self._buffer_background_events = buffer_background_events
        # Every agent's events pass through here, which makes this the one place
        # that sees the whole session's spend.
        self._stats = stats
        self._current_model = current_model
        self._states: dict[str, _AgentRenderState] = {}
        self._active_agent: str | None = None
        self._focused_agent = "main"
        self._foreground_stack: list[str] = []
        self._foreground_parent_calls: dict[str, str] = {}
        self._streamed_foreground_calls: dict[str, TurnStatus] = {}
        self._foreground_streaming = False
        self._safe_boundary_open = False
        self._background_pending: set[str] = set()
        self._input_active = False
        self._actions = action_multiplexer or ActionMultiplexer(display_mode)
        self._action_boundary_frames = action_boundary_frames
        self._action_boundary_bytes = action_boundary_bytes

    @property
    def focused_agent(self) -> str:
        return self._focused_agent

    @property
    def foreground_agent(self) -> str:
        return self._foreground_stack[-1] if self._foreground_stack else self._focused_agent

    @property
    def display_mode(self) -> DisplayMode:
        return self._actions.mode

    @property
    def queued_action_count(self) -> int:
        return self._actions.queued_count

    def action_status(self) -> str:
        status = f"actions: {self.display_mode.value}"
        if self.queued_action_count:
            status += f" ({self.queued_action_count} queued)"
        return status

    async def set_display_mode(self, mode: DisplayMode) -> DisplayModeChange:
        async with self._lock:
            return self._actions.set_mode(mode)

    async def enter_foreground(self, agent_id: str, parent_tool_use_id: str | None = None) -> None:
        async with self._lock:
            self._foreground_stack.append(agent_id)
            self._safe_boundary_open = False
            self._actions.discard_agent(agent_id)
            if parent_tool_use_id is not None:
                self._foreground_parent_calls[agent_id] = parent_tool_use_id

    async def exit_foreground(self, agent_id: str, status: TurnStatus) -> None:
        async with self._lock:
            if self._foreground_stack and self._foreground_stack[-1] == agent_id:
                self._foreground_stack.pop()
            else:
                with suppress(ValueError):
                    self._foreground_stack.remove(agent_id)
            parent_tool_use_id = self._foreground_parent_calls.pop(agent_id, None)
            if parent_tool_use_id is not None:
                self._streamed_foreground_calls[parent_tool_use_id] = status
            self._safe_boundary_open = False
            resumed = self._state(self.foreground_agent)
            self._foreground_streaming = bool(
                resumed.in_text or resumed.in_reasoning or resumed.active_tool_ids or resumed.arg_streams
            )

    def set_focus(self, agent_id: str) -> None:
        self._focused_agent = agent_id
        self._safe_boundary_open = True
        self._actions.discard_agent(agent_id)
        state = self._state(agent_id)
        buffered = state.background_events
        state.background_events = []
        state.background_text.clear()
        state.background_reported_chars = 0
        state.background_tools.clear()
        state.background_errors.clear()
        self._background_pending.discard(agent_id)
        for event in buffered:
            self._render_locked(agent_id, event)
            if isinstance(event, Error | SessionEndEvent):
                self._foreground_streaming = False
            elif not isinstance(event, IterationEnd):
                self._foreground_streaming = True

    def set_input_active(self, active: bool) -> None:
        self._input_active = active

    def take_pending_text(self, agent_id: str) -> str:
        """What the agent had written when it was cut off, and not yet stored.

        The agent appends an iteration to the context once the model stops
        talking; interrupt it before that and the words are on screen and
        nowhere else, so the next turn is answered by a model that never said
        them.
        """
        state = self._state(agent_id)
        text = "".join(state.pending_text).strip()
        state.pending_text.clear()
        return text

    def _flush(self) -> None:
        """Push a half-written line out, unless the input prompt is up.

        prompt_toolkit holds a line until its newline arrives, so that a partial
        line and the prompt drawn below it never end up on the same row.
        Flushing defeats that, and the output loses its first characters to the
        next redraw. While the prompt is up the half-line therefore waits: text
        appears by the line rather than by the token, which is what it costs to
        be able to type at any moment.
        """
        if not self._input_active:
            sys.stdout.flush()

    async def render(self, agent_id: str, event: StreamEvent) -> None:
        async with self._lock:
            if isinstance(event, IterationEnd) and self._stats is not None:
                model = self._current_model() if self._current_model is not None else None
                self._stats.record(agent_id, event.usage, model)
            if agent_id == self.foreground_agent:
                self._render_locked(agent_id, event)
                if isinstance(event, Error | SessionEndEvent):
                    self._foreground_streaming = False
                    if self.display_mode is DisplayMode.ACTIVE_ONLY:
                        self._flush_background_summaries_locked()
                elif not isinstance(event, IterationEnd):
                    self._foreground_streaming = True
            else:
                self._actions.observe(agent_id, event)
                self._record_background_event_locked(agent_id, event)
                if self.display_mode is DisplayMode.ALL_ACTIONS and self._safe_boundary_open:
                    self._drain_safe_boundary_locked(max_frames=1)
                elif self.display_mode is DisplayMode.ACTIVE_ONLY and not self._foreground_streaming:
                    self._flush_background_summaries_locked()

    async def observe_runtime_event(self, agent_id: str, event: RuntimeEvent) -> None:
        async with self._lock:
            if agent_id != self.foreground_agent:
                self._actions.observe(agent_id, event)
                if self.display_mode is DisplayMode.ALL_ACTIONS and self._safe_boundary_open:
                    self._drain_safe_boundary_locked(max_frames=1)

    async def mark_idle(self) -> None:
        async with self._lock:
            self._foreground_streaming = False
            self._safe_boundary_open = True
            if self.display_mode is DisplayMode.ALL_ACTIONS:
                self._drain_all_actions_locked()
            else:
                self._flush_background_summaries_locked()

    async def incoming(self, text: str) -> None:
        """Put an arriving message on screen, not only into the model's prompt.

        A report from a spawned agent was fed to the model and never shown, so
        the only account of it the user ever saw was the model's summary of
        something they could not read.
        """
        async with self._lock:
            if self._active_agent is not None and self._state(self._active_agent).in_text:
                print()
                self._state(self._active_agent).in_text = False
            print(f"\n{DIM}{'─' * 3} incoming {'─' * 3}{RESET}\n{text}\n")
            self._safe_boundary_open = True
            self._drain_safe_boundary_locked()

    async def notice(self, text: str) -> None:
        async with self._lock:
            if self._active_agent is not None and self._state(self._active_agent).in_text:
                print()
                self._state(self._active_agent).in_text = False
            print(f"{DIM}{text}{RESET}")
            self._safe_boundary_open = True
            self._drain_safe_boundary_locked()

    def _state(self, agent_id: str) -> _AgentRenderState:
        return self._states.setdefault(agent_id, _AgentRenderState())

    def _switch_agent(self, agent_id: str) -> _AgentRenderState:
        switched = self._active_agent != agent_id
        if switched:
            if self._active_agent is not None and self._state(self._active_agent).in_text:
                print()
                self._state(self._active_agent).in_text = False
            if self._active_agent is not None or agent_id != "main":
                print(f"\n{DIM}── {agent_id} ──{RESET}")
            self._active_agent = agent_id
        state = self._state(agent_id)
        if switched and state.field_key is not None:
            sys.stdout.write(f"\n  {YELLOW}{state.field_key} (continued){RESET}: {DIM}")
            self._flush()
            state.field_first_delta = True
        return state

    def _render_locked(self, agent_id: str, event: StreamEvent) -> None:  # noqa: C901
        state = self._switch_agent(agent_id)
        if not isinstance(event, TextDelta):
            state.paragraph_newline_pending = False
        # Reasoning streams in as one delta per token, so the quote marker and
        # the colour reset belong to the run as a whole, not to every delta.
        # Closing it here covers every kind of event that can follow.
        if state.in_reasoning and not isinstance(event, ReasoningDelta):
            sys.stdout.write(f"{RESET}\n")
            self._flush()
            state.in_reasoning = False
            self._safe_boundary_open = True
            self._drain_safe_boundary_locked()
        match event:
            case ReasoningDelta(delta=delta):
                self._safe_boundary_open = False
                if state.in_text:
                    print()
                    state.in_text = False
                if not state.in_reasoning:
                    # <think> is usually followed by a newline, which would open
                    # the quote with an empty line.
                    delta = delta.lstrip("\n")
                    if not delta:
                        return
                    sys.stdout.write(f"{DIM}> ")
                    state.in_reasoning = True
                sys.stdout.write(delta.replace("\n", "\n> "))
                self._flush()

            case TextDelta(delta=delta):
                if not state.in_text:
                    state.in_text = True
                state.pending_text.append(delta)
                if "[Output truncated:" in delta:
                    sys.stdout.write(f"\n{RED}{delta.strip()}{RESET}\n")
                    state.in_text = False
                    state.paragraph_newline_pending = False
                    self._safe_boundary_open = True
                    self._drain_safe_boundary_locked()
                else:
                    self._render_text_delta_locked(state, delta)

            case ImageOutput(data=data, media_type=mt):
                self._safe_boundary_open = False
                if state.in_text:
                    print()
                    state.in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[image saved: {path}]{RESET}")
                self._safe_boundary_open = True
                self._drain_safe_boundary_locked()

            case AudioOutput(data=data, media_type=mt):
                self._safe_boundary_open = False
                if state.in_text:
                    print()
                    state.in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[audio saved: {path}]{RESET}")
                self._safe_boundary_open = True
                self._drain_safe_boundary_locked()

            case VideoOutput(data=data, media_type=mt):
                self._safe_boundary_open = False
                if state.in_text:
                    print()
                    state.in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[video saved: {path}]{RESET}")
                self._safe_boundary_open = True
                self._drain_safe_boundary_locked()

            case ToolUseStart(index=index, tool_use_id=tid, name=name):
                self._safe_boundary_open = False
                if state.in_text:
                    print()
                    state.in_text = False
                sys.stdout.write(f"\n{BOLD}{CYAN}\u25b6 {name}{RESET}")
                self._flush()
                state.arg_streams[tid] = ToolArgStream(tid, index)
                state.active_tool_ids.add(tid)

            case ToolInputDelta(tool_use_id=tid, partial_json=pj):
                self._safe_boundary_open = False
                stream = state.arg_streams.get(tid)
                if stream:
                    for fe in stream.feed(pj):
                        self._render_field_event(state, fe)
                    if stream.done:
                        sys.stdout.write("\n")
                        self._flush()
                        del state.arg_streams[tid]

            case ToolOutputDelta(tool_use_id=tid, key=key, delta=delta):
                self._safe_boundary_open = False
                if tid not in state.streamed_tool_ids:
                    sys.stdout.write("\n")
                state.streamed_tool_ids.add(tid)
                color = RED if key == "stderr" else DIM
                sys.stdout.write(f"{color}{delta}{RESET}")
                self._flush()

            case ToolResult(tool_use_id=tid, name=name, is_error=is_error, content=content):
                foreground_status = self._streamed_foreground_calls.pop(tid, None)
                if foreground_status is not None:
                    if foreground_status is TurnStatus.SUCCEEDED:
                        content = "[foreground agent returned its result to the parent]"
                    elif foreground_status is TurnStatus.CANCELLED:
                        content = "[foreground agent was cancelled; the outcome was returned to the parent]"
                    else:
                        content = "[foreground agent failed; the outcome was returned to the parent]"
                    color = GREEN if foreground_status is TurnStatus.SUCCEEDED else RED
                    sys.stdout.write(f"{RESET}\n{color}{content}{RESET}\n")
                elif is_error:
                    sys.stdout.write(f"{RESET}\n{RED}{content}{RESET}\n")
                elif name in {"run_agent", "spawn_agent"}:
                    sys.stdout.write(f"{RESET}\n{GREEN}{content}{RESET}\n")
                elif tid in state.streamed_tool_ids:
                    sys.stdout.write(f"{RESET}\n")
                else:
                    sys.stdout.write(f"{RESET}\n{GREEN}{content}{RESET}\n")
                self._flush()
                state.active_tool_ids.discard(tid)
                state.arg_streams.pop(tid, None)
                if not state.active_tool_ids and not state.arg_streams:
                    self._safe_boundary_open = True
                    self._drain_safe_boundary_locked()
                else:
                    self._safe_boundary_open = False

            case IterationEnd():
                # The agent has written this iteration into the context itself;
                # what is kept here is only ever the unfinished tail.
                state.pending_text.clear()

            case Error(exception=exc):
                print(f"\n{RED}Error: {exc}{RESET}", file=sys.stderr)
                self._safe_boundary_open = True
                self._drain_safe_boundary_locked()

            case SessionEndEvent(total_usage=usage):
                if state.in_text:
                    print()
                    state.in_text = False
                print(f"{DIM}[{usage.input_tokens}in/{usage.output_tokens}out tokens]{RESET}")
                state.paragraph_newline_pending = False
                self._safe_boundary_open = True
                self._drain_safe_boundary_locked()

    def _record_background_event_locked(self, agent_id: str, event: StreamEvent) -> None:
        state = self._state(agent_id)
        if self._buffer_background_events:
            state.background_events.append(event)
        match event:
            case TextDelta(delta=delta):
                state.background_text.append(delta)
            case ToolUseStart(name=name):
                if name not in state.background_tools:
                    state.background_tools.append(name)
            case Error(exception=exc):
                state.background_errors.append(str(exc))
                if not self._buffer_background_events:
                    self._background_pending.add(agent_id)
            case SessionEndEvent():
                self._finish_background_report_locked(agent_id)
                if not self._buffer_background_events:
                    self._background_pending.add(agent_id)
            case _:
                pass

    def _finish_background_report_locked(self, agent_id: str) -> None:
        state = self._state(agent_id)
        text = "".join(state.background_text).strip()
        state.background_text.clear()
        if not text:
            return
        state.background_reported_chars = len(text)

    def _flush_background_summaries_locked(self) -> None:
        if not self._background_pending:
            return
        for agent_id in sorted(self._background_pending):
            state = self._state(agent_id)
            parts = [f"{agent_id} completed"]
            if state.background_tools:
                parts.append(f"tools={','.join(state.background_tools)}")
            if state.background_reported_chars:
                parts.append(f"reported {state.background_reported_chars} chars")
            if state.background_errors:
                # The count alone says something went wrong and not what, which
                # is the half that decides whether to retry, fix or give up.
                parts.append(f"error: {state.background_errors[-1]}")
            print(f"{DIM}[background {'; '.join(parts)}]{RESET}")
            state.background_reported_chars = 0
            state.background_tools.clear()
            state.background_errors.clear()
        self._background_pending.clear()

    def _discard_background_summaries_locked(self) -> None:
        for agent_id in self._background_pending:
            state = self._state(agent_id)
            state.background_reported_chars = 0
            state.background_tools.clear()
            state.background_errors.clear()
        self._background_pending.clear()

    def _drain_safe_boundary_locked(self, *, max_frames: int | None = None) -> None:
        if self.display_mode is DisplayMode.ALL_ACTIONS:
            self._discard_background_summaries_locked()
            for frame in self._actions.drain(
                max_frames=max_frames or self._action_boundary_frames,
                max_bytes=self._action_boundary_bytes,
            ):
                sys.stdout.write(frame)
            self._flush()

    def _drain_all_actions_locked(self) -> None:
        while self._actions.queued_count:
            before = self._actions.queued_count
            self._drain_safe_boundary_locked()
            if self._actions.queued_count >= before:
                break

    def _render_text_delta_locked(self, state: _AgentRenderState, delta: str) -> None:
        if delta:
            self._safe_boundary_open = False
        start = 0
        for index, character in enumerate(delta):
            if character == "\n":
                if state.paragraph_newline_pending:
                    sys.stdout.write(delta[start : index + 1])
                    self._flush()
                    self._safe_boundary_open = True
                    self._drain_safe_boundary_locked()
                    start = index + 1
                    state.paragraph_newline_pending = False
                else:
                    state.paragraph_newline_pending = True
            else:
                state.paragraph_newline_pending = False
        if start < len(delta):
            self._safe_boundary_open = False
            sys.stdout.write(delta[start:])
            self._flush()

    def _render_field_event(
        self,
        state: _AgentRenderState,
        event: ToolFieldStart | ToolFieldDelta | ToolFieldEnd,
    ) -> None:
        match event:
            case ToolFieldStart(key=key):
                sys.stdout.write(f"\n  {YELLOW}{key}{RESET}: {DIM}")
                self._flush()
                state.field_key = key
                state.field_first_delta = True
            case ToolFieldDelta(text=text):
                if state.field_first_delta and "\n" in text:
                    sys.stdout.write("\n")
                state.field_first_delta = False
                sys.stdout.write(text)
                self._flush()
            case ToolFieldEnd():
                sys.stdout.write(RESET)
                self._flush()
                state.field_key = None


async def render_runtime_event(renderer: ReplRenderer, envelope: AgentEventEnvelope) -> None:
    match envelope.event:
        case ForegroundEntered():
            await renderer.enter_foreground(envelope.agent_id, envelope.parent_tool_use_id)
        case ForegroundExited(status=status):
            await renderer.exit_foreground(envelope.agent_id, status)
        case (
            AgentStarted()
            | AgentStopped()
            | TurnStarted()
            | TurnFinished()
            | OutcomeDelivered()
            | InputReceived()
            | ConfigurationChanged()
            | MessageCommitted()
            | ContextForked()
            | ContextCleared()
        ):
            if envelope.execution_mode is ExecutionMode.BACKGROUND:
                await renderer.observe_runtime_event(envelope.agent_id, envelope.event)
        case event:
            await renderer.render(envelope.agent_id, event)


async def run_prompt(
    agent: Agent,
    ctx: ContextStore,
    prompt: str,
    event_hub: SessionEventHub,
    run_id: str,
    *,
    source: str,
) -> TurnOutcome:
    identity = new_turn_identity(
        agent_id="main",
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        run_id=run_id,
        context_id=ctx.session_id,
    )
    await event_hub.publish_for(identity, InputReceived(text=prompt, source=source))
    return await observe_agent_turn(agent=agent, context=ctx, prompt=prompt, identity=identity, hub=event_hub)


def _clone_transport_for_spawn(transport: Any) -> Any:
    clone = copy.copy(transport)
    if dataclasses.is_dataclass(clone):
        for field_info in dataclasses.fields(clone):
            value = getattr(clone, field_info.name)
            if isinstance(value, dict):
                setattr(clone, field_info.name, dict(value))
            elif isinstance(value, list):
                setattr(clone, field_info.name, list(value))
            elif isinstance(value, set):
                setattr(clone, field_info.name, set(value))
    if hasattr(clone, "last_usage"):
        setattr(clone, "last_usage", None)
    if hasattr(clone, "_thought_signatures"):
        setattr(clone, "_thought_signatures", {})
    return clone


_AGENT_ORCHESTRATION_TOOLS = frozenset(
    {"run_agent", "spawn_agent", "send_message", "list_peers", "monitor", "interrupt_agent", "stop_agent"}
)


def _clone_tools_for_child(tools: list[Tool[Any]], *, foreground: bool) -> list[Tool[Any]]:
    excluded = _AGENT_ORCHESTRATION_TOOLS if foreground else {"run_agent", "spawn_agent"}
    return [
        Tool(
            name=tool.name,
            description=tool.description,
            handler=tool.handler,
            guards=tool.guards,
            context=tool.context,
            concurrency=tool.concurrency,
            detachable=tool.detachable and not foreground,
        )
        for tool in tools
        if tool.name not in excluded
    ]


_media_counter = 0


def _save_media(data: bytes, media_type: str) -> str:
    """Save media bytes to a temp file, return the path."""
    import tempfile

    global _media_counter
    _media_counter += 1
    ext = media_type.split("/")[-1].split(";")[0]
    fd, path = tempfile.mkstemp(suffix=f".{ext}", prefix=f"axio_{_media_counter:03d}_")
    os.write(fd, data)
    os.close(fd)
    return path


# ── Input handling ───────────────────────────────────────────────────


async def _read_input_async(session: Any, renderer: ReplRenderer, on_interrupt: Callable[[], None]) -> str:
    from prompt_toolkit.patch_stdout import patch_stdout

    renderer.set_input_active(True)
    try:
        # raw=True keeps our own ANSI colouring intact while the prompt is up.
        with patch_stdout(raw=True):
            while True:
                try:
                    return str(await session.prompt_async("repl> ")).strip()
                except KeyboardInterrupt:
                    # The prompt is up for the whole session now, and it puts the
                    # terminal in raw mode - so Ctrl+C arrives here as a keypress
                    # and never reaches the signal handler that used to stop a
                    # running turn. Do its job, and go back to waiting.
                    on_interrupt()
    finally:
        renderer.set_input_active(False)


def _peer_name(root: Path) -> str:
    return f"axio-repl:{root.name}:{os.getpid()}"


def _resolve_local_agent_id(value: str) -> str | None:
    if value == "main":
        return "main"
    records = local_background_agent_records()
    exact = [record.id for record in records if record.id == value or record.name == value]
    if len(exact) == 1:
        return exact[0]
    prefixed = [record.id for record in records if record.id.startswith(value)]
    if len(prefixed) == 1:
        return prefixed[0]
    return None


def _resolve_command_agent_id(value: str, renderer: ReplRenderer) -> str | None:
    if value == "current":
        return renderer.focused_agent
    return _resolve_local_agent_id(value)


def _show_agents(renderer: ReplRenderer) -> None:
    print(f"Focused agent: {BOLD}{renderer.focused_agent}{RESET}")
    records = local_background_agent_records()
    if not records:
        print("No local background agents.")
        return
    for record in records:
        marker = "*" if record.id == renderer.focused_agent else " "
        print(f"{marker} {record.id} name={record.name!r} kind={record.kind} pid={record.pid}")


async def _handle_agent_actions(
    renderer: ReplRenderer,
    value: str,
    publish: Callable[[RuntimeEvent], Awaitable[None]] | None = None,
) -> bool:
    """Show or update the observation-only background action policy."""
    if not value:
        print(f"Agent actions: {BOLD}{renderer.display_mode.value}{RESET}; {renderer.queued_action_count} queued")
        return True
    try:
        mode = DisplayMode.parse(value)
    except ValueError as exc:
        print(str(exc))
        return False
    change = await renderer.set_display_mode(mode)
    detail = ""
    if change.discarded_frames:
        detail = f"; discarded {change.discarded_frames} queued frame(s) ({change.discarded_bytes} bytes)"
    print(f"Agent actions: {BOLD}{mode.value}{RESET}{detail}")
    if change.current is not change.previous and publish is not None:
        await publish(ConfigurationChanged(name="agent_actions", value=mode.value, source="interactive"))
    return True


# ── REPL commands ────────────────────────────────────────────────────


class Command(NamedTuple):
    """A REPL command with separate show (no arg) and apply (with arg) modes."""

    show: Callable[[], None]
    apply: Callable[[str], None]


# CLI arg attr → slash command name (for unified init).
_CLI_TO_SLASH: dict[str, str] = {
    "thinking": "/thinking",
    "temperature": "/temperature",
    "max_tokens": "/max-tokens",
    "debug": "/debug",
}


def _apply_cli_args(args: object, commands: dict[str, Command]) -> None:
    """Apply CLI arguments through the same command handlers as slash commands."""
    for attr, cmd_name in _CLI_TO_SLASH.items():
        val: Any = getattr(args, attr, None)
        if val is None or val is False:
            continue
        arg = "on" if isinstance(val, bool) else val if isinstance(val, str) else str(val)
        commands[cmd_name].apply(arg)


# ── model ──


class _ConciseFormatter(logging.Formatter):
    """Drop the traceback, keep the message.

    A tool failing is ordinary for an agent — a missing file is a normal
    outcome of exploring, not an incident — and the model is told about it
    through the tool result anyway. Dumping a stack trace per failure buries
    the session, and background agents make it worse by interleaving.
    """

    def format(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)
        record.exc_info = None
        record.exc_text = None
        return super().format(record)


def setup_logging(debug: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    fmt = "%(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt) if debug else _ConciseFormatter(fmt))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if debug else logging.WARNING)


def _columnise(items: list[str], width: int, gap: int = 2) -> list[str]:
    """Lay items out in columns, filling downwards the way ls does."""
    if not items:
        return []
    column_width = max(len(i) for i in items) + gap
    columns = max(1, width // column_width)
    rows = -(-len(items) // columns)
    lines = []
    for row in range(rows):
        cells = [items[i].ljust(column_width) for c in range(columns) if (i := c * rows + row) < len(items)]
        lines.append("".join(cells).rstrip())
    return lines


def _show_model(transport: Any) -> None:
    model = transport.model
    caps = ", ".join(sorted(c.value for c in model.capabilities))
    print(f"Current model: {BOLD}{model.id}{RESET}")
    print(f"Capabilities: {caps}")
    print("Available:")
    width = shutil.get_terminal_size((100, 24)).columns
    for line in _columnise(sorted(transport.models.keys()), max(20, width - 2)):
        print(f"  {line}")


def _resolve_model_arg(transport: Any, model_id: str) -> ModelSpec:
    resolver = getattr(transport, "resolve_model", None)
    if callable(resolver):
        return cast(ModelSpec, resolver(model_id))
    return cast(ModelSpec, transport.models[model_id])


def _adopt_catalogue_metadata(transport: Any) -> None:
    """Replace the default model placeholder with the catalogue's own entry.

    A transport's default is written as a bare ``ModelSpec``: an id and nothing
    else, so it claims no capabilities, 128K of context and 8K of output. Left
    that way after the catalogue is loaded, the agent reads it as "this model
    cannot use tools" and runs with none of them - silently, since nothing about
    the session says the tools were dropped.
    """
    with suppress(KeyError, AttributeError):
        transport.model = _resolve_model_arg(transport, transport.model.id)


def _choose_model(transport: Any, arg: str) -> ModelSpec | None:
    """Resolve *arg* to one model, reporting why when it cannot."""
    try:
        return _resolve_model_arg(transport, arg)
    except KeyError:
        pass

    matches = transport.models.search(arg)
    if len(matches) == 1:
        return cast(ModelSpec, next(iter(matches.values())))

    # Ids are usually vendor-prefixed, so typing the bare name names one model
    # exactly even when it is a substring of several others.
    exact = [m for k, m in matches.items() if k.rsplit("/", 1)[-1].casefold() == arg.casefold()]
    if len(exact) == 1:
        return cast(ModelSpec, exact[0])

    if not matches:
        print(f"No model matching {arg!r}. Available: {', '.join(transport.models.keys())}")
    else:
        print(f"Ambiguous — matches: {', '.join(matches.keys())}")
    return None


def _apply_model(
    transport: Any,
    agent: Agent,
    tools: list[Tool[Any]],
    root: Path,
    agents_text: str,
    arg: str,
    parent_peer_id: str | None = None,
) -> None:
    chosen = _choose_model(transport, arg)
    if chosen is None:
        return
    transport.model = chosen
    agent.system = build_system_prompt(root, transport.model, tools, agents_text, parent_peer_id=parent_peer_id)
    print(f"Switched to {BOLD}{transport.model.id}{RESET}")


# ── thinking ──


def _show_thinking(transport: Any) -> None:
    level = getattr(transport, "thinking_level", None)
    budget = getattr(transport, "thinking_budget", None)
    get_opts = getattr(transport, "get_thinking_options", None)
    valid_levels = get_opts() if get_opts else None
    if level:
        print(f"Thinking level: {BOLD}{level}{RESET}")
    elif budget is not None:
        print(f"Thinking budget: {BOLD}{budget}{RESET} tokens")
    else:
        print("Thinking: default")
    if valid_levels is not None:
        print(f"Valid levels: {', '.join(valid_levels)}")
    elif get_opts is not None:
        print("Usage: /thinking <budget_tokens>")


def _apply_thinking(transport: Any, arg: str) -> None:
    get_opts = getattr(transport, "get_thinking_options", None)
    valid_levels = get_opts() if get_opts else None
    if arg.isdigit():
        if valid_levels is not None:
            model_id = getattr(getattr(transport, "model", None), "id", "?")
            print(f"{model_id} uses thinking levels, not token budgets.")
            print(f"Valid levels: {', '.join(valid_levels)}")
            return
        transport.thinking_budget = int(arg)
        transport.thinking_level = None
        print(f"Thinking budget: {BOLD}{arg}{RESET} tokens")
    else:
        name = arg.upper()
        if valid_levels is not None and name not in valid_levels:
            print(f"{name} is not valid. Valid levels: {', '.join(valid_levels)}")
            return
        transport.thinking_level = name
        transport.thinking_budget = None
        print(f"Thinking level: {BOLD}{name}{RESET}")


# ── temperature ──


def _show_temperature(transport: Any) -> None:
    temp = getattr(transport, "temperature", None)
    print(f"Temperature: {BOLD}{temp if temp is not None else 'default'}{RESET}")


def _apply_temperature(transport: Any, arg: str) -> None:
    try:
        val = float(arg)
    except ValueError:
        print(f"Invalid temperature: {arg!r}")
        return
    if hasattr(transport, "temperature"):
        transport.temperature = val
        print(f"Temperature: {BOLD}{val}{RESET}")
    else:
        print("Transport does not support temperature")


# ── iterations ──


def _show_iterations(agent: Agent) -> None:
    print(f"Max iterations: {BOLD}{agent.max_iterations}{RESET}")


def _apply_iterations(agent: Agent, arg: str) -> None:
    try:
        val = int(arg)
    except ValueError:
        print(f"Invalid value: {arg!r}")
        return
    if val < 1:
        # Zero reads as "no limit" and means the opposite: the loop runs no
        # iterations at all, so the agent answers nothing and reports that it
        # ran out. There is no unlimited - a large number is how you say it.
        print(f"Max iterations must be at least 1. The default, {BOLD}1000{RESET}, is already out of the way.")
        return
    agent.max_iterations = val
    print(f"Max iterations: {BOLD}{val}{RESET}")


# ── max-tokens ──


def _show_max_tokens(transport: Any) -> None:
    cur = getattr(transport, "max_output_tokens", None)
    model_default = getattr(getattr(transport, "model", None), "max_output_tokens", None)
    if cur:
        print(f"Max output tokens: {BOLD}{cur}{RESET} (model default: {model_default})")
    else:
        print(f"Max output tokens: {BOLD}{model_default}{RESET} (model default)")


def _apply_max_tokens(transport: Any, arg: str) -> None:
    model_default = getattr(getattr(transport, "model", None), "max_output_tokens", None)
    if arg == "default":
        transport.max_output_tokens = None
        print(f"Max output tokens: {BOLD}{model_default}{RESET} (model default)")
        return
    try:
        val = int(arg)
    except ValueError:
        print(f"Invalid value: {arg!r}")
        return
    transport.max_output_tokens = val
    print(f"Max output tokens: {BOLD}{val}{RESET}")


# ── debug ──


def _show_debug(transport: Any) -> None:
    cur = getattr(transport, "debug", False)
    print(f"Debug: {BOLD}{'on' if cur else 'off'}{RESET}")


def _apply_debug(transport: Any, arg: str) -> None:
    val = arg.lower()
    if val == "on":
        transport.debug = True
        print(f"Debug: {BOLD}on{RESET} (request/response bodies logged to stderr)")
    elif val == "off":
        transport.debug = False
        print(f"Debug: {BOLD}off{RESET}")
    else:
        print("Usage: /debug on|off")


# ── Session journal ──


_JOURNAL_EVENT_KINDS: dict[type[object], str] = {
    AgentStarted: "agent_started",
    AgentStopped: "agent_stopped",
    TurnStarted: "turn_started",
    TurnFinished: "turn_finished",
    ForegroundEntered: "foreground_entered",
    ForegroundExited: "foreground_exited",
    OutcomeDelivered: "outcome_delivered",
    InputReceived: "input_received",
    ConfigurationChanged: "configuration_changed",
    MessageCommitted: "message_committed",
    ContextForked: "context_forked",
    ContextCleared: "context_cleared",
}

_DURABLE_JOURNAL_EVENTS = (
    MessageCommitted,
    ContextForked,
    ContextCleared,
    TurnFinished,
    OutcomeDelivered,
    AgentStopped,
)


async def _write_runtime_event(journal: _journal.SessionJournal, envelope: AgentEventEnvelope) -> None:
    event = envelope.event
    kind = _JOURNAL_EVENT_KINDS.get(type(event), "stream_event")
    payload: object
    if isinstance(event, MessageCommitted):
        payload = {
            "hub_seq": envelope.seq,
            "run_id": envelope.run_id,
            "message": event.message,
        }
    else:
        payload = {
            "hub_seq": envelope.seq,
            "run_id": envelope.run_id,
            "event": event,
        }
    accepted = await journal.publish(
        kind,
        payload,
        agent_id=envelope.agent_id,
        parent_agent_id=envelope.parent_agent_id,
        turn_id=envelope.turn_id,
        context_id=envelope.context_id,
        execution_mode=envelope.execution_mode.value,
        parent_tool_use_id=envelope.parent_tool_use_id,
    )
    if accepted and isinstance(event, _DURABLE_JOURNAL_EVENTS):
        await journal.sync()


@asynccontextmanager
async def _session_journal(
    event_hub: SessionEventHub,
    *,
    disabled: bool,
    root: Path | None,
    one_shot: bool,
    cwd: Path,
) -> AsyncIterator[_journal.SessionJournal | None]:
    if disabled:
        yield None
        return

    warning_emitted = False

    def warn_degraded(error: BaseException) -> None:
        nonlocal warning_emitted
        if warning_emitted:
            return
        warning_emitted = True
        print(
            f"Session journal degraded; subsequent records may be missing: {type(error).__name__}: {error}",
            file=sys.stderr,
        )

    try:
        journal = await _journal.SessionJournal.open(
            session_id=event_hub.session_id,
            root=root,
            start_payload={
                "application": AGENT_NAME,
                "version": AGENT_VERSION,
                "cwd": cwd,
                "mode": "one-shot" if one_shot else "interactive",
            },
            on_degraded=warn_degraded,
        )
    except OSError as error:
        warn_degraded(error)
        yield None
        return

    print(f"Session log: {journal.events_path}", file=sys.stderr if one_shot else sys.stdout)

    async def record(envelope: AgentEventEnvelope) -> None:
        await _write_runtime_event(journal, envelope)

    unsubscribe = event_hub.subscribe(record)
    end_payload: object = {"status": "complete"}
    try:
        yield journal
    except asyncio.CancelledError:
        end_payload = {"status": "cancelled"}
        raise
    except BaseException as error:
        end_payload = {"status": "error", "exception": error}
        raise
    finally:
        unsubscribe()
        await journal.close(end_payload)


# ── Main ─────────────────────────────────────────────────────────────


def _build_argument_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="REPL coding assistant (axio)")
    parser.add_argument("prompt", nargs="?", default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--transport", default=None, help="Transport name (auto-detected if omitted)")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--thinking", default=None, help="Thinking level or token budget (integer)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--debug", action="store_true", help="Log request/response bodies to stderr")
    parser.add_argument(
        "--agent-actions",
        choices=(DisplayMode.ACTIVE_ONLY.value, DisplayMode.ALL_ACTIONS.value),
        default=DisplayMode.ACTIVE_ONLY.value,
        help="Show framed actions from non-active agents (default: off)",
    )
    parser.add_argument("--no-session-log", action="store_true", help="Do not write the session JSONL journal")
    parser.add_argument(
        "--session-log-dir",
        type=Path,
        default=None,
        help="Root directory for session journals (default: XDG state directory)",
    )
    parser.add_argument(
        "--sandbox",
        choices=("auto", "docker", "none"),
        default="auto",
        help="Run file and shell tools inside a Docker container (default: auto — used when a daemon is reachable)",
    )
    parser.add_argument("--sandbox-image", default="python:3.12-slim", help="Image for --sandbox docker")
    return parser


async def main() -> None:
    args = _build_argument_parser().parse_args()

    setup_logging(args.debug)
    root = Path.cwd().resolve()
    event_hub = SessionEventHub()
    main_run_id = uuid4().hex
    journal_root = args.session_log_dir.expanduser().resolve() if args.session_log_dir is not None else None

    async with (
        _session_journal(
            event_hub,
            disabled=args.no_session_log,
            root=journal_root,
            one_shot=args.prompt is not None,
            cwd=root,
        ),
        aiohttp.ClientSession() as session,
        AsyncExitStack() as stack,
    ):
        transport_cls, _ = _select_transport(args.transport)
        agents_text = load_agents_instructions(root)
        transport = transport_cls(session=session)
        try:
            await transport.fetch_models()
        except StreamError as exc:
            print(f"Cannot reach {transport.name}: {exc}", file=sys.stderr)
            sys.exit(1)
        _adopt_catalogue_metadata(transport)

        if args.model:
            # The same resolution as the /model command: exact id, routing
            # variant, or a fragment of one. Naming a model on the command line
            # used to demand the full id, capitals and vendor prefix included.
            chosen = _choose_model(transport, args.model)
            if chosen is None:
                sys.exit(1)
            transport.model = chosen

        # Transport-level commands (available before agent creation).
        commands: dict[str, Command] = {
            "/thinking": Command(lambda: _show_thinking(transport), lambda a: _apply_thinking(transport, a)),
            "/temperature": Command(lambda: _show_temperature(transport), lambda a: _apply_temperature(transport, a)),
            "/max-tokens": Command(lambda: _show_max_tokens(transport), lambda a: _apply_max_tokens(transport, a)),
            "/debug": Command(lambda: _show_debug(transport), lambda a: _apply_debug(transport, a)),
        }
        _apply_cli_args(args, commands)

        tools, sandbox_desc, tool_root = await _sandbox.build_tools(
            stack, list(TOOLS), args.sandbox, args.sandbox_image, root
        )
        print(f"Tools: {BOLD}{sandbox_desc}{RESET}")
        system = build_system_prompt(tool_root, transport.model, tools, agents_text)
        agent = Agent(
            system=system,
            tools=tools,
            transport=transport,
            max_iterations=args.max_iterations,
            last_iteration_message=LAST_ITERATION_HINT,
        )
        ctx = ObservedContextStore(MemoryContextStore(), event_hub)
        set_session_event_hub(event_hub)
        parent_peer_id: str | None = None

        async def _publish_main_event(event: Any) -> None:
            await event_hub.publish(
                event,
                run_id=main_run_id,
                agent_id="main",
                parent_agent_id=None,
                turn_id=None,
                execution_mode=ExecutionMode.FOREGROUND,
                context_id=ctx.session_id,
            )

        await _publish_main_event(AgentStarted(name=AGENT_NAME, kind="repl-agent"))
        for config_name, config_value in (
            ("transport", getattr(transport, "name", type(transport).__name__)),
            ("model", transport.model.id),
            ("sandbox", sandbox_desc),
            ("max_iterations", agent.max_iterations),
            ("max_output_tokens", getattr(transport, "max_output_tokens", None)),
            ("temperature", getattr(transport, "temperature", None)),
            ("thinking_level", getattr(transport, "thinking_level", None)),
            ("thinking_budget", getattr(transport, "thinking_budget", None)),
            ("agent_actions", args.agent_actions),
        ):
            await _publish_main_event(ConfigurationChanged(name=config_name, value=config_value, source="startup"))

        def _command_configuration(command_name: str) -> object:
            match command_name:
                case "/model":
                    return transport.model.id
                case "/iterations":
                    return agent.max_iterations
                case "/thinking":
                    return {
                        "level": getattr(transport, "thinking_level", None),
                        "budget": getattr(transport, "thinking_budget", None),
                    }
                case "/temperature":
                    return getattr(transport, "temperature", None)
                case "/max-tokens":
                    return getattr(transport, "max_output_tokens", None)
                case "/debug":
                    return getattr(transport, "debug", False)
                case _:
                    return None

        async def _make_child_agent(
            inherit_context: bool,
            *,
            foreground: bool,
        ) -> tuple[Agent, ContextStore]:
            child_ctx = await ctx.fork() if inherit_context else ObservedContextStore(MemoryContextStore(), event_hub)
            child_transport = _clone_transport_for_spawn(agent.transport)
            child_tools = _clone_tools_for_child(agent.tools, foreground=foreground)
            child_system = build_system_prompt(
                tool_root,
                child_transport.model,
                child_tools,
                agents_text,
                parent_peer_id=None if foreground else parent_peer_id,
            )
            return agent.copy(
                transport=child_transport,
                system=child_system,
                tools=child_tools,
                max_iterations=agent.max_iterations,
                last_iteration_message=LAST_ITERATION_HINT,
            ), child_ctx

        async def _make_spawn_agent(inherit_context: bool) -> tuple[Agent, ContextStore]:
            return await _make_child_agent(inherit_context, foreground=False)

        async def _make_run_agent(inherit_context: bool) -> tuple[Agent, ContextStore]:
            return await _make_child_agent(inherit_context, foreground=True)

        set_spawn_agent_factory(_make_spawn_agent)
        set_run_agent_factory(_make_run_agent)

        # Agent-dependent commands.
        commands["/model"] = Command(
            lambda: _show_model(transport),
            lambda a: _apply_model(transport, agent, tools, tool_root, agents_text, a, parent_peer_id),
        )
        commands["/iterations"] = Command(
            lambda: _show_iterations(agent),
            lambda a: _apply_iterations(agent, a),
        )

        loop = asyncio.get_event_loop()
        peer_queue: asyncio.Queue[str] = asyncio.Queue()

        async def _queue_background_outcome(outcome: TurnOutcome) -> None:
            agent_id = outcome.identity.agent_id
            if outcome.succeeded and outcome.text.strip():
                text = f"Report from background agent {agent_id}:\n\n{outcome.text.strip()}"
            elif outcome.succeeded:
                text = f"[agent {agent_id}] finished its turn and is idle."
            else:
                text = f"[agent {agent_id}] turn failed: {outcome.error or 'unknown error'}"
            peer_queue.put_nowait(text)

        set_background_outcome_handler(_queue_background_outcome)

        stats = _panel.SessionStats()
        renderer = ReplRenderer(
            buffer_background_events=args.prompt is not None,
            stats=stats,
            current_model=lambda: transport.model,
            display_mode=DisplayMode.parse(args.agent_actions),
        )

        async def _render_envelope(envelope: AgentEventEnvelope) -> None:
            await render_runtime_event(renderer, envelope)

        unsubscribe_renderer = event_hub.subscribe(_render_envelope)
        prompt_session = _panel.make_session(
            lambda: _panel.status_line(transport.model, stats, renderer.action_status()),
            on_interrupt=lambda: _on_sigint(),
        )
        prompt_task: asyncio.Task[TurnOutcome] | None = None
        input_task: asyncio.Task[str] | None = None
        inbox_task: asyncio.Task[str] | None = None
        main_status = TurnStatus.SUCCEEDED
        # Lets monitor() see messages that arrived but have not been read:
        # they cannot be delivered until the current turn finishes.
        set_pending_message_probe(peer_queue.qsize)
        peer_server: PeerServer | None = None

        async def _on_peer_message(message: PeerMessage) -> None:
            await peer_queue.put(format_message_for_dialog(message))

        async def _run_turn(prompt: str, *, source: str) -> None:
            nonlocal main_status, prompt_task
            prompt_task = asyncio.create_task(run_prompt(agent, ctx, prompt, event_hub, main_run_id, source=source))
            try:
                outcome = await prompt_task
                main_status = outcome.status
            except asyncio.CancelledError:
                main_status = TurnStatus.CANCELLED
                # Keep the half-written answer. The agent stores an iteration
                # once the model stops talking, so an interrupted one is on
                # screen and nowhere else - and the next turn would be answered
                # by a model with no memory of saying it.
                partial = renderer.take_pending_text("main")
                if partial:
                    await ctx.append(Message(role="assistant", content=[TextBlock(text=partial)]))
                print(f"\n{DIM}[interrupted]{RESET}")
            finally:
                prompt_task = None
                await renderer.mark_idle()

        async def _interrupt_focused_agent(agent_id: str) -> None:
            await interrupt_agent(agent_id, reason="SIGINT")
            await renderer.notice("[interrupted]")
            await renderer.mark_idle()

        async def _run_one_shot_background_agents() -> None:
            records = local_background_agent_records()
            if not records:
                return
            if renderer.display_mode is DisplayMode.ALL_ACTIONS:
                await renderer.notice(f"[waiting for {len(records)} background agent(s)]")
                await wait_local_background_agents_idle([record.id for record in records])
                await renderer.mark_idle()
                return
            if len(records) == 1:
                renderer.set_focus(records[0].id)
                await renderer.notice(f"[following background agent {records[0].id}]")
            else:
                await renderer.notice(f"[waiting for {len(records)} background agents]")
                renderer.set_focus(records[0].id)
            await wait_local_background_agents_idle([record.id for record in records])
            for record in records[1:]:
                renderer.set_focus(record.id)
            await renderer.mark_idle()

        def _collect_queued(first: str) -> list[str]:
            """Everything waiting, not just the one that woke us.

            A turn per message means a turn per report when a swarm finishes
            together: three answers arriving at once cost three full prefills of
            the same context to deliver three paragraphs.
            """
            prompts = [first]
            while True:
                try:
                    prompts.append(peer_queue.get_nowait())
                except asyncio.QueueEmpty:
                    return prompts

        async def _run_peer_turn(first: str) -> None:
            prompts = _collect_queued(first)
            for prompt in prompts:
                await renderer.incoming(prompt)
            await _run_turn("\n\n".join(prompts), source="peer")

        async def _drain_peer_messages() -> None:
            try:
                first = peer_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await _run_peer_turn(first)

        def _on_sigint() -> None:
            nonlocal prompt_task
            if renderer.focused_agent != "main":
                asyncio.create_task(_interrupt_focused_agent(renderer.focused_agent))
            elif prompt_task is not None and not prompt_task.done():
                prompt_task.cancel()

        loop.add_signal_handler(signal.SIGINT, _on_sigint)

        try:
            try:
                peer_server = await PeerServer(
                    _peer_name(root),
                    kind="axio-repl",
                    handler=_on_peer_message,
                    cwd=str(root),
                ).start()
                parent_peer_id = peer_server.id
                agent.system = build_system_prompt(
                    tool_root,
                    transport.model,
                    tools,
                    agents_text,
                    parent_peer_id=parent_peer_id,
                )
                notify.add_listener(peer_server.id, peer_queue.put_nowait)
            except OSError as exc:
                print(f"{DIM}[peer messaging disabled: {exc}]{RESET}")
                notify.add_listener(None, peer_queue.put_nowait)

            if args.prompt:
                await _run_turn(args.prompt, source="one-shot")
                await _run_one_shot_background_agents()
                renderer.set_focus("main")
                await _drain_peer_messages()
                return

            agent_commands = ["/agents", "/agent-actions", "/agent-focus", "/agent-interrupt", "/agent-stop"]
            commands_list = ", ".join(["/help", *commands, *agent_commands, "/quit"])
            label = getattr(transport, "name", "unknown")
            print(f"REPL ready ({label}). Enter sends, Esc interrupts and sends, Up recalls.")
            print(f"Commands: {commands_list}")

            while True:
                if input_task is None:
                    input_task = asyncio.create_task(_read_input_async(prompt_session, renderer, _on_sigint))
                if inbox_task is None:
                    inbox_task = asyncio.create_task(peer_queue.get())

                done, _ = await asyncio.wait(
                    {input_task, inbox_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if inbox_task in done:
                    peer_prompt = inbox_task.result()
                    inbox_task = None
                    await _run_peer_turn(peer_prompt)
                    continue

                if input_task not in done:
                    continue

                try:
                    user_input = input_task.result()
                except EOFError:
                    print()
                    break
                finally:
                    input_task = None
                # Back before the turn runs, not after it. A turn is exactly
                # when the prompt is wanted: to read the status line, and to
                # type the next thing without waiting for the answer.
                input_task = asyncio.create_task(_read_input_async(prompt_session, renderer, _on_sigint))

                if not user_input:
                    continue
                lowered = user_input.lower()
                if lowered.startswith("/"):
                    await _publish_main_event(InputReceived(text=user_input, source="interactive-command"))
                if lowered in {"/quit", "/exit", "/q"}:
                    break
                if lowered == "/help":
                    tool_list = ", ".join(t.name for t in tools)
                    print(f"Type your request. Tools: {tool_list}")
                    print(f"Commands: {commands_list}")
                    continue
                if lowered == "/agents":
                    _show_agents(renderer)
                    continue
                if lowered == "/agent-actions" or lowered.startswith("/agent-actions "):
                    raw_mode = user_input[len("/agent-actions") :].strip()
                    await _handle_agent_actions(renderer, raw_mode, _publish_main_event)
                    continue
                if lowered == "/agent-focus" or lowered.startswith("/agent-focus "):
                    arg = user_input[len("/agent-focus") :].strip()
                    if not arg:
                        _show_agents(renderer)
                        continue
                    agent_id = _resolve_local_agent_id(arg)
                    if agent_id is None:
                        print(f"No local agent matching {arg!r}")
                        continue
                    renderer.set_focus(agent_id)
                    print(f"Focused agent: {BOLD}{agent_id}{RESET}")
                    await _publish_main_event(
                        ConfigurationChanged(name="input_target", value=agent_id, source="interactive")
                    )
                    continue
                if lowered == "/agent-stop" or lowered.startswith("/agent-stop "):
                    arg = user_input[len("/agent-stop") :].strip() or renderer.focused_agent
                    agent_id = _resolve_command_agent_id(arg, renderer)
                    if agent_id is None:
                        print(f"No local agent matching {arg!r}")
                        continue
                    if agent_id == "main":
                        print("Use /quit to exit the main REPL agent.")
                        continue
                    print(await stop_agent(agent_id, reason="user requested stop"))
                    if renderer.focused_agent == agent_id:
                        await renderer.mark_idle()
                        renderer.set_focus("main")
                        print(f"Focused agent: {BOLD}main{RESET}")
                    continue
                if lowered == "/agent-interrupt" or lowered.startswith("/agent-interrupt "):
                    arg = user_input[len("/agent-interrupt") :].strip() or renderer.focused_agent
                    agent_id = _resolve_command_agent_id(arg, renderer)
                    if agent_id is None:
                        print(f"No local agent matching {arg!r}")
                        continue
                    if agent_id == "main":
                        print("Press Ctrl-C while the main agent is streaming to interrupt it.")
                        continue
                    print(await interrupt_agent(agent_id, reason="user requested interrupt"))
                    if renderer.focused_agent == agent_id:
                        await renderer.mark_idle()
                    continue

                matched = False
                for prefix, cmd in commands.items():
                    if lowered == prefix or lowered.startswith(prefix + " "):
                        cmd_arg = user_input[len(prefix) :].strip() or None
                        if cmd_arg is None:
                            cmd.show()
                        else:
                            previous_value = _command_configuration(prefix)
                            cmd.apply(cmd_arg)
                            current_value = _command_configuration(prefix)
                            if current_value != previous_value:
                                await _publish_main_event(
                                    ConfigurationChanged(
                                        name=prefix.removeprefix("/"),
                                        value=current_value,
                                        source="interactive",
                                    )
                                )
                        matched = True
                        break
                if matched:
                    continue

                if renderer.focused_agent == "main":
                    await _run_turn(user_input, source="interactive")
                elif is_local_background_agent(renderer.focused_agent):
                    delivered = await enqueue_local_agent_prompt(renderer.focused_agent, user_input, wait=True)
                    await renderer.mark_idle()
                    if not delivered:
                        print(f"Agent {renderer.focused_agent!r} is no longer running.")
                        renderer.set_focus("main")
                else:
                    print(f"Agent {renderer.focused_agent!r} is no longer local; focusing main.")
                    renderer.set_focus("main")
        except asyncio.CancelledError:
            main_status = TurnStatus.CANCELLED
            raise
        except BaseException:
            main_status = TurnStatus.FAILED
            raise
        finally:
            for task in (input_task, inbox_task):
                if task is not None and not task.done():
                    task.cancel()
            await stop_local_background_agents()
            await ctx.close()
            await _publish_main_event(AgentStopped(status=main_status))
            notify.remove_listener(peer_server.id if peer_server is not None else None)
            if peer_server is not None:
                await peer_server.close()
            unsubscribe_renderer()
            set_background_outcome_handler(None)
            set_run_agent_factory(None)
            set_spawn_agent_factory(None)
            set_session_event_hub(None)
            set_pending_message_probe(None)
            loop.remove_signal_handler(signal.SIGINT)


def main_sync() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main_sync()
