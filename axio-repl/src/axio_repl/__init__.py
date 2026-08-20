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
import inspect
import logging
import os
import shutil
import signal
import sys
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from datetime import datetime
from importlib.metadata import entry_points
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple, cast
from uuid import uuid4

import aiohttp
from axio import notify
from axio._asyncio import cancel_task_once
from axio.agent import Agent
from axio.blocks import TextBlock
from axio.context import ContextStore, MemoryContextStore
from axio.effort import EFFORT_LEVELS, EffortRuntime, EffortState
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
from axio.messages import INPUT_PROVENANCE_SYSTEM_INSTRUCTION, InputProvenance, Message
from axio.models import Capability, ModelSpec
from axio.tool import Tool
from axio.tool_args import ToolArgStream
from axio_tools_agents.monitoring import monitor
from axio_tools_agents.peers import (
    PeerMessage,
    PeerServer,
    background_agent_state,
    enqueue_local_agent_context,
    enqueue_local_agent_messages,
    format_message_for_dialog,
    interrupt_agent,
    interrupt_local_agent_turn,
    is_local_background_agent,
    list_peers,
    local_background_agent_records,
    run_agent,
    send_message,
    set_background_input_admitted_handler,
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
    EditorSnapshot,
    ExecutionMode,
    ForegroundEntered,
    ForegroundExited,
    InputBuffered,
    InputClaimed,
    InputDelivered,
    InputRecalled,
    InputReceived,
    InterruptionCommitted,
    InterruptionRequested,
    MessageCommitted,
    ObservedContextStore,
    OutcomeDelivered,
    RecoveryApplied,
    RuntimeEvent,
    SessionEventHub,
    ShutdownRecorded,
    TurnFinished,
    TurnIdentity,
    TurnOutcome,
    TurnStarted,
    TurnStatus,
    new_turn_identity,
    observe_agent_turn,
    observe_agent_turn_messages,
)
from axio_tools_local.list_files import list_files
from axio_tools_local.patch_file import patch_file
from axio_tools_local.read_file import read_file
from axio_tools_local.shell import shell
from axio_tools_local.write_file import write_file

from axio_repl import _agent_config, _journal, _panel, _sandbox, _search, _version
from axio_repl._coordinator import (
    ClaimBatch,
    ContextArrival,
    ForegroundCoordinatorState,
    PendingInputCoordinator,
    PendingInputStatus,
    claim_batch_arrivals,
    ordered_arrivals,
)
from axio_repl._deferred_tools import DeferredToolNotification, DeferredToolRegistry
from axio_repl._identity import append_runtime_identity_metadata, resolve_effective_username
from axio_repl._input import ExitArmingState, InputSubmitted, SubmissionDisposition
from axio_repl._input import InterruptRequested as PromptInterruptRequested
from axio_repl._multiplexer import (
    ActionMultiplexer,
    DisplayMode,
    DisplayModeChange,
    format_agent_identity,
    normalize_agent_name,
    sanitize_identity_component,
)
from axio_repl._powerline import agent_header, tool_title
from axio_repl._recovery import RecoveryError, RecoveryMaterialization, materialize_recovery
from axio_repl._terminal import TerminalUI
from axio_repl._terminal_sanitizer import IncrementalTerminalSanitizer, sanitize_terminal_text
from axio_repl._theme import (
    DEFAULT_THEME,
    TerminalTheme,
    resolve_terminal_presentation,
    resolve_theme,
    theme_names,
)
from axio_repl._theme import RESET as RESET

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
MAX_PENDING_INPUTS = 32
AGENT_VERSION = "0.2.3"

# ── ANSI helpers ─────────────────────────────────────────────────────

DIM = DEFAULT_THEME.reasoning.ansi
BOLD = DEFAULT_THEME.command.ansi
GREEN = DEFAULT_THEME.success.ansi
YELLOW = DEFAULT_THEME.warning.ansi
RED = DEFAULT_THEME.error.ansi
MUTED_AMBER = DEFAULT_THEME.stderr.ansi


def _styled(style: str, text: str) -> str:
    """Apply a style without relying on terminal state across line redraws."""
    if not text:
        return ""
    if not style:
        return text
    line_break = f"{RESET}\n{style}"
    return style + text.replace("\n", line_break) + RESET


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


def _select_transport(name: str | None, credential_override: bool = False) -> tuple[Callable[..., Any], str]:
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
        if env_vars and not credential_override and not _transport_has_credentials(name):
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


def _select_configured_tools(
    selection: str | tuple[str, ...] | None,
    available_tools: list[Tool[Any]] | None = None,
) -> list[Tool[Any]]:
    tools = TOOLS if available_tools is None else available_tools
    if selection is None:
        return list(tools)
    names = tuple(part.strip() for part in selection.split(",")) if isinstance(selection, str) else selection
    if any(not name for name in names):
        raise ValueError("--tools must be 'all', 'none', or a comma-separated list of tool names")
    if names == ("all",):
        return list(tools)
    if names == ("none",) or not names:
        return []
    if "all" in names or "none" in names:
        raise ValueError("'all' and 'none' cannot be combined with named tools")
    if len(set(names)) != len(names):
        raise ValueError("tool selection must not contain duplicate names")
    available = {tool.name: tool for tool in tools}
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"unknown tool(s): {', '.join(unknown)}")
    requested = set(names)
    return [tool for tool in tools if tool.name in requested]


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
    sandbox_note: str = "",
    model_context: str = "",
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
    lines += ["Input provenance:", f"- {INPUT_PROVENANCE_SYSTEM_INSTRUCTION}", ""]
    if sandbox_note:
        lines += [sandbox_note, ""]
    if model_context:
        lines += [
            "Operator model context (trusted local profile data; descriptive, not an enforcement mechanism):",
            model_context,
            "",
        ]

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
        lines += [agents_text, ""]

    return "\n".join(lines)


def _build_runtime_system_prompt(
    root: Path,
    model: ModelSpec,
    tools: list[Tool[Any]],
    agents_text: str,
    *,
    effective_username: str | None = None,
    effort: EffortRuntime | None = None,
    parent_peer_id: str | None = None,
    sandbox_note: str = "",
    model_context: str = "",
) -> str:
    """Compose the stable provider system prompt and append runtime identity once."""

    system = build_system_prompt(
        root,
        model,
        tools,
        agents_text,
        parent_peer_id=parent_peer_id,
        sandbox_note=sandbox_note,
        model_context=model_context,
    )
    if effort is not None:
        system = effort.system_prompt(system)
    if effective_username is not None:
        system = append_runtime_identity_metadata(system, effective_username)
    return system


# ── Readline history ─────────────────────────────────────────────────


# ── Event rendering ──────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class _BoundaryMode:
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class _TextMode:
    paragraph_newline_pending: bool = False
    paragraph_boundary_open: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class _ReasoningMode:
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class _ToolFieldMode:
    tool_use_id: str
    key: str
    first_delta: bool = True
    chunk_line_closed: bool = False


type _StructuralMode = _BoundaryMode | _TextMode | _ReasoningMode | _ToolFieldMode


class _AgentRenderState:
    def __init__(self) -> None:
        self.mode: _StructuralMode = _BoundaryMode()
        self.arg_streams: dict[str, ToolArgStream] = {}
        self.active_tool_ids: set[str] = set()
        self.streamed_tool_ids: set[str] = set()
        self.tool_names: dict[str, str] = {}
        self.background_text: list[str] = []
        self.pending_text: list[str] = []
        self.background_tools: list[str] = []
        self.background_errors: list[str] = []
        self.text_sanitizer = IncrementalTerminalSanitizer()
        self.reasoning_sanitizer = IncrementalTerminalSanitizer()
        self.tool_output_sanitizers: dict[tuple[str, str], IncrementalTerminalSanitizer] = {}
        self.tool_field_sanitizers: dict[tuple[str, str], IncrementalTerminalSanitizer] = {}
        self.tool_arg_at_line_start = False


@dataclasses.dataclass(frozen=True, slots=True)
class _TurnKey:
    agent_id: str
    run_id: str
    turn_id: str


@dataclasses.dataclass(slots=True)
class _TurnPresentation:
    key: _TurnKey
    execution_mode: ExecutionMode
    live: bool
    agent_name: str | None
    header_emitted: bool = False
    stdout_started: bool = False
    error_seen: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class _BackgroundSummary:
    identity: str
    reported_chars: int
    tools: tuple[str, ...]
    failed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _CompletedBackgroundTurn:
    agent_name: str | None
    suppress_display: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _ParentToolKey:
    parent_agent_id: str
    parent_run_id: str
    parent_turn_id: str
    tool_use_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class _ForegroundCall:
    parent_key: _ParentToolKey
    child_agent_id: str
    child_agent_name: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class _ForegroundResult:
    child_agent_id: str
    child_agent_name: str | None
    status: TurnStatus


@dataclasses.dataclass(frozen=True, slots=True)
class _IncomingPrompt:
    text: str
    arrival_seq: int | None = None
    source: str | None = None
    author: str | None = None
    display_text: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    suppress_display: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class _InterruptTransaction:
    request_seq: int
    request: PromptInterruptRequested
    claimed: ClaimBatch | None


@dataclasses.dataclass(frozen=True, slots=True)
class _AcceptedInterrupt:
    request_seq: int
    request: PromptInterruptRequested


@dataclasses.dataclass(frozen=True, slots=True)
class _ReadyClaim:
    batch: ClaimBatch
    messages: tuple[Message, ...]
    source_input_ids: tuple[str | None, ...]
    source: str


def _pending_prompt_count(
    peer_queue: asyncio.Queue[_IncomingPrompt],
    buffered_prompts: deque[_IncomingPrompt],
    inbox_task: asyncio.Task[_IncomingPrompt] | None,
) -> int:
    claimed_by_inbox = int(inbox_task is not None and inbox_task.done() and not inbox_task.cancelled())
    return peer_queue.qsize() + len(buffered_prompts) + claimed_by_inbox


def _retain_interrupted_partial(
    partials: dict[str, str],
    pending_interrupt_turns: set[str],
    *,
    turn_id: str,
    partial: str,
    preemption_reason: str | None,
) -> None:
    """Retain only partials that shutdown or an accepted interrupt must consume."""

    if partial and (preemption_reason is None or turn_id in pending_interrupt_turns):
        partials[turn_id] = partial


async def _cancel_and_settle_tasks(*tasks: asyncio.Task[Any] | None) -> tuple[Any, ...]:
    """Cancel unfinished coordinator tasks and retrieve every task outcome."""
    managed_tasks = tuple(task for task in tasks if task is not None)
    for task in managed_tasks:
        cancel_task_once(task)
    if not managed_tasks:
        return ()
    return tuple(await asyncio.gather(*managed_tasks, return_exceptions=True))


def _peer_incoming_prompt(message: PeerMessage) -> _IncomingPrompt:
    identity = format_agent_identity(message.from_id, message.from_name)
    return _IncomingPrompt(
        text=format_message_for_dialog(message),
        display_text=f"Peer message from {identity}:\n\n{message.body}",
        agent_id=message.from_id,
        agent_name=normalize_agent_name(message.from_name),
    )


class ReplRenderer:
    def __init__(
        self,
        *,
        main_agent_name: str | None = None,
        stats: _panel.SessionStats | None = None,
        current_model: Callable[[], ModelSpec | None] | None = None,
        display_mode: DisplayMode = DisplayMode.ACTIVE_ONLY,
        action_multiplexer: ActionMultiplexer | None = None,
        suspended_action_multiplexer: ActionMultiplexer | None = None,
        action_boundary_frames: int = 4,
        action_boundary_bytes: int = 16 * 1024,
        max_identity_cache: int = 256,
        powerline: bool = False,
        theme: TerminalTheme = DEFAULT_THEME,
        effective_username: str = "unknown",
    ) -> None:
        if max_identity_cache < 2:
            raise ValueError("max_identity_cache must be at least 2")
        self._lock = asyncio.Lock()
        # Every agent's events pass through here, which makes this the one place
        # that sees the whole session's spend.
        self._stats = stats
        self._current_model = current_model
        self._states: dict[str, _AgentRenderState] = {"main": _AgentRenderState()}
        self._agent_names: OrderedDict[str, str] = OrderedDict()
        self._max_identity_cache = max_identity_cache
        self._powerline = powerline
        self._theme = theme
        self._effective_username = sanitize_identity_component(effective_username) or "unknown"
        if normalized_main_name := normalize_agent_name(main_agent_name):
            self._agent_names["main"] = normalized_main_name
        self._turns: dict[_TurnKey, _TurnPresentation] = {}
        self._current_turn_by_agent: dict[str, _TurnKey] = {}
        self._completed_background_turns: OrderedDict[_TurnKey, _CompletedBackgroundTurn] = OrderedDict()
        self._active_agent: str | None = None
        self._focused_agent = "main"
        self._foreground_stack: list[str] = []
        self._foreground_parent_calls: dict[str, _ForegroundCall] = {}
        self._streamed_foreground_calls: dict[_ParentToolKey, _ForegroundResult] = {}
        self._foreground_streaming = False
        self._background_summaries: OrderedDict[_TurnKey, _BackgroundSummary] = OrderedDict()
        self._input_active = False
        self._panel_message = ""
        self._actions = action_multiplexer or ActionMultiplexer(display_mode, powerline=powerline, theme=theme)
        # A parent's sibling tool still belongs to the active turn while a child
        # owns the terminal, so it has an always-on queue separate from background actions.
        self._suspended_actions = suspended_action_multiplexer or ActionMultiplexer(
            DisplayMode.ALL_ACTIONS,
            powerline=powerline,
            theme=theme,
        )
        self._suspended_tool_calls: set[tuple[str, str]] = set()
        self._action_boundary_frames = action_boundary_frames
        self._action_boundary_bytes = action_boundary_bytes

    @property
    def focused_agent(self) -> str:
        return self._focused_agent

    @property
    def focused_turn_id(self) -> str | None:
        key = self._current_turn_by_agent.get(self._focused_agent)
        return key.turn_id if key is not None else None

    @property
    def foreground_agent(self) -> str:
        return self._foreground_stack[-1] if self._foreground_stack else self._focused_agent

    @property
    def display_mode(self) -> DisplayMode:
        return self._actions.mode

    @property
    def queued_action_count(self) -> int:
        return self._actions.queued_count + self._suspended_actions.queued_count

    @property
    def retained_action_bytes(self) -> int:
        return self._actions.retained_bytes + self._suspended_actions.retained_bytes

    @property
    def max_retained_action_bytes(self) -> int:
        return self._actions.max_retained_bytes + self._suspended_actions.max_retained_bytes

    @property
    def retained_agent_state_count(self) -> int:
        return len(self._states)

    @property
    def retained_identity_count(self) -> int:
        return len(self._agent_names)

    def action_status(self) -> str:
        status = f"actions: {self.display_mode.value}"
        if self.queued_action_count:
            status += f" ({self.queued_action_count} queued)"
        return status

    def agent_status(self) -> str:
        agent_id = self._status_agent_id()
        turn = self._current_turn_by_agent.get(agent_id)
        if turn is None:
            phase = "idle" if agent_id == "main" else background_agent_state(agent_id)[0]
        else:
            state = self._state(agent_id)
            if state.active_tool_ids:
                phase = f"tools: {self._active_tool_summary(state)}"
            elif isinstance(state.mode, _ReasoningMode):
                phase = "reasoning"
            elif isinstance(state.mode, _TextMode):
                phase = "responding"
            else:
                phase = "waiting for model" if agent_id == "main" else "running"
        identity = self._panel_identity(agent_id)
        status = f"{identity}: {phase}"
        if agent_id != self._focused_agent:
            status += f"; focus {self._panel_identity(self._focused_agent)}"
        return status

    @property
    def panel_message(self) -> str:
        return self._panel_message

    def show_panel(self, text: str) -> None:
        self._panel_message = sanitize_terminal_text(text).strip()

    def clear_panel(self) -> None:
        self._panel_message = ""

    async def set_display_mode(self, mode: DisplayMode) -> DisplayModeChange:
        async with self._lock:
            return self._actions.set_mode(mode)

    async def remember_agent(self, agent_id: str, agent_name: str) -> None:
        async with self._lock:
            self._remember_agent_locked(agent_id, agent_name)

    def agent_identity(self, agent_id: str, agent_name: str | None = None) -> str:
        return format_agent_identity(agent_id, agent_name if agent_name is not None else self._agent_name(agent_id))

    async def start_turn(
        self,
        agent_id: str,
        event: TurnStarted,
        *,
        run_id: str,
        turn_id: str | None,
        execution_mode: ExecutionMode,
    ) -> None:
        async with self._lock:
            key = _TurnKey(agent_id, run_id, turn_id or "")
            previous = self._current_turn_by_agent.get(agent_id)
            if previous is not None and previous != key:
                self._turns.pop(previous, None)
            agent_name = self._agent_name(agent_id)
            live = execution_mode is ExecutionMode.FOREGROUND or agent_id == self.foreground_agent
            presentation = _TurnPresentation(
                key=key,
                execution_mode=execution_mode,
                live=live,
                agent_name=agent_name,
            )
            self._turns[key] = presentation
            self._current_turn_by_agent[agent_id] = key
            state = self._state(agent_id)
            self._reset_state_for_turn_locked(state)
            if live:
                self._foreground_streaming = True
                self._actions.discard_agent(agent_id)
                return
            if execution_mode is ExecutionMode.BACKGROUND:
                self._actions.observe(agent_id, event, agent_name=agent_name)
                if self.display_mode is DisplayMode.ALL_ACTIONS and self._safe_boundary_is_open_locked():
                    self._drain_safe_boundary_locked(max_frames=1)

    async def finish_turn(
        self,
        agent_id: str,
        event: TurnFinished,
        *,
        run_id: str,
        turn_id: str | None,
        execution_mode: ExecutionMode,
    ) -> None:
        async with self._lock:
            key = _TurnKey(agent_id, run_id, turn_id or "")
            presentation = self._turns.pop(key, None)
            if self._current_turn_by_agent.get(agent_id) == key:
                self._current_turn_by_agent.pop(agent_id, None)
            if presentation is None:
                presentation = _TurnPresentation(
                    key=key,
                    execution_mode=execution_mode,
                    live=execution_mode is ExecutionMode.FOREGROUND or agent_id == self.foreground_agent,
                    agent_name=self._agent_name(agent_id),
                )
            if execution_mode is ExecutionMode.BACKGROUND:
                if presentation.live:
                    completed = _CompletedBackgroundTurn(presentation.agent_name, suppress_display=True)
                else:
                    self._actions.observe(agent_id, event, agent_name=presentation.agent_name)
                    summary = self._background_summary_locked(agent_id, presentation, event)
                    self._background_summaries[key] = summary
                    while len(self._background_summaries) > 1024:
                        self._background_summaries.popitem(last=False)
                    completed = _CompletedBackgroundTurn(presentation.agent_name, suppress_display=False)
                self._completed_background_turns[key] = completed
                while len(self._completed_background_turns) > 1024:
                    self._completed_background_turns.popitem(last=False)
                if self.display_mode is DisplayMode.ALL_ACTIONS and self._safe_boundary_is_open_locked():
                    self._drain_safe_boundary_locked(max_frames=1)
                elif self.display_mode is DisplayMode.ACTIVE_ONLY and not self._foreground_streaming:
                    self._flush_background_summaries_locked()
            self._purge_parent_turn_locked(key)
            self._cleanup_turn_state_locked(agent_id)

    async def take_background_outcome_presentation(
        self,
        agent_id: str,
        run_id: str,
        turn_id: str,
    ) -> tuple[str | None, bool]:
        async with self._lock:
            completed = self._completed_background_turns.pop(_TurnKey(agent_id, run_id, turn_id), None)
            if completed is None:
                return self._agent_name(agent_id), False
            return completed.agent_name, completed.suppress_display

    async def enter_foreground(
        self,
        agent_id: str,
        parent_tool_use_id: str | None = None,
        parent_agent_id: str | None = None,
    ) -> None:
        async with self._lock:
            owner = parent_agent_id or self.foreground_agent
            owner_state = self._state(owner)
            for tool_use_id in owner_state.active_tool_ids:
                if tool_use_id == parent_tool_use_id:
                    continue
                name = owner_state.tool_names.get(tool_use_id, "tool")
                self._suspended_tool_calls.add((owner, tool_use_id))
                self._suspended_actions.adopt_tool(
                    owner,
                    tool_use_id,
                    name,
                    agent_name=self._agent_name(owner),
                )
            self._foreground_stack.append(agent_id)
            self._actions.discard_agent(agent_id)
            if parent_tool_use_id is not None:
                parent_turn = self._current_turn_by_agent.get(owner)
                parent_key = _ParentToolKey(
                    parent_agent_id=owner,
                    parent_run_id=parent_turn.run_id if parent_turn is not None else "",
                    parent_turn_id=parent_turn.turn_id if parent_turn is not None else "",
                    tool_use_id=parent_tool_use_id,
                )
                self._foreground_parent_calls[agent_id] = _ForegroundCall(
                    parent_key=parent_key,
                    child_agent_id=agent_id,
                    child_agent_name=self._agent_name(agent_id),
                )

    async def exit_foreground(self, agent_id: str, status: TurnStatus) -> None:
        async with self._lock:
            if self._foreground_stack and self._foreground_stack[-1] == agent_id:
                self._foreground_stack.pop()
            else:
                with suppress(ValueError):
                    self._foreground_stack.remove(agent_id)
            call = self._foreground_parent_calls.pop(agent_id, None)
            if call is not None:
                current_parent_turn = self._current_turn_by_agent.get(call.parent_key.parent_agent_id)
                parent_still_matches = not call.parent_key.parent_turn_id or (
                    current_parent_turn is not None
                    and current_parent_turn.run_id == call.parent_key.parent_run_id
                    and current_parent_turn.turn_id == call.parent_key.parent_turn_id
                )
                if parent_still_matches:
                    self._streamed_foreground_calls[call.parent_key] = _ForegroundResult(
                        child_agent_id=call.child_agent_id,
                        child_agent_name=call.child_agent_name,
                        status=status,
                    )
            resumed = self._state(self.foreground_agent)
            self._foreground_streaming = bool(
                not isinstance(resumed.mode, _BoundaryMode) or resumed.active_tool_ids or resumed.arg_streams
            )
            self._drain_safe_boundary_locked()

    def set_focus(self, agent_id: str) -> None:
        self._focused_agent = agent_id
        presentation = self._current_presentation(agent_id)
        if presentation is not None and presentation.agent_name is not None:
            self._remember_agent_locked(agent_id, presentation.agent_name)
        self._actions.discard_agent(agent_id)

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

    async def render(
        self,
        agent_id: str,
        event: StreamEvent,
        *,
        run_id: str | None = None,
        turn_id: str | None = None,
        execution_mode: ExecutionMode | None = None,
    ) -> None:
        async with self._lock:
            presentation = self._presentation(agent_id, run_id, turn_id)
            live = (
                presentation.live
                if presentation is not None
                else execution_mode is ExecutionMode.FOREGROUND or agent_id == self.foreground_agent
            )
            if isinstance(event, IterationEnd) and self._stats is not None:
                model = self._current_model() if self._current_model is not None else None
                self._stats.record(agent_id, event.usage, model)
            if self._is_suspended_tool_event(agent_id, event):
                self._observe_suspended_tool_event_locked(agent_id, event)
                if self._safe_boundary_is_open_locked():
                    self._drain_safe_boundary_locked()
            elif live:
                self._render_locked(agent_id, event, presentation)
                if isinstance(event, Error | SessionEndEvent):
                    self._foreground_streaming = False
                    if self.display_mode is DisplayMode.ACTIVE_ONLY:
                        self._flush_background_summaries_locked()
                elif not isinstance(event, IterationEnd):
                    self._foreground_streaming = True
            else:
                agent_name = presentation.agent_name if presentation is not None else self._agent_name(agent_id)
                self._actions.observe(agent_id, event, agent_name=agent_name)
                self._record_background_event_locked(agent_id, event)
                if self.display_mode is DisplayMode.ALL_ACTIONS and self._safe_boundary_is_open_locked():
                    self._drain_safe_boundary_locked(max_frames=1)
                elif self.display_mode is DisplayMode.ACTIVE_ONLY and not self._foreground_streaming:
                    self._flush_background_summaries_locked()

    async def observe_runtime_event(self, agent_id: str, event: RuntimeEvent) -> None:
        async with self._lock:
            if agent_id != self.foreground_agent:
                self._actions.observe(agent_id, event, agent_name=self._agent_name(agent_id))
                if self.display_mode is DisplayMode.ALL_ACTIONS and self._safe_boundary_is_open_locked():
                    self._drain_safe_boundary_locked(max_frames=1)

    async def stop_agent(self, agent_id: str, event: AgentStopped, *, background: bool) -> None:
        async with self._lock:
            if background and agent_id != self.foreground_agent:
                self._actions.observe(agent_id, event, agent_name=self._agent_name(agent_id))
                if self.display_mode is DisplayMode.ALL_ACTIONS and self._safe_boundary_is_open_locked():
                    self._drain_safe_boundary_locked(max_frames=1)
            self._cleanup_agent_locked(agent_id)

    async def mark_idle(self) -> None:
        async with self._lock:
            self._foreground_streaming = False
            self._drain_all_actions_locked()
            if self.display_mode is DisplayMode.ALL_ACTIONS:
                self._discard_background_summaries_locked()
            else:
                self._flush_background_summaries_locked()

    async def submitted(self, text: str, submitted_at: datetime) -> None:
        """Persist one accepted editor value after coordinator admission."""

        async with self._lock:
            with self._persistent_insertion_locked():
                print(
                    _panel.submitted_message(
                        text,
                        self._effective_username,
                        submitted_at,
                        powerline=self._powerline,
                        theme=self._theme,
                    )
                )

    async def incoming(
        self,
        text: str,
        *,
        agent_id: str | None = None,
        agent_name: str | None = None,
        suppress_display: bool = False,
    ) -> None:
        """Put an arriving message on screen, not only into the model's prompt.

        A report from a spawned agent was fed to the model and never shown, so
        the only account of it the user ever saw was the model's summary of
        something they could not read.
        """
        async with self._lock:
            if suppress_display:
                return
            text = sanitize_terminal_text(text)
            with self._persistent_insertion_locked():
                if self._powerline and agent_id is not None:
                    identity = format_agent_identity(agent_id, agent_name or self._agent_name(agent_id))
                    print(f"\n{agent_header(identity, self._theme)}\n{text}\n")
                    self._drain_safe_boundary_locked()
                    return
                source = (
                    f" from agent {format_agent_identity(agent_id, agent_name or self._agent_name(agent_id))}"
                    if agent_id is not None
                    else ""
                )
                print(f"\n{self._theme.agent.ansi}{'─' * 3} incoming{source} {'─' * 3}{self._theme.reset}\n{text}\n")
                self._drain_safe_boundary_locked()

    async def notice(self, text: str) -> None:
        async with self._lock:
            self.show_panel(text)

    def _state(self, agent_id: str) -> _AgentRenderState:
        return self._states.setdefault(agent_id, _AgentRenderState())

    @contextmanager
    def _persistent_insertion_locked(self) -> Iterator[None]:
        """Bracket an out-of-band scrollback line without tearing stream structure."""

        continuation: tuple[_AgentRenderState, _ToolFieldMode] | None = None
        if self._active_agent is not None:
            state = self._state(self._active_agent)
            mode = state.mode
            if isinstance(mode, _TextMode) and mode.paragraph_boundary_open:
                sys.stdout.write(self._theme.reset)
                state.mode = _BoundaryMode()
            elif isinstance(mode, _ToolFieldMode) and mode.chunk_line_closed:
                state.mode = _BoundaryMode()
            elif not isinstance(mode, _BoundaryMode):
                sys.stdout.write(f"{self._theme.reset}\n")
                self._flush()
                state.mode = _BoundaryMode()
            if isinstance(mode, _ToolFieldMode):
                continuation = (state, mode)
        yield
        if continuation is not None:
            state, mode = continuation
            if self._input_active:
                state.tool_arg_at_line_start = True
                state.mode = dataclasses.replace(mode, chunk_line_closed=not mode.first_delta)
            else:
                sys.stdout.write(f"  {self._theme.warning.ansi}{mode.key} (continued){self._theme.reset}: ")
                state.tool_arg_at_line_start = False
                state.mode = dataclasses.replace(mode, first_delta=True, chunk_line_closed=False)
                self._flush()

    def _reset_state_for_turn_locked(self, state: _AgentRenderState) -> None:
        state.mode = _BoundaryMode()
        state.arg_streams.clear()
        state.active_tool_ids.clear()
        state.streamed_tool_ids.clear()
        state.tool_names.clear()
        state.background_text.clear()
        state.pending_text.clear()
        state.background_tools.clear()
        state.background_errors.clear()
        state.text_sanitizer.reset()
        state.reasoning_sanitizer.reset()
        state.tool_output_sanitizers.clear()
        state.tool_field_sanitizers.clear()
        state.tool_arg_at_line_start = False

    def _remember_agent_locked(self, agent_id: str, agent_name: str) -> None:
        normalized = normalize_agent_name(agent_name)
        if normalized is None:
            return
        self._agent_names.pop(agent_id, None)
        self._agent_names[agent_id] = normalized
        protected = {"main", self._focused_agent}
        while len(self._agent_names) > self._max_identity_cache:
            candidate = next((known_id for known_id in self._agent_names if known_id not in protected), None)
            if candidate is None:
                break
            self._agent_names.pop(candidate, None)

    def _agent_name(self, agent_id: str) -> str | None:
        presentation = self._current_presentation(agent_id)
        if presentation is not None and presentation.agent_name is not None:
            return presentation.agent_name
        foreground_call = self._foreground_parent_calls.get(agent_id)
        if foreground_call is not None and foreground_call.child_agent_name is not None:
            return foreground_call.child_agent_name
        return self._agent_names.get(agent_id)

    def _status_agent_id(self) -> str:
        for agent_id in reversed(self._foreground_stack):
            if agent_id in self._current_turn_by_agent:
                return agent_id
        if "main" in self._current_turn_by_agent:
            return "main"
        if self._focused_agent in self._current_turn_by_agent:
            return self._focused_agent
        return self._focused_agent

    def _panel_identity(self, agent_id: str) -> str:
        return self._agent_name(agent_id) or agent_id

    @staticmethod
    def _active_tool_summary(state: _AgentRenderState) -> str:
        names = [name for tool_use_id, name in state.tool_names.items() if tool_use_id in state.active_tool_ids]
        counts: OrderedDict[str, int] = OrderedDict()
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        return ", ".join(f"{name} ×{count}" if count > 1 else name for name, count in counts.items())

    def _presentation(
        self,
        agent_id: str,
        run_id: str | None,
        turn_id: str | None,
    ) -> _TurnPresentation | None:
        if run_id is not None and turn_id is not None:
            return self._turns.get(_TurnKey(agent_id, run_id, turn_id))
        return self._current_presentation(agent_id)

    def _current_presentation(self, agent_id: str) -> _TurnPresentation | None:
        key = self._current_turn_by_agent.get(agent_id)
        return self._turns.get(key) if key is not None else None

    def _event_starts_stdout(
        self,
        event: StreamEvent,
        presentation: _TurnPresentation | None,
    ) -> bool:
        if presentation is None or presentation.stdout_started:
            return False
        match event:
            case TextDelta(delta=delta) | ToolOutputDelta(delta=delta):
                return bool(delta)
            case ReasoningDelta(delta=delta):
                return bool(delta.lstrip("\n"))
            case IterationEnd() | Error():
                return False
            case SessionEndEvent():
                return not presentation.error_seen
            case _:
                return True

    def _ensure_stdout_header_locked(
        self,
        agent_id: str,
        presentation: _TurnPresentation | None,
    ) -> None:
        if presentation is None or presentation.stdout_started:
            return
        presentation.stdout_started = True
        if presentation.header_emitted:
            return
        presentation.header_emitted = True
        if agent_id == "main":
            return
        identity = format_agent_identity(agent_id, presentation.agent_name)
        if self._powerline:
            print(f"\n{agent_header(identity, self._theme)}")
        else:
            print(f"\n{self._theme.agent.ansi}── agent {identity} ──{self._theme.reset}")

    def _parent_tool_key(
        self,
        agent_id: str,
        tool_use_id: str,
        presentation: _TurnPresentation | None,
    ) -> _ParentToolKey:
        return _ParentToolKey(
            parent_agent_id=agent_id,
            parent_run_id=presentation.key.run_id if presentation is not None else "",
            parent_turn_id=presentation.key.turn_id if presentation is not None else "",
            tool_use_id=tool_use_id,
        )

    def _purge_parent_turn_locked(self, key: _TurnKey) -> None:
        self._streamed_foreground_calls = {
            parent_key: result
            for parent_key, result in self._streamed_foreground_calls.items()
            if not (
                parent_key.parent_agent_id == key.agent_id
                and parent_key.parent_run_id == key.run_id
                and parent_key.parent_turn_id == key.turn_id
            )
        }
        self._foreground_parent_calls = {
            child_id: call
            for child_id, call in self._foreground_parent_calls.items()
            if not (
                call.parent_key.parent_agent_id == key.agent_id
                and call.parent_key.parent_run_id == key.run_id
                and call.parent_key.parent_turn_id == key.turn_id
            )
        }

    def _purge_legacy_foreground_parent_locked(self, parent_agent_id: str) -> None:
        def is_legacy_parent(parent_key: _ParentToolKey) -> bool:
            return (
                parent_key.parent_agent_id == parent_agent_id
                and not parent_key.parent_run_id
                and not parent_key.parent_turn_id
            )

        self._streamed_foreground_calls = {
            parent_key: result
            for parent_key, result in self._streamed_foreground_calls.items()
            if not is_legacy_parent(parent_key)
        }
        self._foreground_parent_calls = {
            child_id: call
            for child_id, call in self._foreground_parent_calls.items()
            if not is_legacy_parent(call.parent_key)
        }

    def _cleanup_turn_state_locked(self, agent_id: str) -> None:
        if agent_id == "main":
            return
        self._states.pop(agent_id, None)
        if self._active_agent == agent_id:
            self._active_agent = None

    def _cleanup_agent_locked(self, agent_id: str) -> None:
        self._cleanup_turn_state_locked(agent_id)
        self._current_turn_by_agent.pop(agent_id, None)
        self._turns = {key: value for key, value in self._turns.items() if key.agent_id != agent_id}
        self._completed_background_turns = OrderedDict(
            (key, value) for key, value in self._completed_background_turns.items() if key.agent_id != agent_id
        )
        if agent_id not in {"main", self._focused_agent}:
            self._agent_names.pop(agent_id, None)
        self._foreground_parent_calls.pop(agent_id, None)

    def _switch_agent(self, agent_id: str) -> tuple[_AgentRenderState, bool]:
        switched = self._active_agent != agent_id
        if switched:
            if self._active_agent is not None:
                active_state = self._state(self._active_agent)
                if isinstance(active_state.mode, _ReasoningMode):
                    sys.stdout.write("\n")
                    active_state.mode = _BoundaryMode()
                elif isinstance(active_state.mode, _TextMode):
                    print()
                    active_state.mode = _BoundaryMode()
                elif isinstance(active_state.mode, _ToolFieldMode):
                    if not active_state.mode.chunk_line_closed:
                        sys.stdout.write("\n")
            self._active_agent = agent_id
        state = self._state(agent_id)
        return state, switched

    def _render_locked(
        self,
        agent_id: str,
        event: StreamEvent,
        presentation: _TurnPresentation | None,
    ) -> None:  # noqa: C901
        state, switched = self._switch_agent(agent_id)
        if self._event_starts_stdout(event, presentation):
            self._ensure_stdout_header_locked(agent_id, presentation)
        if not isinstance(event, TextDelta):
            state.text_sanitizer.reset()
        if not isinstance(event, ReasoningDelta):
            state.reasoning_sanitizer.reset()
        if switched and isinstance(state.mode, _ToolFieldMode):
            if self._input_active:
                state.tool_arg_at_line_start = True
                state.mode = dataclasses.replace(
                    state.mode,
                    chunk_line_closed=not state.mode.first_delta,
                )
            else:
                sys.stdout.write(f"  {self._theme.warning.ansi}{state.mode.key} (continued){self._theme.reset}: ")
                self._flush()
                state.tool_arg_at_line_start = False
                state.mode = dataclasses.replace(state.mode, first_delta=True, chunk_line_closed=False)
        if isinstance(state.mode, _ReasoningMode) and not isinstance(event, ReasoningDelta):
            sys.stdout.write("\n")
            self._flush()
            state.mode = _BoundaryMode()
            self._drain_safe_boundary_locked()
        match event:
            case ReasoningDelta(delta=delta):
                delta = state.reasoning_sanitizer.feed(delta)
                if isinstance(state.mode, _TextMode):
                    print()
                    state.mode = _BoundaryMode()
                if not isinstance(state.mode, _ReasoningMode):
                    # <think> is usually followed by a newline, which would open
                    # the quote with an empty line.
                    delta = delta.lstrip("\n")
                    if not delta:
                        return
                    delta = f"> {delta}"
                    state.mode = _ReasoningMode()
                sys.stdout.write(_styled(self._theme.reasoning.ansi, delta.replace("\n", "\n> ")))
                self._flush()

            case TextDelta(delta=delta):
                safe_delta = state.text_sanitizer.feed(delta)
                if not isinstance(state.mode, _TextMode):
                    state.mode = _TextMode()
                state.pending_text.append(delta)
                if "[Output truncated:" in safe_delta:
                    sys.stdout.write(f"\n{_styled(self._theme.error.ansi, safe_delta.strip())}\n")
                    state.mode = _BoundaryMode()
                    self._drain_safe_boundary_locked()
                else:
                    self._render_text_delta_locked(state, safe_delta)

            case ImageOutput(data=data, media_type=mt):
                if isinstance(state.mode, _TextMode):
                    print()
                    state.mode = _BoundaryMode()
                path = _save_media(data, mt)
                print(f"{self._theme.success.ansi}[image saved: {path}]{self._theme.reset}")
                self._drain_safe_boundary_locked()

            case AudioOutput(data=data, media_type=mt):
                if isinstance(state.mode, _TextMode):
                    print()
                    state.mode = _BoundaryMode()
                path = _save_media(data, mt)
                print(f"{self._theme.success.ansi}[audio saved: {path}]{self._theme.reset}")
                self._drain_safe_boundary_locked()

            case VideoOutput(data=data, media_type=mt):
                if isinstance(state.mode, _TextMode):
                    print()
                    state.mode = _BoundaryMode()
                path = _save_media(data, mt)
                print(f"{self._theme.success.ansi}[video saved: {path}]{self._theme.reset}")
                self._drain_safe_boundary_locked()

            case ToolUseStart(index=index, tool_use_id=tid, name=name):
                if isinstance(state.mode, _TextMode):
                    print()
                    state.mode = _BoundaryMode()
                safe_name = sanitize_terminal_text(name)
                if self._powerline:
                    sys.stdout.write(f"\n{tool_title(safe_name, self._theme)}")
                else:
                    sys.stdout.write(f"\n{self._theme.tool.ansi}▶ {safe_name}{self._theme.reset}")
                self._flush()
                state.arg_streams[tid] = ToolArgStream(tid, index)
                state.active_tool_ids.add(tid)
                state.tool_names[tid] = name
                state.tool_arg_at_line_start = False

            case ToolInputDelta(tool_use_id=tid, partial_json=pj):
                stream = state.arg_streams.get(tid)
                if stream:
                    for fe in stream.feed(pj):
                        self._render_field_event(state, tid, fe)
                    if stream.done:
                        if not state.tool_arg_at_line_start:
                            sys.stdout.write("\n")
                        state.tool_arg_at_line_start = False
                        self._flush()
                        del state.arg_streams[tid]
                        if self._only_passive_tools_active(state):
                            self._drain_safe_boundary_locked()

            case ToolOutputDelta(tool_use_id=tid, key=key, delta=delta):
                if tid not in state.streamed_tool_ids:
                    sys.stdout.write("\n")
                state.streamed_tool_ids.add(tid)
                color = self._theme.stderr.ansi if key == "stderr" else self._theme.stdout.ansi
                sanitizer = state.tool_output_sanitizers.setdefault((tid, key), IncrementalTerminalSanitizer())
                sys.stdout.write(_styled(color, sanitizer.feed(delta)))
                self._flush()

            case ToolResult(tool_use_id=tid, name=name, is_error=is_error, content=content):
                content = sanitize_terminal_text(content)
                parent_key = self._parent_tool_key(agent_id, tid, presentation)
                foreground_result = self._streamed_foreground_calls.pop(parent_key, None)
                if foreground_result is not None:
                    foreground_status = foreground_result.status
                    identity = format_agent_identity(
                        foreground_result.child_agent_id,
                        foreground_result.child_agent_name,
                    )
                    if foreground_status is TurnStatus.SUCCEEDED:
                        content = f"[foreground agent {identity} returned its result to the parent]"
                    elif foreground_status is TurnStatus.CANCELLED:
                        content = (
                            f"[foreground agent {identity} was cancelled; the outcome was returned to the parent]"
                        )
                    else:
                        content = f"[foreground agent {identity} failed; the outcome was returned to the parent]"
                    color = (
                        self._theme.success.ansi
                        if foreground_status is TurnStatus.SUCCEEDED
                        else self._theme.error.ansi
                    )
                    sys.stdout.write(f"{self._theme.reset}\n{_styled(color, content)}\n")
                elif is_error:
                    sys.stdout.write(f"{self._theme.reset}\n{_styled(self._theme.error.ansi, content)}\n")
                elif name in {"run_agent", "spawn_agent"}:
                    sys.stdout.write(f"{self._theme.reset}\n{_styled(self._theme.success.ansi, content)}\n")
                elif tid in state.streamed_tool_ids:
                    sys.stdout.write(f"{self._theme.reset}\n")
                else:
                    sys.stdout.write(f"{self._theme.reset}\n{_styled(self._theme.success.ansi, content)}\n")
                self._flush()
                state.active_tool_ids.discard(tid)
                state.arg_streams.pop(tid, None)
                state.tool_names.pop(tid, None)
                state.tool_arg_at_line_start = False
                for sanitizer_key in tuple(state.tool_output_sanitizers):
                    if sanitizer_key[0] == tid:
                        state.tool_output_sanitizers.pop(sanitizer_key).reset()
                for sanitizer_key in tuple(state.tool_field_sanitizers):
                    if sanitizer_key[0] == tid:
                        state.tool_field_sanitizers.pop(sanitizer_key).reset()
                if (not state.active_tool_ids and not state.arg_streams) or self._only_passive_tools_active(state):
                    self._drain_safe_boundary_locked()

            case IterationEnd():
                # The agent has written this iteration into the context itself;
                # what is kept here is only ever the unfinished tail.
                state.pending_text.clear()
                state.text_sanitizer.reset()
                state.reasoning_sanitizer.reset()
                self._purge_legacy_foreground_parent_locked(agent_id)

            case Error(exception=exc):
                if self._input_active and isinstance(state.mode, _ToolFieldMode):
                    if not state.mode.chunk_line_closed:
                        sys.stdout.write(f"{self._theme.reset}\n")
                        self._flush()
                    state.mode = _BoundaryMode()
                    state.tool_arg_at_line_start = False
                if presentation is not None:
                    presentation.error_seen = True
                    if presentation.stdout_started:
                        sys.stdout.flush()
                identity = format_agent_identity(
                    agent_id,
                    presentation.agent_name if presentation is not None else self._agent_name(agent_id),
                )
                error_text = sanitize_terminal_text(f"Error from agent {identity}: {exc}")
                print(
                    f"\n{_styled(self._theme.error.ansi, error_text)}",
                    file=sys.stderr,
                    flush=True,
                )
                self._drain_safe_boundary_locked()

            case SessionEndEvent():
                self._purge_legacy_foreground_parent_locked(agent_id)
                if presentation is not None and presentation.error_seen and not presentation.stdout_started:
                    self._discard_suspended_owner_locked(agent_id)
                    self._drain_safe_boundary_locked()
                    return
                if isinstance(state.mode, _TextMode):
                    print()
                    state.mode = _BoundaryMode()
                state.mode = _BoundaryMode()
                self._discard_suspended_owner_locked(agent_id)
                self._drain_safe_boundary_locked()

    def _record_background_event_locked(self, agent_id: str, event: StreamEvent) -> None:
        state = self._state(agent_id)
        match event:
            case TextDelta(delta=delta):
                state.background_text.append(delta)
            case ToolUseStart(name=name):
                if name not in state.background_tools:
                    state.background_tools.append(name)
            case Error(exception=exc):
                state.background_errors.append(str(exc))
            case _:
                pass

    @staticmethod
    def _only_passive_tools_active(state: _AgentRenderState) -> bool:
        return (
            bool(state.active_tool_ids)
            and not state.arg_streams
            and all(state.tool_names.get(tool_use_id) == "monitor" for tool_use_id in state.active_tool_ids)
        )

    def _safe_boundary_is_open_locked(self) -> bool:
        if self._active_agent is None:
            return True
        state = self._state(self._active_agent)
        if isinstance(state.mode, _TextMode):
            return state.mode.paragraph_boundary_open
        if not isinstance(state.mode, _BoundaryMode):
            return False
        return not state.active_tool_ids or self._only_passive_tools_active(state)

    def _background_summary_locked(
        self,
        agent_id: str,
        presentation: _TurnPresentation,
        event: TurnFinished,
    ) -> _BackgroundSummary:
        state = self._states.get(agent_id)
        if state is None:
            return _BackgroundSummary(
                identity=format_agent_identity(agent_id, presentation.agent_name),
                reported_chars=0,
                tools=(),
                failed=event.status is not TurnStatus.SUCCEEDED,
            )
        text = "".join(state.background_text).strip()
        summary = _BackgroundSummary(
            identity=format_agent_identity(agent_id, presentation.agent_name),
            reported_chars=len(text),
            tools=tuple(state.background_tools),
            failed=event.status is not TurnStatus.SUCCEEDED or bool(state.background_errors),
        )
        state.background_text.clear()
        state.background_tools.clear()
        state.background_errors.clear()
        return summary

    def _flush_background_summaries_locked(self) -> None:
        if not self._background_summaries:
            return
        notices: list[str] = []
        for summary in self._background_summaries.values():
            parts = [f"agent {summary.identity} completed"]
            if summary.tools:
                parts.append(f"tools={','.join(summary.tools)}")
            if summary.reported_chars:
                parts.append(f"reported {summary.reported_chars} chars")
            if summary.failed:
                parts.append("failed; details follow in incoming report")
            notices.append(f"Background {'; '.join(parts)}")
        self.show_panel("\n".join(notices))
        self._background_summaries.clear()

    def _discard_background_summaries_locked(self) -> None:
        self._background_summaries.clear()

    def _drain_safe_boundary_locked(self, *, max_frames: int | None = None) -> None:
        frame_budget = max_frames or self._action_boundary_frames
        byte_budget = self._action_boundary_bytes
        suspended = self._suspended_actions.drain(max_frames=frame_budget, max_bytes=byte_budget)
        for frame in suspended:
            sys.stdout.write(frame)
            frame_budget -= 1
            byte_budget -= len(frame.encode("utf-8"))
        if self.display_mode is DisplayMode.ALL_ACTIONS:
            self._discard_background_summaries_locked()
            if frame_budget > 0 and byte_budget > 0:
                for frame in self._actions.drain(max_frames=frame_budget, max_bytes=byte_budget):
                    sys.stdout.write(frame)
            self._flush()
        elif suspended:
            self._flush()

    def _drain_all_actions_locked(self) -> None:
        while self.queued_action_count:
            before = self.queued_action_count
            self._drain_safe_boundary_locked()
            if self.queued_action_count >= before:
                break

    def _is_suspended_tool_event(self, agent_id: str, event: StreamEvent) -> bool:
        match event:
            case (
                ToolInputDelta(tool_use_id=tool_use_id)
                | ToolOutputDelta(tool_use_id=tool_use_id)
                | ToolResult(tool_use_id=tool_use_id)
            ):
                return (agent_id, tool_use_id) in self._suspended_tool_calls
            case _:
                return False

    def _observe_suspended_tool_event_locked(self, agent_id: str, event: StreamEvent) -> None:
        state = self._state(agent_id)
        self._suspended_actions.observe(agent_id, event, agent_name=self._agent_name(agent_id))
        match event:
            case ToolInputDelta(tool_use_id=tool_use_id, partial_json=partial_json):
                stream = state.arg_streams.get(tool_use_id)
                if stream is not None:
                    tuple(stream.feed(partial_json))
                    if stream.done:
                        state.arg_streams.pop(tool_use_id, None)
                        if isinstance(state.mode, _ToolFieldMode) and state.mode.tool_use_id == tool_use_id:
                            state.mode = _BoundaryMode()
            case ToolOutputDelta(tool_use_id=tool_use_id):
                state.streamed_tool_ids.add(tool_use_id)
            case ToolResult(tool_use_id=tool_use_id):
                state.active_tool_ids.discard(tool_use_id)
                state.arg_streams.pop(tool_use_id, None)
                state.tool_names.pop(tool_use_id, None)
                self._suspended_tool_calls.discard((agent_id, tool_use_id))
            case _:
                pass

    def _discard_suspended_owner_locked(self, agent_id: str) -> None:
        self._suspended_actions.discard_agent(agent_id)
        self._suspended_tool_calls = {
            (owner, tool_use_id) for owner, tool_use_id in self._suspended_tool_calls if owner != agent_id
        }

    def _render_text_delta_locked(self, state: _AgentRenderState, delta: str) -> None:
        if not isinstance(state.mode, _TextMode):
            raise RuntimeError("text delta requires text render mode")
        mode = state.mode
        if delta:
            mode = dataclasses.replace(mode, paragraph_boundary_open=False)
        start = 0
        for index, character in enumerate(delta):
            if character == "\n":
                if mode.paragraph_newline_pending:
                    sys.stdout.write(delta[start : index + 1])
                    self._flush()
                    mode = dataclasses.replace(
                        mode,
                        paragraph_newline_pending=False,
                        paragraph_boundary_open=True,
                    )
                    state.mode = mode
                    self._drain_safe_boundary_locked()
                    start = index + 1
                else:
                    mode = dataclasses.replace(mode, paragraph_newline_pending=True)
            else:
                mode = dataclasses.replace(
                    mode,
                    paragraph_newline_pending=False,
                    paragraph_boundary_open=False,
                )
        if start < len(delta):
            sys.stdout.write(delta[start:])
            self._flush()
        state.mode = mode

    def _render_field_event(
        self,
        state: _AgentRenderState,
        tool_use_id: str,
        event: ToolFieldStart | ToolFieldDelta | ToolFieldEnd,
    ) -> None:
        match event:
            case ToolFieldStart(key=key):
                key = sanitize_terminal_text(key).replace("\n", " ")
                state.mode = _ToolFieldMode(tool_use_id=tool_use_id, key=key)
                state.tool_field_sanitizers[(tool_use_id, key)] = IncrementalTerminalSanitizer()
                if not self._input_active:
                    leading = "" if state.tool_arg_at_line_start else "\n"
                    sys.stdout.write(f"{leading}  {self._theme.warning.ansi}{key}{self._theme.reset}: ")
                    self._flush()
                    state.tool_arg_at_line_start = False
            case ToolFieldDelta(text=text):
                mode = state.mode
                if not isinstance(mode, _ToolFieldMode) or mode.tool_use_id != tool_use_id:
                    raise RuntimeError(f"tool field delta for inactive stream {tool_use_id}")
                sanitizer = state.tool_field_sanitizers.setdefault(
                    (tool_use_id, mode.key), IncrementalTerminalSanitizer()
                )
                text = sanitizer.feed(text)
                if not text:
                    return
                if mode.chunk_line_closed:
                    sys.stdout.write(f"  {self._theme.warning.ansi}{mode.key} (continued){self._theme.reset}: ")
                    state.tool_arg_at_line_start = False
                    mode = dataclasses.replace(mode, first_delta=True, chunk_line_closed=False)
                elif self._input_active and mode.first_delta:
                    leading = "" if state.tool_arg_at_line_start else "\n"
                    sys.stdout.write(f"{leading}  {self._theme.warning.ansi}{mode.key}{self._theme.reset}: ")
                    state.tool_arg_at_line_start = False
                if mode.first_delta and "\n" in text:
                    sys.stdout.write("\n")
                state.mode = dataclasses.replace(mode, first_delta=False)
                styled = _styled(self._theme.reasoning.ansi, text)
                if self._input_active and text.endswith("\n") and self._theme.reasoning.ansi:
                    styled = styled[: -len(self._theme.reasoning.ansi + self._theme.reset)]
                sys.stdout.write(styled)
                if self._input_active and text:
                    # A partial terminal frame would share a row with prompt_toolkit's
                    # redraw. Complete one visual line per received value fragment;
                    # the next fragment is labelled as a continuation of this field.
                    if not text.endswith("\n"):
                        sys.stdout.write(f"{self._theme.reset}\n")
                    state.tool_arg_at_line_start = True
                    state.mode = dataclasses.replace(state.mode, chunk_line_closed=True)
                self._flush()
            case ToolFieldEnd():
                if isinstance(state.mode, _ToolFieldMode) and state.mode.tool_use_id == tool_use_id:
                    if not state.mode.chunk_line_closed and not (self._input_active and state.mode.first_delta):
                        sys.stdout.write(self._theme.reset)
                        self._flush()
                    state.tool_field_sanitizers.pop((tool_use_id, state.mode.key), None)
                    state.mode = _BoundaryMode()


async def render_runtime_event(renderer: ReplRenderer, envelope: AgentEventEnvelope) -> None:
    match envelope.event:
        case AgentStarted(name=name):
            await renderer.remember_agent(envelope.agent_id, name)
            if envelope.execution_mode is ExecutionMode.BACKGROUND:
                await renderer.observe_runtime_event(envelope.agent_id, envelope.event)
        case ForegroundEntered():
            await renderer.enter_foreground(
                envelope.agent_id,
                envelope.parent_tool_use_id,
                envelope.parent_agent_id,
            )
        case ForegroundExited(status=status):
            await renderer.exit_foreground(envelope.agent_id, status)
        case TurnStarted() as event:
            await renderer.start_turn(
                envelope.agent_id,
                event,
                run_id=envelope.run_id,
                turn_id=envelope.turn_id,
                execution_mode=envelope.execution_mode,
            )
        case TurnFinished() as event:
            await renderer.finish_turn(
                envelope.agent_id,
                event,
                run_id=envelope.run_id,
                turn_id=envelope.turn_id,
                execution_mode=envelope.execution_mode,
            )
        case AgentStopped() as event:
            await renderer.stop_agent(
                envelope.agent_id,
                event,
                background=envelope.execution_mode is ExecutionMode.BACKGROUND,
            )
        case (
            OutcomeDelivered()
            | InputReceived()
            | InputBuffered()
            | InputRecalled()
            | InputClaimed()
            | InputDelivered()
            | InterruptionRequested()
            | InterruptionCommitted()
            | EditorSnapshot()
            | ShutdownRecorded()
            | RecoveryApplied()
            | ConfigurationChanged()
            | MessageCommitted()
            | ContextForked()
            | ContextCleared()
        ):
            if envelope.execution_mode is ExecutionMode.BACKGROUND:
                await renderer.observe_runtime_event(envelope.agent_id, envelope.event)
        case event:
            await renderer.render(
                envelope.agent_id,
                event,
                run_id=envelope.run_id,
                turn_id=envelope.turn_id,
                execution_mode=envelope.execution_mode,
            )


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
    return await observe_agent_turn(
        agent=agent,
        context=ctx,
        prompt=prompt,
        identity=identity,
        hub=event_hub,
        provenance=InputProvenance(human_authored=True, source=source, author="human"),
    )


async def run_prompt_messages(
    agent: Agent,
    ctx: ContextStore,
    messages: tuple[Message, ...],
    event_hub: SessionEventHub,
    run_id: str,
    *,
    source: str,
    identity: TurnIdentity | None = None,
    on_input_committed: Callable[[], Awaitable[None]] | None = None,
    source_input_ids: tuple[str | None, ...] | None = None,
) -> TurnOutcome:
    if not messages:
        raise ValueError("messages must not be empty")
    turn_identity = identity or new_turn_identity(
        agent_id="main",
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        run_id=run_id,
        context_id=ctx.session_id,
    )
    for message in messages:
        text = "".join(block.text for block in message.content if isinstance(block, TextBlock))
        message_source = message.provenance.source if message.provenance is not None else source
        await event_hub.publish_for(turn_identity, InputReceived(text=text, source=message_source))
    return await observe_agent_turn_messages(
        agent=agent,
        context=ctx,
        messages=messages,
        identity=turn_identity,
        hub=event_hub,
        on_input_committed=on_input_committed,
        source_input_ids=source_input_ids,
    )


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
            # Reusing a generated schema as explicit would discard Annotated validators.
            schema=tool.schema if tool._schema_explicit else MappingProxyType({}),
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


async def _read_input_async(
    session: Any,
    renderer: ReplRenderer,
    on_interrupt: Callable[[], None],
    admit_submission: Callable[[str, str, int | None], Awaitable[InputSubmitted]],
    initial_text: str = "",
    *,
    prompt_factory: Callable[[], object] = _panel.prompt_message,
) -> InputSubmitted:
    async def finish_submission(
        editor_value: str,
        target_agent_id: str,
        reserved_seq: int | None,
        submitted_at: datetime | None,
        prior_failure: BaseException | None = None,
    ) -> InputSubmitted:
        failure = prior_failure
        clear_editor = False
        try:
            admission: asyncio.Future[InputSubmitted] = asyncio.ensure_future(
                admit_submission(editor_value.strip(), target_agent_id, reserved_seq)
            )
            while True:
                try:
                    submitted = await asyncio.shield(admission)
                    break
                except asyncio.CancelledError as exc:
                    if admission.done():
                        submitted = admission.result()
                        failure = failure or exc
                        break
                    failure = failure or exc
            clear_editor = submitted.disposition is SubmissionDisposition.PENDING
            if submitted.disposition is SubmissionDisposition.PENDING and submitted_at is not None:
                rendering = asyncio.ensure_future(renderer.submitted(submitted.text, submitted_at))
                while True:
                    try:
                        await asyncio.shield(rendering)
                        break
                    except asyncio.CancelledError as exc:
                        if rendering.done():
                            rendering.result()
                            failure = failure or exc
                            break
                        failure = failure or exc
        finally:
            _panel.complete_submission(
                session,
                editor_value,
                clear_editor=clear_editor,
            )
        if failure is not None:
            raise failure
        return submitted

    renderer.set_input_active(True)
    try:
        while True:
            try:
                if initial_text:
                    value = await session.prompt_async(prompt_factory(), default=initial_text)
                    initial_text = ""
                else:
                    value = await session.prompt_async(prompt_factory())
                editor_value = str(value)
                text = editor_value.strip()
                if text:
                    target_agent_id = _panel.accepted_target(session, renderer.focused_agent)
                    return await finish_submission(
                        editor_value,
                        target_agent_id,
                        _panel.accepted_sequence(session),
                        _panel.accepted_at(session),
                    )
                _panel.complete_submission(session, editor_value, clear_editor=True)
            except KeyboardInterrupt as exc:
                # The prompt is up for the whole session now, and it puts the
                # terminal in raw mode - so Ctrl+C arrives here as a keypress
                # and never reaches the signal handler that used to stop a
                # running turn. Do its job, and go back to waiting.
                reserved_seq = _panel.accepted_sequence(session)
                if reserved_seq is not None:
                    editor_value = _panel.editor_text(session)
                    target_agent_id = _panel.accepted_target(session, renderer.focused_agent)
                    await finish_submission(
                        editor_value,
                        target_agent_id,
                        reserved_seq,
                        _panel.accepted_at(session),
                        exc,
                    )
                    raise exc
                on_interrupt()
            except asyncio.CancelledError as exc:
                reserved_seq = _panel.accepted_sequence(session)
                if reserved_seq is None:
                    raise
                editor_value = _panel.editor_text(session)
                target_agent_id = _panel.accepted_target(session, renderer.focused_agent)
                await finish_submission(
                    editor_value,
                    target_agent_id,
                    reserved_seq,
                    _panel.accepted_at(session),
                    exc,
                )
                raise exc
            except BaseException as exc:
                reserved_seq = _panel.accepted_sequence(session)
                if reserved_seq is None:
                    raise
                editor_value = _panel.editor_text(session)
                target_agent_id = _panel.accepted_target(session, renderer.focused_agent)
                await finish_submission(
                    editor_value,
                    target_agent_id,
                    reserved_seq,
                    _panel.accepted_at(session),
                    exc,
                )
                raise exc
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
    _command_print(f"Focused agent: {BOLD}{renderer.focused_agent}{RESET}")
    records = local_background_agent_records()
    if not records:
        _command_print("No local background agents.")
        return
    for record in records:
        marker = "*" if record.id == renderer.focused_agent else " "
        _command_print(f"{marker} {record.id} name={record.name!r} kind={record.kind} pid={record.pid}")


async def _handle_agent_actions(
    renderer: ReplRenderer,
    value: str,
    publish: Callable[[RuntimeEvent], Awaitable[object]] | None = None,
) -> bool:
    """Show or update the observation-only background action policy."""
    if not value:
        _command_print(
            f"Agent actions: {BOLD}{renderer.display_mode.value}{RESET}; {renderer.queued_action_count} queued"
        )
        return True
    try:
        mode = DisplayMode.parse(value)
    except ValueError as exc:
        _command_print(str(exc))
        return False
    change = await renderer.set_display_mode(mode)
    detail = ""
    if change.discarded_frames or change.discarded_bytes:
        detail = f"; discarded {change.discarded_frames} queued frame(s) ({change.discarded_bytes} bytes)"
    _command_print(f"Agent actions: {BOLD}{mode.value}{RESET}{detail}")
    if change.current is not change.previous and publish is not None:
        await publish(ConfigurationChanged(name="agent_actions", value=mode.value, source="interactive"))
    return True


# ── REPL commands ────────────────────────────────────────────────────


class Command(NamedTuple):
    """A REPL command with separate show (no arg) and apply (with arg) modes."""

    show: Callable[[], None]
    apply: Callable[[str], object]


_COMMAND_OUTPUT: ContextVar[list[str] | None] = ContextVar("axio_repl_command_output", default=None)
_COMMAND_THEME: ContextVar[TerminalTheme] = ContextVar("axio_repl_command_theme", default=DEFAULT_THEME)


def _command_print(*values: object, sep: str = " ", end: str = "\n") -> None:
    rendered = sep.join(str(value) for value in values) + end
    theme = _COMMAND_THEME.get()
    if theme is not DEFAULT_THEME:
        rendered = rendered.replace(BOLD, theme.command.ansi).replace(RESET, theme.reset)
    output = _COMMAND_OUTPUT.get()
    if output is None:
        print(rendered, end="")
        return
    output.append(rendered)


@contextmanager
def _command_theme(theme: TerminalTheme) -> Iterator[None]:
    token = _COMMAND_THEME.set(theme)
    try:
        yield
    finally:
        _COMMAND_THEME.reset(token)


@contextmanager
def _capture_command_output(theme: TerminalTheme = DEFAULT_THEME) -> Iterator[list[str]]:
    output: list[str] = []
    output_token = _COMMAND_OUTPUT.set(output)
    with _command_theme(theme):
        try:
            yield output
        finally:
            _COMMAND_OUTPUT.reset(output_token)


# CLI arg attr → slash command name (for unified init).
_CLI_TO_SLASH: dict[str, str] = {
    "effort": "/effort",
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
        result = commands[cmd_name].apply(arg)
        if cmd_name == "/effort" and result is None:
            raise SystemExit(2)


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
    _command_print(f"Current model: {BOLD}{model.id}{RESET}")
    _command_print(f"Capabilities: {caps}")
    _command_print("Available:")
    width = shutil.get_terminal_size((100, 24)).columns
    for line in _columnise(sorted(transport.models.keys()), max(20, width - 2)):
        _command_print(f"  {line}")


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
        _command_print(f"No model matching {arg!r}. Available: {', '.join(transport.models.keys())}")
    else:
        _command_print(f"Ambiguous — matches: {', '.join(matches.keys())}")
    return None


def _apply_model(
    transport: Any,
    agent: Agent,
    tools: list[Tool[Any]],
    root: Path,
    agents_text: str,
    arg: str,
    parent_peer_id: str | None = None,
    sandbox_note: str = "",
    effort: EffortRuntime | None = None,
    effective_username: str | None = None,
    model_context: str = "",
) -> None:
    chosen = _choose_model(transport, arg)
    if chosen is None:
        return
    transport.model = chosen
    if effort is not None:
        requested = effort.state.requested
        try:
            state = effort.reapply()
            reset_reason = None
        except ValueError as exc:
            state = effort.configure("default")
            reset_reason = str(exc)
    else:
        requested = None
        state = None
        reset_reason = None
    agent.system = _build_runtime_system_prompt(
        root,
        transport.model,
        tools,
        agents_text,
        effective_username=effective_username,
        effort=effort,
        parent_peer_id=parent_peer_id,
        sandbox_note=sandbox_note,
        model_context=model_context,
    )
    _command_print(f"Switched to {BOLD}{transport.model.id}{RESET}")
    if reset_reason is not None:
        _command_print(
            f"Effort {BOLD}{requested}{RESET} is unavailable for this model; reset to {BOLD}default{RESET}."
        )
        _command_print(reset_reason)
    elif state is not None and state.requested is not None:
        _command_print(f"Effort reapplied: {BOLD}{state.requested}{RESET} via {state.mechanism.value}")
        if state.note:
            _command_print(state.note)


# ── effort ──


def _show_effort(effort: EffortRuntime) -> None:
    state = effort.state
    requested = state.requested or "default"
    _command_print(f"Requested effort: {BOLD}{requested}{RESET}")
    if state.provider_value is None:
        detail = ""
    elif state.mechanism.value == "native-budget" and isinstance(state.provider_value, int):
        detail = f" (budget: {state.provider_value} tokens)"
    elif state.mechanism.value == "native-budget":
        detail = f" (thinking level: {state.provider_value})"
    else:
        detail = f" (provider effort: {state.provider_value})"
    _command_print(f"Effective mechanism: {BOLD}{state.mechanism.value}{RESET}{detail}")
    if state.note:
        _command_print(f"Limitation: {state.note}")
    values = ", ".join(("default", *state.allowed))
    _command_print(f"Valid values: {values}")


def _apply_effort(effort: EffortRuntime, arg: str) -> EffortState | None:
    try:
        state = effort.configure(arg)
    except ValueError as exc:
        _command_print(str(exc))
        return None
    _show_effort(effort)
    return state


def _apply_agent_effort(
    effort: EffortRuntime,
    agent: Agent,
    root: Path,
    tools: list[Tool[Any]],
    agents_text: str,
    parent_peer_id: str | None,
    sandbox_note: str,
    arg: str,
    effective_username: str | None = None,
    model_context: str = "",
) -> EffortState | None:
    state = _apply_effort(effort, arg)
    if state is None:
        return None
    agent.system = _build_runtime_system_prompt(
        root,
        cast(Any, agent.transport).model,
        tools,
        agents_text,
        effective_username=effective_username,
        effort=effort,
        parent_peer_id=parent_peer_id,
        sandbox_note=sandbox_note,
        model_context=model_context,
    )
    return state


# ── temperature ──


def _show_temperature(transport: Any) -> None:
    temp = getattr(transport, "temperature", None)
    _command_print(f"Temperature: {BOLD}{temp if temp is not None else 'default'}{RESET}")


def _apply_temperature(transport: Any, arg: str) -> None:
    try:
        val = float(arg)
    except ValueError:
        _command_print(f"Invalid temperature: {arg!r}")
        return
    if hasattr(transport, "temperature"):
        transport.temperature = val
        _command_print(f"Temperature: {BOLD}{val}{RESET}")
    else:
        _command_print("Transport does not support temperature")


# ── iterations ──


def _show_iterations(agent: Agent) -> None:
    _command_print(f"Max iterations: {BOLD}{agent.max_iterations}{RESET}")


def _apply_iterations(agent: Agent, arg: str) -> None:
    try:
        val = int(arg)
    except ValueError:
        _command_print(f"Invalid value: {arg!r}")
        return
    if val < 1:
        # Zero reads as "no limit" and means the opposite: the loop runs no
        # iterations at all, so the agent answers nothing and reports that it
        # ran out. There is no unlimited - a large number is how you say it.
        _command_print(
            f"Max iterations must be at least 1. The default, {BOLD}1000{RESET}, is already out of the way."
        )
        return
    agent.max_iterations = val
    _command_print(f"Max iterations: {BOLD}{val}{RESET}")


# ── max-tokens ──


def _show_max_tokens(transport: Any) -> None:
    cur = getattr(transport, "max_output_tokens", None)
    model_default = getattr(getattr(transport, "model", None), "max_output_tokens", None)
    if cur:
        _command_print(f"Max output tokens: {BOLD}{cur}{RESET} (model default: {model_default})")
    else:
        _command_print(f"Max output tokens: {BOLD}{model_default}{RESET} (model default)")


def _apply_max_tokens(transport: Any, arg: str) -> None:
    model_default = getattr(getattr(transport, "model", None), "max_output_tokens", None)
    if arg == "default":
        transport.max_output_tokens = None
        _command_print(f"Max output tokens: {BOLD}{model_default}{RESET} (model default)")
        return
    try:
        val = int(arg)
    except ValueError:
        _command_print(f"Invalid value: {arg!r}")
        return
    transport.max_output_tokens = val
    _command_print(f"Max output tokens: {BOLD}{val}{RESET}")


# ── debug ──


def _show_debug(transport: Any) -> None:
    cur = getattr(transport, "debug", False)
    _command_print(f"Debug: {BOLD}{'on' if cur else 'off'}{RESET}")


def _apply_debug(transport: Any, arg: str) -> None:
    val = arg.lower()
    if val == "on":
        transport.debug = True
        _command_print(f"Debug: {BOLD}on{RESET} (request/response bodies logged to stderr)")
    elif val == "off":
        transport.debug = False
        _command_print(f"Debug: {BOLD}off{RESET}")
    else:
        _command_print("Usage: /debug on|off")


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
    InputBuffered: "input_buffered",
    InputRecalled: "input_recalled",
    InputClaimed: "input_claimed",
    InputDelivered: "input_delivered",
    InterruptionRequested: "interruption_requested",
    InterruptionCommitted: "interruption_committed",
    EditorSnapshot: "editor_snapshot",
    ShutdownRecorded: "shutdown_recorded",
    RecoveryApplied: "recovery_applied",
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
    InputBuffered,
    InputRecalled,
    InputClaimed,
    InputDelivered,
    InterruptionRequested,
    InterruptionCommitted,
    EditorSnapshot,
    ShutdownRecorded,
    RecoveryApplied,
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
            "source_input_id": event.source_input_id,
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

    if one_shot:
        print(f"Session log: {journal.events_path}", file=sys.stderr)

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

    class VersionAction(argparse.Action):
        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: object,
            option_string: str | None = None,
        ) -> None:
            del namespace, option_string, values
            print(_version.version_report(module_source=__file__))
            parser.exit(0)

    parser = argparse.ArgumentParser(description="REPL coding assistant (axio)", allow_abbrev=False)
    parser.add_argument("prompt", nargs="?", default=None, help="Single prompt (non-interactive)")
    parser.add_argument(
        "--version",
        action=VersionAction,
        nargs=0,
        help="Show executable and source provenance, then exit",
    )
    parser.add_argument("--agent", default=None, help="Named agent bundle from CONFIG_DIR/agents/NAME")
    parser.add_argument("--config-dir", type=Path, default=None, help="Configuration root (default: XDG config/axio)")
    parser.add_argument("--list-agents", action="store_true", help="List available named agent bundles and exit")
    parser.add_argument("--transport", default=None, help="Transport name (auto-detected if omitted)")
    parser.add_argument("--transport-base-url", default=None, help="Override the transport API base URL")
    parser.add_argument(
        "--transport-api-key-env",
        default=None,
        metavar="ENV_VAR",
        help="Read the transport API key from this environment variable",
    )
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--effort", default=None, help=f"Effort level: default, {', '.join(EFFORT_LEVELS)}")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--max-iterations", type=int, default=1000)
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument("--debug", dest="debug", action="store_true", help="Log request/response bodies")
    debug_group.add_argument("--no-debug", dest="debug", action="store_false", help="Disable request/response logging")
    parser.set_defaults(debug=False)
    parser.add_argument(
        "--agent-actions",
        choices=(DisplayMode.ACTIVE_ONLY.value, DisplayMode.ALL_ACTIONS.value),
        default=DisplayMode.ACTIVE_ONLY.value,
        help="Show framed actions from non-active agents (default: off)",
    )
    parser.add_argument(
        "--theme",
        choices=theme_names(),
        default=DEFAULT_THEME.name,
        help="Terminal theme (default: default)",
    )
    powerline_group = parser.add_mutually_exclusive_group()
    powerline_group.add_argument(
        "--powerline",
        dest="powerline",
        action="store_true",
        help="Use Powerline segments for the prompt, tool names, and agent frames",
    )
    powerline_group.add_argument(
        "--no-powerline",
        dest="powerline",
        action="store_false",
        help="Use the plain terminal presentation",
    )
    parser.set_defaults(powerline=False)
    session_log_group = parser.add_mutually_exclusive_group()
    session_log_group.add_argument(
        "--session-log", dest="no_session_log", action="store_false", help="Write the session JSONL journal"
    )
    session_log_group.add_argument(
        "--no-session-log", dest="no_session_log", action="store_true", help="Do not write the session JSONL journal"
    )
    parser.set_defaults(no_session_log=False)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="EVENTS_JSONL",
        help="Recover context, pending input, and editor state from a stopped session journal",
    )
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
    parser.add_argument(
        "--sandbox-image",
        default=_sandbox.DEFAULT_SANDBOX_IMAGE,
        help="Image for --sandbox docker (default: locally built standard image)",
    )
    parser.add_argument(
        "--sandbox-network",
        default=None,
        help="User-defined internal Docker network for restricted service access (default: no network)",
    )
    parser.add_argument(
        "--sandbox-memory",
        default=_sandbox.DEFAULT_SANDBOX_MEMORY,
        help=f"Sandbox memory limit (default: {_sandbox.DEFAULT_SANDBOX_MEMORY})",
    )
    parser.add_argument(
        "--sandbox-cpus",
        default=_sandbox.DEFAULT_SANDBOX_CPUS,
        help=f"Sandbox CPU limit (default: {_sandbox.DEFAULT_SANDBOX_CPUS})",
    )
    parser.add_argument("--sandbox-proxy", default=None, help="HTTP(S) policy proxy URL inside the sandbox network")
    parser.add_argument("--sandbox-no-proxy", default=None, help="Comma-separated proxy bypass hostnames")
    parser.add_argument("--sandbox-pypi-index", default=None, help="Internal PyPI/simple index URL")
    parser.add_argument("--sandbox-npm-registry", default=None, help="Internal npm registry URL")
    parser.add_argument("--sandbox-cargo-index", default=None, help="Internal Cargo registry index URL")
    parser.add_argument("--sandbox-go-proxy", default=None, help="Internal GOPROXY URL")
    parser.add_argument(
        "--sandbox-go-sumdb",
        default=None,
        help="Explicit GOSUMDB setting (default: preserve Go's checksum database behavior)",
    )
    parser.add_argument(
        "--sandbox-datasets",
        type=Path,
        default=None,
        help="Host dataset directory mounted read-only at /datasets",
    )
    parser.add_argument(
        "--sandbox-ca-cert",
        type=Path,
        default=None,
        help="Full CA bundle (system roots plus interception CA) mounted read-only for common clients",
    )
    parser.add_argument(
        "--tools",
        default=None,
        help="Enabled tools: 'all', 'none', or a comma-separated list (default: all)",
    )
    return parser


async def main() -> None:
    parser = _build_argument_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    config_dir = (
        args.config_dir.expanduser().resolve() if args.config_dir is not None else _agent_config.default_config_dir()
    )
    if args.list_agents:
        try:
            agent_names = _agent_config.list_agent_names(config_dir)
        except _agent_config.AgentConfigError as error:
            parser.error(str(error))
        print("\n".join(agent_names))
        return
    try:
        agent_name = _agent_config.resolve_agent_name(args.agent)
        profile = _agent_config.load_agent_profile(config_dir, agent_name, cwd=root)
        _agent_config.apply_profile_to_args(args, profile, _agent_config.explicit_cli_destinations(argv))
        profile_instructions = profile.instructions_text()
        api_key = _agent_config.resolve_api_key(args.transport_api_key_env)
        theme = resolve_theme(args.theme)
    except _agent_config.AgentConfigError as error:
        parser.error(str(error))
    effective_username = resolve_effective_username()
    model_context = profile.model_context or ""
    theme, args.powerline = resolve_terminal_presentation(
        theme,
        powerline=args.powerline,
        one_shot=args.prompt is not None,
        stdout_is_tty=sys.stdout.isatty(),
        no_color="NO_COLOR" in os.environ,
    )
    if (args.transport_base_url is not None or args.transport_api_key_env is not None) and args.transport is None:
        parser.error("transport connection settings require an explicit transport name")
    recovery: RecoveryMaterialization | None = None
    if args.resume is not None:
        if args.prompt is not None:
            parser.error("--resume is interactive-only and cannot be combined with a one-shot prompt")
        if args.no_session_log:
            parser.error("--resume requires session journaling; enable runtime.session_log or pass --session-log")
        try:
            recovery = materialize_recovery(args.resume.expanduser().resolve())
        except (OSError, RecoveryError) as error:
            parser.error(f"cannot recover session: {error}")

    try:
        sandbox_options = _sandbox.SandboxOptions(
            network=args.sandbox_network,
            memory=args.sandbox_memory,
            cpus=args.sandbox_cpus,
            proxy=args.sandbox_proxy,
            no_proxy=args.sandbox_no_proxy,
            pypi_index=args.sandbox_pypi_index,
            npm_registry=args.sandbox_npm_registry,
            cargo_index=args.sandbox_cargo_index,
            go_proxy=args.sandbox_go_proxy,
            go_sumdb=args.sandbox_go_sumdb,
            datasets=args.sandbox_datasets.expanduser().resolve() if args.sandbox_datasets else None,
            ca_certificate=args.sandbox_ca_cert.expanduser().resolve() if args.sandbox_ca_cert else None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if sandbox_options.requires_docker and (
        args.sandbox == "none" or (args.sandbox == "auto" and not _sandbox.docker_available())
    ):
        parser.error("restricted sandbox settings require Docker, but sandbox execution is unavailable")

    setup_logging(args.debug)
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
        ) as session_journal,
        aiohttp.ClientSession() as session,
        AsyncExitStack() as stack,
    ):
        if recovery is not None and session_journal is None:
            parser.error("recovery stopped because the new session journal could not be opened")
        transport_cls, _ = (
            _select_transport(args.transport, credential_override=True)
            if api_key is not None
            else _select_transport(args.transport)
        )
        project_instructions = load_agents_instructions(root)
        agents_text = "\n\n".join(
            part
            for part in (
                f"Agent profile instructions:\n{profile_instructions}" if profile_instructions else "",
                f"AGENTS.md instructions:\n{project_instructions}" if project_instructions else "",
            )
            if part
        )
        transport_kwargs: dict[str, object] = {"session": session}
        if args.transport_base_url is not None:
            transport_kwargs["base_url"] = args.transport_base_url
        if api_key is not None:
            transport_kwargs["api_key"] = api_key
        try:
            transport_signature = inspect.signature(transport_cls)
        except (TypeError, ValueError):
            transport_signature = None
        if transport_signature is not None:
            try:
                transport_signature.bind(**transport_kwargs)
            except TypeError:
                parser.error(f"transport {args.transport!r} does not accept the configured connection settings")
        transport = transport_cls(**transport_kwargs)
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

        effort = EffortRuntime(transport)

        # Transport-level commands (available before agent creation).
        commands: dict[str, Command] = {
            "/effort": Command(lambda: _show_effort(effort), lambda a: _apply_effort(effort, a)),
            "/temperature": Command(lambda: _show_temperature(transport), lambda a: _apply_temperature(transport, a)),
            "/max-tokens": Command(lambda: _show_max_tokens(transport), lambda a: _apply_max_tokens(transport, a)),
            "/debug": Command(lambda: _show_debug(transport), lambda a: _apply_debug(transport, a)),
        }
        with _command_theme(theme):
            _apply_cli_args(args, commands)

        try:
            tools, sandbox_desc, tool_root, sandbox_note = await _sandbox.build_tools(
                stack, list(TOOLS), args.sandbox, args.sandbox_image, root, sandbox_options
            )
        except RuntimeError as exc:
            parser.error(str(exc))
        try:
            tools = _select_configured_tools(args.tools, tools)
        except ValueError as error:
            parser.error(str(error))
        if args.prompt is not None:
            print(f"Tools: {sandbox_desc}", file=sys.stderr)
        system = _build_runtime_system_prompt(
            tool_root,
            transport.model,
            tools,
            agents_text,
            effective_username=effective_username,
            effort=effort,
            sandbox_note=sandbox_note,
            model_context=model_context,
        )
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

        async def _publish_main_event(
            event: RuntimeEvent,
            *,
            reserved_seq: int | None = None,
        ) -> AgentEventEnvelope:
            return await event_hub.publish(
                event,
                run_id=main_run_id,
                agent_id="main",
                parent_agent_id=None,
                turn_id=None,
                execution_mode=ExecutionMode.FOREGROUND,
                context_id=ctx.session_id,
                reserved_seq=reserved_seq,
            )

        await _publish_main_event(AgentStarted(name=AGENT_NAME, kind="repl-agent"))
        for config_name, config_value in (
            ("agent", profile.name),
            ("transport", getattr(transport, "name", type(transport).__name__)),
            ("model", transport.model.id),
            ("sandbox", sandbox_desc),
            ("max_iterations", agent.max_iterations),
            ("max_output_tokens", getattr(transport, "max_output_tokens", None)),
            ("temperature", getattr(transport, "temperature", None)),
            ("effort", effort.state.to_dict()),
            ("agent_actions", args.agent_actions),
            ("theme", theme.name),
            ("powerline", args.powerline),
        ):
            await _publish_main_event(ConfigurationChanged(name=config_name, value=config_value, source="startup"))

        if recovery is not None and recovery.messages:
            recovery_identity = new_turn_identity(
                agent_id="main",
                parent_agent_id=None,
                execution_mode=ExecutionMode.FOREGROUND,
                run_id=main_run_id,
                context_id=ctx.session_id,
            )
            ctx.bind_identity(recovery_identity)
            await ctx.append_many(list(recovery.messages))

        def _command_configuration(command_name: str) -> object:
            match command_name:
                case "/model":
                    return transport.model.id
                case "/iterations":
                    return agent.max_iterations
                case "/effort":
                    return effort.state.to_dict()
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
            child_system = _build_runtime_system_prompt(
                tool_root,
                child_transport.model,
                child_tools,
                agents_text,
                effective_username=effective_username,
                effort=effort,
                parent_peer_id=None if foreground else parent_peer_id,
                sandbox_note=sandbox_note,
                model_context=model_context,
            )
            return agent.copy(
                transport=child_transport,
                system=child_system,
                tools=child_tools,
                max_iterations=agent.max_iterations,
                last_iteration_message=LAST_ITERATION_HINT,
                deferred_tool_sink=None if foreground else deferred_tools,
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
            lambda a: _apply_model(
                transport,
                agent,
                tools,
                tool_root,
                agents_text,
                a,
                parent_peer_id,
                sandbox_note=sandbox_note,
                effort=effort,
                effective_username=effective_username,
                model_context=model_context,
            ),
        )
        commands["/effort"] = Command(
            lambda: _show_effort(effort),
            lambda a: _apply_agent_effort(
                effort,
                agent,
                tool_root,
                tools,
                agents_text,
                parent_peer_id,
                sandbox_note,
                a,
                effective_username,
                model_context,
            ),
        )
        commands["/iterations"] = Command(
            lambda: _show_iterations(agent),
            lambda a: _apply_iterations(agent, a),
        )

        loop = asyncio.get_event_loop()
        peer_queue: asyncio.Queue[_IncomingPrompt] = asyncio.Queue()
        pending_peer_prompts: deque[_IncomingPrompt] = deque()

        async def _sequence_incoming(prompt: _IncomingPrompt, *, source: str) -> _IncomingPrompt:
            if prompt.source is not None and prompt.source != source:
                raise ValueError("incoming prompt source cannot change")
            author = prompt.author or prompt.agent_id or "axio-repl"
            if prompt.arrival_seq is not None:
                return dataclasses.replace(prompt, source=source, author=author)
            envelope = await _publish_main_event(InputReceived(text=prompt.text, source=source))
            return dataclasses.replace(prompt, arrival_seq=envelope.seq, source=source, author=author)

        async def _admit_incoming(prompt: _IncomingPrompt, *, source: str) -> None:
            await peer_queue.put(await _sequence_incoming(prompt, source=source))
            _preempt_active_tool_dispatch(f"incoming {source.replace('-', ' ')}")

        async def _deliver_deferred_tool(notification: DeferredToolNotification) -> None:
            text = notification.as_user_text()
            if notification.agent_id != "main" and is_local_background_agent(notification.agent_id):
                await event_hub.publish(
                    InputReceived(text=text, source="deferred-tool"),
                    run_id=notification.run_id,
                    agent_id=notification.agent_id,
                    parent_agent_id="main",
                    turn_id=None,
                    execution_mode=ExecutionMode.BACKGROUND,
                )
                delivered = await enqueue_local_agent_messages(
                    notification.agent_id,
                    (
                        Message(
                            role="user",
                            content=[TextBlock(text=text)],
                            provenance=InputProvenance(
                                human_authored=False,
                                source="deferred-tool",
                                author=notification.tool_name,
                            ),
                        ),
                    ),
                )
                if delivered:
                    return
                text = f"[Deferred result from stopped agent {notification.agent_id}]\n\n{text}"
            await _admit_incoming(
                _IncomingPrompt(text=text, display_text=text, author=notification.tool_name),
                source="deferred-tool",
            )

        deferred_tools = DeferredToolRegistry(_deliver_deferred_tool)
        agent.deferred_tool_sink = deferred_tools

        async def _queue_background_outcome(outcome: TurnOutcome) -> None:
            agent_id = outcome.identity.agent_id
            agent_name, suppress_display = await renderer.take_background_outcome_presentation(
                agent_id,
                outcome.identity.run_id,
                outcome.identity.turn_id,
            )
            identity = format_agent_identity(agent_id, agent_name)
            if outcome.succeeded and outcome.text.strip():
                text = f"Report from background agent {agent_id}:\n\n{outcome.text.strip()}"
                display_text = f"Report from background agent {identity}:\n\n{outcome.text.strip()}"
            elif outcome.succeeded:
                text = f"[agent {agent_id}] finished its turn and is idle."
                display_text = f"[agent {identity}] finished its turn and is idle."
            else:
                detail = outcome.error or "unknown error"
                text = f"[agent {agent_id}] turn failed: {detail}"
                display_text = f"[agent {identity}] turn failed: {detail}"
            await _admit_incoming(
                _IncomingPrompt(
                    text=text,
                    display_text=display_text,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    author=agent_id,
                    suppress_display=suppress_display,
                ),
                source="background-outcome",
            )

        set_background_outcome_handler(_queue_background_outcome)

        stats = _panel.SessionStats()
        renderer = ReplRenderer(
            main_agent_name=AGENT_NAME,
            stats=stats,
            current_model=lambda: transport.model,
            display_mode=DisplayMode.parse(args.agent_actions),
            powerline=args.powerline,
            theme=theme,
            effective_username=effective_username,
        )
        startup_notices: list[str] = []

        async def _render_envelope(envelope: AgentEventEnvelope) -> None:
            await render_runtime_event(renderer, envelope)

        unsubscribe_renderer = event_hub.subscribe(_render_envelope)
        pending_input = PendingInputCoordinator(
            _publish_main_event,
            lambda event, sequence: _publish_main_event(event, reserved_seq=sequence),
        )
        initial_editor_text = recovery.editor_text if recovery is not None else ""
        if recovery is not None:
            remapped_targets: set[str] = set()
            for recovered_input in recovery.pending_inputs:
                target_agent_id = recovered_input.target_agent_id
                if target_agent_id != "main":
                    remapped_targets.add(target_agent_id)
                    target_agent_id = "main"
                await pending_input.admit(recovered_input.text, target_agent_id)
            application_ids = [*recovery.recovery_ids]
            application_ids.extend(
                f"{recovery.source_session_id}:pending:{recovered_input.source_id}"
                for recovered_input in recovery.pending_inputs
            )
            if recovery.editor_text:
                application_ids.append(f"{recovery.source_session_id}:editor")
            await _publish_main_event(EditorSnapshot(recovery.editor_text))
            await _publish_main_event(
                RecoveryApplied(
                    source_session_id=recovery.source_session_id,
                    recovery_ids=tuple(application_ids),
                )
            )
            if recovery.discarded_tail_bytes:
                startup_notices.append(f"Ignored {recovery.discarded_tail_bytes} unterminated journal byte(s).")
            if remapped_targets:
                names = ", ".join(sorted(remapped_targets))
                startup_notices.append(f"Recovered pending input retargeted from unavailable agent(s): {names}.")
        interrupt_queue: asyncio.Queue[_AcceptedInterrupt] = asyncio.Queue()
        shutdown_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        shutdown_queued = False
        foreground_state = ForegroundCoordinatorState()
        exit_arming = ExitArmingState()

        def _queue_shutdown(reason: str) -> None:
            nonlocal shutdown_queued
            if shutdown_queued:
                return
            shutdown_queued = True
            shutdown_queue.put_nowait(reason)

        def _on_sigint() -> None:
            _queue_shutdown("sigint")

        def _on_sigterm() -> None:
            _queue_shutdown("sigterm")

        async def _recall_pending_input() -> str | None:
            batch = await pending_input.recall_all()
            return batch.editor_text if batch is not None else None

        def _queue_escape() -> None:
            target_agent_id = renderer.focused_agent
            request = PromptInterruptRequested(
                target_agent_id=target_agent_id,
                captured_turn_id=(
                    foreground_state.active_turn_id("main") if target_agent_id == "main" else renderer.focused_turn_id
                ),
            )
            task = asyncio.create_task(
                _admit_escape(request),
                name="axio-repl-escape-admission",
            )
            incoming_admission_tasks.add(task)
            task.add_done_callback(incoming_admission_tasks.discard)

        def _handle_empty_eof(now: float) -> bool:
            nonlocal exit_arming
            exit_arming, should_exit = exit_arming.press(now)
            if not should_exit:
                renderer.show_panel("Press Ctrl-D again within 2 seconds to exit.")
            return should_exit

        prompt_session = _panel.make_session(
            lambda: _panel.status_line(
                transport.model,
                stats,
                renderer.action_status(),
                agent_status=renderer.agent_status(),
                panel_message=renderer.panel_message,
            ),
            on_interrupt=_queue_escape,
            on_shutdown=_on_sigint,
            recall_pending=_recall_pending_input,
            on_empty_eof=_handle_empty_eof,
            capture_target=lambda: renderer.focused_agent,
            reserve_sequence=event_hub.reserve_sequence,
            theme=theme,
        )
        setattr(prompt_session, "_axio_terminal_reset", theme.reset)
        input_prompt_factory = _panel.make_prompt_factory(
            effective_username,
            powerline=args.powerline,
            theme=theme,
        )
        terminal = TerminalUI(prompt_session)
        terminal_started = False
        shutdown_reason = "complete"
        prompt_task: asyncio.Task[TurnOutcome] | None = None
        foreground_task: asyncio.Task[None] | None = None
        input_task: asyncio.Task[InputSubmitted] | None = None
        inbox_task: asyncio.Task[_IncomingPrompt] | None = None
        interrupt_task: asyncio.Task[_AcceptedInterrupt] | None = None
        terminal_failure_task: asyncio.Task[None] | None = None
        shutdown_task: asyncio.Task[str] | None = None
        main_status = TurnStatus.SUCCEEDED
        # Lets monitor() see messages that arrived but have not been read:
        # they cannot be delivered until the current turn finishes.
        set_pending_message_probe(lambda: _pending_prompt_count(peer_queue, pending_peer_prompts, inbox_task))
        peer_server: PeerServer | None = None
        incoming_admission_tasks: set[asyncio.Task[None]] = set()

        async def _admit_escape(request: PromptInterruptRequested) -> None:
            envelope = await _publish_main_event(
                InterruptionRequested(
                    target_agent_id=request.target_agent_id,
                    captured_turn_id=request.captured_turn_id,
                )
            )
            if (
                request.target_agent_id == "main"
                and request.captured_turn_id is not None
                and request.captured_turn_id == foreground_state.active_turn_id("main")
                and prompt_task is not None
                and not prompt_task.done()
                and prompt_task.cancelling() == 0
            ):
                pending_interrupt_turns.add(request.captured_turn_id)
                deferred_tools.request_preemption(request.captured_turn_id)
                cancel_task_once(prompt_task)
            await interrupt_queue.put(_AcceptedInterrupt(envelope.seq, request))

        def _queue_notification(text: str) -> None:
            task = asyncio.create_task(
                _admit_incoming(_IncomingPrompt(text=text, author="axio"), source="notification"),
                name="axio-repl-notification-admission",
            )
            incoming_admission_tasks.add(task)
            task.add_done_callback(incoming_admission_tasks.discard)

        async def _on_peer_message(message: PeerMessage) -> None:
            await _admit_incoming(_peer_incoming_prompt(message), source="peer")

        interrupted_partials: dict[str, str] = {}
        pending_interrupt_turns: set[str] = set()
        tool_preemption_reasons: dict[str, str] = {}

        async def _run_turn(
            prompt: str | None = None,
            *,
            messages: tuple[Message, ...] = (),
            source: str,
            on_input_committed: Callable[[], Awaitable[None]] | None = None,
            source_input_ids: tuple[str | None, ...] | None = None,
        ) -> None:
            nonlocal foreground_state, main_status, prompt_task, shutdown_reason
            if (prompt is None) == (not messages):
                raise ValueError("exactly one of prompt or messages is required")
            identity = new_turn_identity(
                agent_id="main",
                parent_agent_id=None,
                execution_mode=ExecutionMode.FOREGROUND,
                run_id=main_run_id,
                context_id=ctx.session_id,
            )
            foreground_state = foreground_state.start("main", identity.turn_id)
            if prompt is not None:
                await event_hub.publish_for(identity, InputReceived(text=prompt, source=source))
                turn = observe_agent_turn(
                    agent=agent,
                    context=ctx,
                    prompt=prompt,
                    identity=identity,
                    hub=event_hub,
                    provenance=InputProvenance(human_authored=True, source=source, author="human"),
                )
            else:
                turn = run_prompt_messages(
                    agent,
                    ctx,
                    messages,
                    event_hub,
                    main_run_id,
                    source=source,
                    identity=identity,
                    on_input_committed=on_input_committed,
                    source_input_ids=source_input_ids,
                )
            prompt_task = asyncio.create_task(turn)
            try:
                if terminal_failure_task is None:
                    outcome = await prompt_task
                else:
                    done, _ = await asyncio.wait(
                        {prompt_task, terminal_failure_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if terminal_failure_task in done:
                        shutdown_reason = "terminal_failure"
                        if not prompt_task.done():
                            cancel_task_once(prompt_task)
                            await asyncio.gather(prompt_task, return_exceptions=True)
                        await terminal_failure_task
                        raise RuntimeError("terminal failure monitor stopped unexpectedly")
                    outcome = prompt_task.result()
                main_status = outcome.status
            except asyncio.CancelledError:
                main_status = TurnStatus.CANCELLED
                if prompt_task is not None and not prompt_task.done():
                    cancel_task_once(prompt_task)
                    await asyncio.gather(prompt_task, return_exceptions=True)
                partial = renderer.take_pending_text("main")
                reason = tool_preemption_reasons.pop(identity.turn_id, None)
                _retain_interrupted_partial(
                    interrupted_partials,
                    pending_interrupt_turns,
                    turn_id=identity.turn_id,
                    partial=partial,
                    preemption_reason=reason,
                )
                if reason is None:
                    renderer.show_panel("Main turn interrupted.")
                else:
                    renderer.show_panel(f"Main turn preempted for {reason}; unfinished tools continue.")
            finally:
                tool_preemption_reasons.pop(identity.turn_id, None)
                prompt_task = None
                foreground_state = foreground_state.complete("main", identity.turn_id)
                await renderer.mark_idle()

        async def _interrupt_focused_agent(agent_id: str, turn_id: str) -> bool:
            if not interrupt_local_agent_turn(agent_id, turn_id):
                return False
            partial = renderer.take_pending_text(agent_id)
            if partial:
                interrupted_partials[turn_id] = partial
            await renderer.notice("[interrupted]")
            await renderer.mark_idle()
            return True

        async def _run_one_shot_background_agents() -> None:
            records = local_background_agent_records()
            if not records:
                return
            await renderer.notice(f"[waiting for {len(records)} background agent(s)]")
            await wait_local_background_agents_idle([record.id for record in records])
            await renderer.mark_idle()

        def _collect_queued(first: _IncomingPrompt) -> list[_IncomingPrompt]:
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

        async def _run_peer_prompts(prompts: list[_IncomingPrompt]) -> None:
            for prompt in prompts:
                await renderer.incoming(
                    prompt.display_text or prompt.text,
                    agent_id=prompt.agent_id,
                    agent_name=prompt.agent_name,
                    suppress_display=prompt.suppress_display,
                )
            messages: list[Message] = []
            for prompt in prompts:
                if prompt.source is None:
                    raise RuntimeError("incoming prompt reached delivery without a source")
                messages.append(
                    Message(
                        role="user",
                        content=[TextBlock(text=prompt.text)],
                        provenance=InputProvenance(
                            human_authored=False,
                            source=prompt.source,
                            author=prompt.author or "axio-repl",
                        ),
                    )
                )
            await _run_turn(messages=tuple(messages), source="internal")

        async def _run_peer_turn(first: _IncomingPrompt) -> None:
            await _run_peer_prompts(_collect_queued(first))

        async def _drain_peer_messages() -> None:
            try:
                first = peer_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await _run_peer_turn(first)

        def _preempt_active_tool_dispatch(reason: str) -> None:
            current = prompt_task
            active_turn_id = foreground_state.active_turn_id("main")
            if (
                current is not None
                and not current.done()
                and current.cancelling() == 0
                and deferred_tools.has_active_dispatch(active_turn_id)
            ):
                if active_turn_id is not None:
                    tool_preemption_reasons.setdefault(active_turn_id, reason)
                deferred_tools.request_preemption(active_turn_id)
                cancel_task_once(current)

        def _request_target_tool_preemption(target_agent_id: str) -> str | None:
            if target_agent_id == "main":
                _preempt_active_tool_dispatch("queued user input")
                return None
            turn_id = renderer.focused_turn_id if renderer.focused_agent == target_agent_id else None
            if deferred_tools.request_preemption(turn_id):
                return turn_id
            return None

        def _preempt_background_tool_dispatch(agent_id: str, turn_id: str | None) -> None:
            if turn_id is not None and deferred_tools.request_preemption(turn_id):
                interrupt_local_agent_turn(agent_id, turn_id)

        def _on_tool_dispatch_started(agent_id: str, turn_id: str | None) -> None:
            if agent_id == "main":
                if turn_id != foreground_state.active_turn_id("main"):
                    return
                if pending_input.pending_count:
                    _preempt_active_tool_dispatch("queued user input")
                elif pending_peer_prompts or not peer_queue.empty():
                    _preempt_active_tool_dispatch("incoming message")
                return
            has_queued_input = any(
                (entry.status is PendingInputStatus.PENDING and entry.intended_target_agent_id == agent_id)
                or (entry.status is PendingInputStatus.CLAIMED and entry.claimed_target_agent_id == agent_id)
                for entry in pending_input.state.entries
            )
            if has_queued_input and deferred_tools.request_preemption(turn_id):
                if turn_id is not None:
                    interrupt_local_agent_turn(agent_id, turn_id)

        deferred_tools.set_dispatch_started_handler(_on_tool_dispatch_started)
        set_background_input_admitted_handler(_preempt_background_tool_dispatch)

        loop.add_signal_handler(signal.SIGINT, _on_sigint)
        loop.add_signal_handler(signal.SIGTERM, _on_sigterm)

        try:
            try:
                peer_server = await PeerServer(
                    _peer_name(root),
                    kind="axio-repl",
                    handler=_on_peer_message,
                    cwd=str(root),
                ).start()
                parent_peer_id = peer_server.id
                agent.system = _build_runtime_system_prompt(
                    tool_root,
                    transport.model,
                    tools,
                    agents_text,
                    effective_username=effective_username,
                    effort=effort,
                    parent_peer_id=parent_peer_id,
                    sandbox_note=sandbox_note,
                    model_context=model_context,
                )
                notify.add_listener(
                    peer_server.id,
                    _queue_notification,
                )
            except OSError as exc:
                startup_notices.append(f"Peer messaging disabled: {exc}")
                notify.add_listener(None, _queue_notification)

            if args.prompt:
                await _run_turn(args.prompt, source="one-shot")
                await _run_one_shot_background_agents()
                renderer.set_focus("main")
                await _drain_peer_messages()
                return

            await terminal.start()
            terminal_started = True
            terminal_failure_task = asyncio.create_task(
                terminal.wait_failed(),
                name="axio-repl-terminal-failure",
            )
            agent_commands = ["/agents", "/agent-actions", "/agent-focus", "/agent-interrupt", "/agent-stop"]
            commands_list = ", ".join(["/help", *commands, *agent_commands, "/quit"])
            label = getattr(transport, "name", "unknown")
            startup_panel = [
                f"REPL ready ({label}). Enter queues, Esc interrupts without changing the editor, Up recalls pending.",
                f"Tools: {sandbox_desc}",
                f"Commands: {commands_list}",
            ]
            if session_journal is not None:
                startup_panel.append(f"Session log: {session_journal.events_path}")
            startup_panel.extend(startup_notices)
            renderer.show_panel("\n".join(startup_panel))

            ready_claims: deque[_ReadyClaim] = deque()
            settling_interrupts: deque[_InterruptTransaction] = deque()
            input_closed = False

            async def _dispatch_command(user_input: str) -> tuple[bool, bool]:
                lowered = user_input.lower()
                if lowered in {"/quit", "/exit", "/q"}:
                    return True, True
                if lowered == "/help":
                    tool_list = ", ".join(t.name for t in tools)
                    _command_print(f"Type your request. Tools: {tool_list}")
                    _command_print(f"Commands: {commands_list}")
                    return True, False
                if lowered == "/agents":
                    _show_agents(renderer)
                    return True, False
                if lowered == "/agent-actions" or lowered.startswith("/agent-actions "):
                    raw_mode = user_input[len("/agent-actions") :].strip()
                    await _handle_agent_actions(renderer, raw_mode, _publish_main_event)
                    return True, False
                if lowered == "/agent-focus" or lowered.startswith("/agent-focus "):
                    arg = user_input[len("/agent-focus") :].strip()
                    if not arg:
                        _show_agents(renderer)
                        return True, False
                    agent_id = _resolve_local_agent_id(arg)
                    if agent_id is None:
                        _command_print(f"No local agent matching {arg!r}")
                        return True, False
                    renderer.set_focus(agent_id)
                    _command_print(f"Focused agent: {BOLD}{agent_id}{RESET}")
                    await _publish_main_event(
                        ConfigurationChanged(name="input_target", value=agent_id, source="interactive")
                    )
                    return True, False
                if lowered == "/agent-stop" or lowered.startswith("/agent-stop "):
                    arg = user_input[len("/agent-stop") :].strip() or renderer.focused_agent
                    agent_id = _resolve_command_agent_id(arg, renderer)
                    if agent_id is None:
                        _command_print(f"No local agent matching {arg!r}")
                        return True, False
                    if agent_id == "main":
                        _command_print("Use /quit to exit the main REPL agent.")
                        return True, False
                    _command_print(await stop_agent(agent_id, reason="user requested stop"))
                    if renderer.focused_agent == agent_id:
                        await renderer.mark_idle()
                        renderer.set_focus("main")
                        _command_print(f"Focused agent: {BOLD}main{RESET}")
                    return True, False
                if lowered == "/agent-interrupt" or lowered.startswith("/agent-interrupt "):
                    arg = user_input[len("/agent-interrupt") :].strip() or renderer.focused_agent
                    agent_id = _resolve_command_agent_id(arg, renderer)
                    if agent_id is None:
                        _command_print(f"No local agent matching {arg!r}")
                        return True, False
                    if agent_id == "main":
                        _command_print("Press Escape while the main agent is streaming to interrupt it.")
                        return True, False
                    _command_print(await interrupt_agent(agent_id, reason="user requested interrupt"))
                    if renderer.focused_agent == agent_id:
                        await renderer.mark_idle()
                    return True, False

                for prefix, cmd in commands.items():
                    if lowered != prefix and not lowered.startswith(prefix + " "):
                        continue
                    cmd_arg = user_input[len(prefix) :].strip() or None
                    if cmd_arg is None:
                        cmd.show()
                    else:
                        previous_value = _command_configuration(prefix)
                        previous_effort = effort.state.to_dict() if prefix == "/model" else None
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
                        if previous_effort is not None and effort.state.to_dict() != previous_effort:
                            await _publish_main_event(
                                ConfigurationChanged(
                                    name="effort",
                                    value=effort.state.to_dict(),
                                    source="interactive",
                                )
                            )
                    return True, False
                return False, False

            async def _dispatch_command_to_panel(user_input: str) -> tuple[bool, bool]:
                with _capture_command_output(theme) as output:
                    handled, should_exit = await _dispatch_command(user_input)
                if output:
                    renderer.show_panel("".join(output))
                return handled, should_exit

            def _can_dispatch_during_foreground(user_input: str) -> bool:
                command = user_input.lower().split(maxsplit=1)[0]
                if command in {"/help", "/agents", "/agent-actions"}:
                    return True
                return command in commands and command == user_input.lower()

            def _is_known_command(user_input: str) -> bool:
                command = user_input.lower().split(maxsplit=1)[0]
                return command in {
                    "/quit",
                    "/exit",
                    "/q",
                    "/help",
                    "/agents",
                    "/agent-actions",
                    "/agent-focus",
                    "/agent-interrupt",
                    "/agent-stop",
                    *commands,
                }

            async def _admit_editor_submission(
                text: str,
                target_agent_id: str,
                reserved_seq: int | None,
            ) -> InputSubmitted:
                renderer.clear_panel()
                if _is_known_command(text):
                    if reserved_seq is not None:
                        await event_hub.discard_reserved_sequence(reserved_seq)
                    return InputSubmitted(
                        text=text,
                        target_agent_id=target_agent_id,
                        disposition=SubmissionDisposition.COMMAND,
                        arrival_seq=reserved_seq,
                    )
                if pending_input.pending_count >= MAX_PENDING_INPUTS:
                    await _publish_main_event(
                        InputReceived(text=text, source="interactive-retained"),
                        reserved_seq=reserved_seq,
                    )
                    return InputSubmitted(
                        text=text,
                        target_agent_id=target_agent_id,
                        disposition=SubmissionDisposition.RETAINED,
                    )
                entry = await pending_input.admit(text, target_agent_id, reserved_seq=reserved_seq)
                child_turn_to_interrupt = _request_target_tool_preemption(target_agent_id)
                if child_turn_to_interrupt is not None:
                    interrupt_local_agent_turn(target_agent_id, child_turn_to_interrupt)
                return InputSubmitted(
                    text=text,
                    target_agent_id=target_agent_id,
                    disposition=SubmissionDisposition.PENDING,
                    input_id=entry.id,
                    arrival_seq=entry.arrival_seq,
                )

            async def _run_targeted_claim(ready: _ReadyClaim) -> None:
                target_agent_id = ready.batch.target_agent_id

                async def mark_delivered() -> None:
                    await pending_input.mark_delivered(ready.batch)

                if target_agent_id == "main":
                    await _run_turn(
                        messages=ready.messages,
                        source=ready.source,
                        on_input_committed=mark_delivered,
                        source_input_ids=ready.source_input_ids,
                    )
                    return
                if is_local_background_agent(target_agent_id):
                    delivered = await enqueue_local_agent_messages(
                        target_agent_id,
                        ready.messages,
                        on_input_committed=mark_delivered,
                        source_input_ids=ready.source_input_ids,
                    )
                    if not delivered:
                        await renderer.notice(
                            f"Agent {target_agent_id!r} is no longer running; claimed input remains recoverable."
                        )
                        if renderer.focused_agent == target_agent_id:
                            renderer.set_focus("main")
                    return
                await renderer.notice(
                    f"Agent {target_agent_id!r} is no longer local; claimed input remains recoverable."
                )
                if renderer.focused_agent == target_agent_id:
                    renderer.set_focus("main")

            async def _finalize_interrupt(transaction: _InterruptTransaction) -> None:
                request = transaction.request
                partial = (
                    interrupted_partials.pop(request.captured_turn_id, "")
                    if request.captured_turn_id is not None
                    else ""
                )
                if request.captured_turn_id is not None:
                    pending_interrupt_turns.discard(request.captured_turn_id)
                claimed_ids = (
                    tuple(entry.id for entry in transaction.claimed.entries) if transaction.claimed is not None else ()
                )
                barrier = await _publish_main_event(
                    InterruptionCommitted(
                        request_seq=transaction.request_seq,
                        target_agent_id=request.target_agent_id,
                        captured_turn_id=request.captured_turn_id,
                        reason="escape",
                        claimed_input_ids=claimed_ids,
                        partial_text=partial,
                    )
                )
                while True:
                    try:
                        pending_peer_prompts.append(peer_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                ordered_peers = sorted(
                    pending_peer_prompts,
                    key=lambda prompt: prompt.arrival_seq if prompt.arrival_seq is not None else sys.maxsize,
                )
                pending_peer_prompts.clear()
                pending_peer_prompts.extend(ordered_peers)
                turn_label = request.captured_turn_id or "idle"
                notice = Message(
                    role="user",
                    content=[TextBlock(text=f"[Turn {turn_label} was interrupted by Escape.]")],
                    provenance=InputProvenance(
                        human_authored=False,
                        source="interrupt",
                        author="axio-repl",
                    ),
                )
                arrivals = list(claim_batch_arrivals(transaction.claimed)) if transaction.claimed is not None else []
                if request.target_agent_id == "main":
                    retained_peers: deque[_IncomingPrompt] = deque()
                    while pending_peer_prompts:
                        prompt = pending_peer_prompts.popleft()
                        if prompt.arrival_seq is not None and prompt.arrival_seq <= barrier.seq:
                            await renderer.incoming(
                                prompt.display_text or prompt.text,
                                agent_id=prompt.agent_id,
                                agent_name=prompt.agent_name,
                                suppress_display=prompt.suppress_display,
                            )
                            arrivals.append(
                                ContextArrival(
                                    seq=prompt.arrival_seq,
                                    target_agent_id="main",
                                    message=Message(
                                        role="user",
                                        content=[TextBlock(text=prompt.text)],
                                        provenance=InputProvenance(
                                            human_authored=False,
                                            source=prompt.source or "peer",
                                            author=prompt.author or "axio-repl",
                                        ),
                                    ),
                                    source=prompt.source or "peer",
                                )
                            )
                        else:
                            retained_peers.append(prompt)
                    pending_peer_prompts.extend(retained_peers)
                arrivals.append(
                    ContextArrival(
                        seq=transaction.request_seq,
                        target_agent_id=request.target_agent_id,
                        message=notice,
                        source="interrupt",
                    )
                )
                ordered = ordered_arrivals(tuple(arrivals), request.target_agent_id, through_seq=barrier.seq)
                messages = tuple(arrival.message for arrival in ordered)
                source_input_ids = tuple(arrival.source_input_id for arrival in ordered)
                if transaction.claimed is None:
                    if request.target_agent_id == "main":
                        await ctx.append_many(list(messages))
                    elif is_local_background_agent(request.target_agent_id):
                        delivered = await enqueue_local_agent_context(
                            request.target_agent_id,
                            messages,
                            wait=True,
                        )
                        if not delivered:
                            await renderer.notice(
                                f"Agent {request.target_agent_id!r} stopped before interruption context commit."
                            )
                    return
                ready_claims.append(
                    _ReadyClaim(
                        batch=transaction.claimed,
                        messages=messages,
                        source_input_ids=source_input_ids,
                        source="interactive-interrupt",
                    )
                )

            async def _accept_interrupt(accepted: _AcceptedInterrupt) -> None:
                nonlocal foreground_state
                request = accepted.request
                foreground_state, should_apply = foreground_state.request_interrupt(
                    request.target_agent_id,
                    request.captured_turn_id,
                )
                if not should_apply:
                    return
                claimed = await pending_input.claim_all_for_interrupt(request.target_agent_id)
                if claimed is not None:
                    _panel.commit_history(prompt_session, tuple(entry.text for entry in claimed.entries))
                transaction = _InterruptTransaction(accepted.request_seq, request, claimed)

                if request.target_agent_id != "main":
                    if request.captured_turn_id is not None:
                        deferred_tools.request_preemption(request.captured_turn_id)
                        await _interrupt_focused_agent(request.target_agent_id, request.captured_turn_id)
                    await _finalize_interrupt(transaction)
                    return

                current_prompt_task = prompt_task
                current_matches = (
                    request.captured_turn_id is not None
                    and request.captured_turn_id == foreground_state.active_turn_id("main")
                    and foreground_task is not None
                    and not foreground_task.done()
                )
                if current_matches:
                    if current_prompt_task is not None:
                        cancel_task_once(current_prompt_task)
                    settling_interrupts.append(transaction)
                    return
                await _finalize_interrupt(transaction)

            async def _start_next_foreground() -> bool:
                nonlocal foreground_task
                while foreground_task is None:
                    oldest_user_seq = (
                        pending_input.state.pending[0].arrival_seq if pending_input.state.pending else None
                    )
                    oldest_peer_seq = pending_peer_prompts[0].arrival_seq if pending_peer_prompts else None
                    oldest_command_seq = pending_commands[0].arrival_seq if pending_commands else None
                    oldest_ready_seq = ready_claims[0].batch.entries[0].arrival_seq if ready_claims else None
                    command_precedes_dialog = oldest_command_seq is not None and all(
                        sequence is None or oldest_command_seq < sequence
                        for sequence in (oldest_user_seq, oldest_peer_seq, oldest_ready_seq)
                    )
                    if pending_commands and (
                        command_precedes_dialog
                        or (oldest_user_seq is None and oldest_peer_seq is None and oldest_ready_seq is None)
                    ):
                        command = pending_commands.popleft()
                        handled, should_exit = await _dispatch_command_to_panel(command.text)
                        if not handled:
                            raise RuntimeError("queued REPL command has no dispatcher")
                        if should_exit:
                            return True
                        continue
                    run_peer = (
                        not ready_claims
                        and oldest_peer_seq is not None
                        and all(
                            sequence is None or oldest_peer_seq < sequence
                            for sequence in (oldest_user_seq, oldest_command_seq)
                        )
                    )
                    if run_peer:
                        prompts: list[_IncomingPrompt] = []
                        while pending_peer_prompts:
                            candidate = pending_peer_prompts[0]
                            if candidate.arrival_seq is None:
                                break
                            if oldest_user_seq is not None and candidate.arrival_seq > oldest_user_seq:
                                break
                            if oldest_command_seq is not None and candidate.arrival_seq > oldest_command_seq:
                                break
                            prompts.append(pending_peer_prompts.popleft())
                        foreground_task = asyncio.create_task(
                            _run_peer_prompts(prompts),
                            name="axio-repl-peer-turn",
                        )
                        return False
                    ready = ready_claims.popleft() if ready_claims else None
                    if ready is None:
                        claimed = await pending_input.claim_oldest()
                        if claimed is None:
                            return False
                        _panel.commit_history(prompt_session, tuple(entry.text for entry in claimed.entries))
                        messages = tuple(arrival.message for arrival in claim_batch_arrivals(claimed))
                        source_input_ids = tuple(entry.id for entry in claimed.entries)
                        ready = _ReadyClaim(
                            batch=claimed,
                            messages=messages,
                            source_input_ids=source_input_ids,
                            source="interactive",
                        )

                    if len(ready.batch.entries) == 1:
                        handled, should_exit = await _dispatch_command_to_panel(ready.batch.entries[0].text)
                    else:
                        handled, should_exit = False, False
                    if should_exit:
                        await pending_input.mark_delivered(ready.batch)
                        return True
                    if handled:
                        await pending_input.mark_delivered(ready.batch)
                        continue
                    foreground_task = asyncio.create_task(
                        _run_targeted_claim(ready),
                        name="axio-repl-interactive-turn",
                    )
                return False

            pending_commands: deque[InputSubmitted] = deque()

            while True:
                if foreground_task is None:
                    if (
                        input_closed
                        and pending_input.pending_count == 0
                        and not ready_claims
                        and not pending_commands
                        and not settling_interrupts
                        and not pending_peer_prompts
                        and peer_queue.empty()
                        and (inbox_task is None or not inbox_task.done())
                        and (interrupt_task is None or not interrupt_task.done())
                        and not incoming_admission_tasks
                    ):
                        break
                    if await _start_next_foreground():
                        break

                if input_task is None and not input_closed:
                    input_task = asyncio.create_task(
                        _read_input_async(
                            prompt_session,
                            renderer,
                            _on_sigint,
                            _admit_editor_submission,
                            initial_editor_text,
                            prompt_factory=input_prompt_factory,
                        )
                    )
                    initial_editor_text = ""
                if inbox_task is None:
                    inbox_task = asyncio.create_task(peer_queue.get())
                if interrupt_task is None:
                    interrupt_task = asyncio.create_task(interrupt_queue.get())
                if shutdown_task is None:
                    shutdown_task = asyncio.create_task(shutdown_queue.get())

                wait_tasks: set[asyncio.Task[Any]] = {inbox_task, interrupt_task, shutdown_task}
                if input_task is not None:
                    wait_tasks.add(input_task)
                if foreground_task is not None:
                    wait_tasks.add(foreground_task)
                if terminal_failure_task is not None:
                    wait_tasks.add(terminal_failure_task)
                admission_waiters = tuple(incoming_admission_tasks)
                wait_tasks.update(admission_waiters)
                done, _ = await asyncio.wait(
                    wait_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if terminal_failure_task is not None and terminal_failure_task in done:
                    shutdown_reason = "terminal_failure"
                    await terminal_failure_task
                    raise RuntimeError("terminal failure monitor stopped unexpectedly")

                for admission_task in admission_waiters:
                    if admission_task in done:
                        admission_task.result()

                if shutdown_task in done:
                    shutdown_reason = shutdown_task.result()
                    main_status = TurnStatus.CANCELLED
                    shutdown_task = None
                    break

                if inbox_task in done:
                    pending_peer_prompts.append(inbox_task.result())
                    ordered_peers = sorted(
                        pending_peer_prompts,
                        key=lambda prompt: prompt.arrival_seq if prompt.arrival_seq is not None else sys.maxsize,
                    )
                    pending_peer_prompts.clear()
                    pending_peer_prompts.extend(ordered_peers)
                    inbox_task = None

                if foreground_task is not None and foreground_task in done:
                    foreground_task.result()
                    foreground_task = None
                    while settling_interrupts:
                        await _finalize_interrupt(settling_interrupts.popleft())

                if interrupt_task in done:
                    await _accept_interrupt(interrupt_task.result())
                    interrupt_task = None

                if input_task is None or input_task not in done:
                    continue

                try:
                    submitted = input_task.result()
                except EOFError:
                    print()
                    shutdown_reason = "double_eof"
                    input_closed = True
                    continue
                finally:
                    input_task = None

                user_input = submitted.text
                if submitted.disposition is SubmissionDisposition.RETAINED:
                    initial_editor_text = user_input
                    await renderer.notice(
                        f"[pending input limit ({MAX_PENDING_INPUTS}) reached; press Up to recall queued input]"
                    )
                    continue
                if submitted.disposition is SubmissionDisposition.COMMAND:
                    if foreground_task is not None and not _can_dispatch_during_foreground(user_input):
                        pending_commands.append(submitted)
                        _panel.complete_submission(
                            prompt_session,
                            _panel.editor_text(prompt_session),
                            clear_editor=True,
                        )
                        renderer.show_panel(f"Queued command: {user_input}")
                        continue
                    handled, should_exit = await _dispatch_command_to_panel(user_input)
                    if not handled:
                        raise RuntimeError("accepted REPL command has no dispatcher")
                    if should_exit:
                        _panel.complete_submission(
                            prompt_session,
                            _panel.editor_text(prompt_session),
                            clear_editor=True,
                        )
                        shutdown_reason = "command"
                        break
                    _panel.complete_submission(
                        prompt_session,
                        _panel.editor_text(prompt_session),
                        clear_editor=True,
                    )
                    continue
        except asyncio.CancelledError:
            main_status = TurnStatus.CANCELLED
            shutdown_reason = "outer_cancellation"
            raise
        except BaseException:
            main_status = TurnStatus.FAILED
            if shutdown_reason == "complete":
                shutdown_reason = "internal_failure"
            raise
        finally:
            shutdown_turn_id = foreground_state.active_turn_id("main")
            foreground_state = foreground_state.request_shutdown(shutdown_reason)
            await _cancel_and_settle_tasks(
                foreground_task,
                input_task,
                inbox_task,
                interrupt_task,
                terminal_failure_task,
                shutdown_task,
                *tuple(incoming_admission_tasks),
            )
            try:
                deferred_snapshots = deferred_tools.snapshots()
                if terminal_started:
                    partial_text = (
                        interrupted_partials.pop(shutdown_turn_id, "") if shutdown_turn_id is not None else ""
                    )
                    pending_ids = tuple(
                        entry.id
                        for entry in pending_input.state.entries
                        if entry.status in {PendingInputStatus.PENDING, PendingInputStatus.CLAIMED}
                    )
                    deferred_ids = tuple(
                        tool_use_id for snapshot in deferred_snapshots for tool_use_id in snapshot.tool_use_ids
                    )
                    deferred_agent_ids = tuple(
                        snapshot.agent_id for snapshot in deferred_snapshots for _ in snapshot.tool_use_ids
                    )
                    deferred_turn_ids = tuple(
                        snapshot.turn_id for snapshot in deferred_snapshots for _ in snapshot.tool_use_ids
                    )
                    deferred_phases = tuple(
                        snapshot.phase.value for snapshot in deferred_snapshots for _ in snapshot.tool_use_ids
                    )
                    await _publish_main_event(EditorSnapshot(_panel.editor_text(prompt_session)))
                    await _publish_main_event(
                        ShutdownRecorded(
                            reason=shutdown_reason,
                            pending_input_ids=pending_ids,
                            deferred_tool_use_ids=deferred_ids,
                            interrupted_turn_id=shutdown_turn_id,
                            partial_text=partial_text,
                            deferred_tool_agent_ids=deferred_agent_ids,
                            deferred_tool_turn_ids=deferred_turn_ids,
                            deferred_tool_phases=deferred_phases,
                        )
                    )
                await stop_local_background_agents()
                await deferred_tools.close()
                await ctx.close()
                await _publish_main_event(AgentStopped(status=main_status))
                notify.remove_listener(peer_server.id if peer_server is not None else None)
                if peer_server is not None:
                    await peer_server.close()
            finally:
                unsubscribe_renderer()
                set_background_outcome_handler(None)
                set_background_input_admitted_handler(None)
                set_run_agent_factory(None)
                set_spawn_agent_factory(None)
                set_session_event_hub(None)
                set_pending_message_probe(None)
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)
                await terminal.close()
                foreground_state = foreground_state.mark_stopped()


def main_sync() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main_sync()
