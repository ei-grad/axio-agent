"""Bounded presentation queues for actions from non-active agents."""

from __future__ import annotations

import json
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import StrEnum

from axio.events import Error, SessionEndEvent, ToolInputDelta, ToolOutputDelta, ToolResult, ToolUseStart
from axio_tools_agents.runtime import AgentStarted, AgentStopped, RuntimeEvent, TurnFinished, TurnStarted

from axio_repl._powerline import action_frame_footer, action_frame_header
from axio_repl._theme import DEFAULT_THEME, RESET, TerminalTheme

_MAX_DISPLAY_COUNT = 999_999_999


class DisplayMode(StrEnum):
    ACTIVE_ONLY = "off"
    ALL_ACTIONS = "on"

    @classmethod
    def parse(cls, value: str) -> DisplayMode:
        normalized = value.strip().lower()
        aliases = {
            "active": cls.ACTIVE_ONLY,
            "active-only": cls.ACTIVE_ONLY,
            "all": cls.ALL_ACTIONS,
            "all-actions": cls.ALL_ACTIONS,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError("agent actions must be 'on' or 'off'") from exc


@dataclass(frozen=True, slots=True)
class DisplayModeChange:
    previous: DisplayMode
    current: DisplayMode
    discarded_frames: int = 0
    discarded_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ActionFrame:
    agent_id: str
    kind: str
    body: str
    sequence: int
    critical: bool = False
    agent_name: str | None = None
    powerline: bool = False
    theme: TerminalTheme = DEFAULT_THEME

    def render(self) -> str:
        identity = format_agent_identity(self.agent_id, self.agent_name)
        kind = sanitize_terminal_text(self.kind).replace("\n", " ")[:40]
        body = sanitize_terminal_text(self.body).rstrip("\n")
        if self.powerline:
            return (
                f"{RESET}\n{action_frame_header(identity, kind, self.theme)}\n{body}\n"
                f"{action_frame_footer(identity, self.theme)}\n{RESET}\n"
            )
        style = self.theme.action.ansi
        if style:
            return (
                f"{RESET}\n{style}── agent {identity} · {kind} ──{RESET}\n{body}\n"
                f"{style}── /agent {identity} ──{RESET}\n{RESET}\n"
            )
        return f"{RESET}\n── agent {identity} · {kind} ──\n{body}\n── /agent {identity} ──\n{RESET}\n"


@dataclass(slots=True)
class _ToolAction:
    name: str
    agent_name: str | None = None
    arguments: list[str] = field(default_factory=list)
    arguments_bytes: int = 0
    call_emitted: bool = False
    output: dict[str, str] = field(default_factory=dict)
    output_bytes: int = 0
    saw_output: bool = False

    @property
    def retained_bytes(self) -> int:
        agent_name_bytes = len(self.agent_name.encode("utf-8")) if self.agent_name is not None else 0
        return len(self.name.encode("utf-8")) + agent_name_bytes + self.arguments_bytes + self.output_bytes


@dataclass(slots=True)
class _AgentCollector:
    identity_bytes: int
    tools: OrderedDict[str, _ToolAction] = field(default_factory=OrderedDict)


def sanitize_terminal_text(value: object) -> str:
    """Remove terminal control sequences while preserving lines and tabs."""
    return _sanitize_terminal(value, preserve_layout=True)


def sanitize_identity_component(value: object) -> str:
    """Return one line of terminal-safe identity text."""
    return " ".join(_sanitize_terminal(value, preserve_layout=False).split())


def _sanitize_terminal(value: object, *, preserve_layout: bool) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    result: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        codepoint = ord(character)
        replacement = "" if preserve_layout else " "
        if character == "\x1b":
            result.append(replacement)
            index = _consume_escape_sequence(text, index)
            continue
        if codepoint == 0x9B:
            result.append(replacement)
            index = _consume_csi(text, index + 1)
            continue
        if codepoint in {0x90, 0x9D, 0x9E, 0x9F}:
            result.append(replacement)
            index = _consume_control_string(text, index + 1)
            continue
        if codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
            if preserve_layout and character in {"\n", "\t"}:
                result.append(character)
            else:
                result.append(replacement)
            index += 1
            continue
        result.append(character)
        index += 1
    return "".join(result)


def _consume_escape_sequence(text: str, index: int) -> int:
    next_index = index + 1
    if next_index >= len(text):
        return next_index
    introducer = text[next_index]
    if introducer == "[":
        return _consume_csi(text, next_index + 1)
    if introducer in {
        "]",
        "P",
        "^",
        "_",
    }:
        return _consume_control_string(text, next_index + 1)
    return min(len(text), next_index + 1)


def _consume_csi(text: str, index: int) -> int:
    while index < len(text):
        if 0x40 <= ord(text[index]) <= 0x7E:
            return index + 1
        index += 1
    return index


def _consume_control_string(text: str, index: int) -> int:
    while index < len(text):
        if text[index] in {"\x07", "\x9c"}:
            return index + 1
        if text[index] == "\x1b" and index + 1 < len(text) and text[index + 1] == "\\":
            return index + 2
        index += 1
    return index


def normalize_agent_name(agent_name: str | None) -> str | None:
    if agent_name is None:
        return None
    return _fit_utf8(sanitize_identity_component(agent_name), 80, suffix="…").strip() or None


def format_agent_identity(agent_id: str, agent_name: str | None = None) -> str:
    """Return a terminal-safe identity whose authoritative id is always visible."""
    clean_id = _fit_utf8(sanitize_identity_component(agent_id), 80, suffix="…") or "unknown"
    clean_name = normalize_agent_name(agent_name)
    if clean_name is None:
        return clean_id
    return f"{clean_name} ({clean_id})"


def _fit_utf8(text: str, limit: int, *, suffix: str = "\n[… truncated]") -> str:
    data = text.encode("utf-8")
    if limit <= 0:
        return ""
    if len(data) <= limit:
        return text
    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) >= limit:
        return suffix_bytes[:limit].decode("utf-8", errors="ignore")
    available = max(0, limit - len(suffix_bytes))
    prefix = data[:available].decode("utf-8", errors="ignore")
    return prefix + suffix


class ActionMultiplexer:
    """Collect complete action frames and drain them fairly at safe boundaries."""

    def __init__(
        self,
        mode: DisplayMode = DisplayMode.ACTIVE_ONLY,
        *,
        powerline: bool = False,
        theme: TerminalTheme = DEFAULT_THEME,
        max_queued_frames: int = 256,
        max_queued_bytes: int = 256 * 1024,
        max_frames_per_agent: int = 64,
        max_frame_bytes: int = 3072,
        output_chunk_chars: int = 1024,
        max_agents: int = 256,
        max_tools: int = 512,
        max_tools_per_agent: int = 64,
        max_retained_bytes: int = 512 * 1024,
    ) -> None:
        if (
            min(
                max_queued_frames,
                max_queued_bytes,
                max_frames_per_agent,
                max_frame_bytes,
                output_chunk_chars,
                max_agents,
                max_tools,
                max_tools_per_agent,
                max_retained_bytes,
            )
            <= 0
        ):
            raise ValueError("multiplexer limits must be positive")
        largest_suppression = ActionFrame(
            agent_id="all agents",
            kind="suppressed",
            body="999999999+ action frames suppressed; 999999999+ incomplete actions discarded",
            sequence=0,
            critical=True,
            powerline=powerline,
            theme=theme,
        )
        if max_retained_bytes < len(largest_suppression.render().encode("utf-8")):
            raise ValueError("max_retained_bytes is too small for the suppression marker")
        self._mode = mode
        self._powerline = powerline
        self._theme = theme
        self._max_queued_frames = max_queued_frames
        self._max_queued_bytes = max_queued_bytes
        self._max_frames_per_agent = max_frames_per_agent
        self._max_frame_bytes = max_frame_bytes
        self._output_chunk_chars = output_chunk_chars
        self._max_agents = max_agents
        self._max_tools = max_tools
        self._max_tools_per_agent = max_tools_per_agent
        self._max_retained_bytes = max_retained_bytes
        self._collectors: OrderedDict[str, _AgentCollector] = OrderedDict()
        self._tool_order: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._queues: dict[str, deque[ActionFrame]] = {}
        self._round_robin: deque[str] = deque()
        self._scheduled: set[str] = set()
        self._queued_bytes = 0
        self._collector_bytes = 0
        self._sequence = 0
        self._suppressed_frames = 0
        self._suppressed_state = 0

    @property
    def mode(self) -> DisplayMode:
        return self._mode

    @property
    def queued_count(self) -> int:
        return self._frame_count() + int(bool(self._suppressed_frames or self._suppressed_state))

    @property
    def queued_bytes(self) -> int:
        return self._queued_bytes

    @property
    def max_retained_bytes(self) -> int:
        return self._max_retained_bytes

    @property
    def retained_collector_bytes(self) -> int:
        return self._collector_bytes

    @property
    def retained_suppression_bytes(self) -> int:
        if not (self._suppressed_frames or self._suppressed_state):
            return 0
        return self._frame_size(self._make_suppression_frame(sequence=0))

    @property
    def retained_bytes(self) -> int:
        return self._queued_bytes + self._collector_bytes + self.retained_suppression_bytes

    @property
    def retained_agent_count(self) -> int:
        return len(self._collectors)

    @property
    def retained_tool_count(self) -> int:
        return len(self._tool_order)

    @property
    def retained_queue_count(self) -> int:
        return len(self._queues)

    @property
    def round_robin_count(self) -> int:
        return len(self._round_robin)

    def set_mode(self, mode: DisplayMode) -> DisplayModeChange:
        previous = self._mode
        if mode is previous:
            return DisplayModeChange(previous=previous, current=mode)
        discarded_frames = self.queued_count
        discarded_bytes = self.retained_bytes
        self._mode = mode
        self._clear()
        return DisplayModeChange(
            previous=previous,
            current=mode,
            discarded_frames=discarded_frames,
            discarded_bytes=discarded_bytes,
        )

    def discard_agent(self, agent_id: str) -> tuple[int, int]:
        queue_agent_id = self._queue_agent_id(agent_id)
        queue = self._queues.pop(queue_agent_id, deque())
        frames = len(queue)
        byte_count = sum(self._frame_size(frame) for frame in queue)
        self._queued_bytes -= byte_count
        self._remove_collector(agent_id, suppress_incomplete=False)
        self._unschedule(queue_agent_id)
        return frames, byte_count

    def observe(
        self,
        agent_id: str,
        event: RuntimeEvent,
        *,
        agent_name: str | None = None,
    ) -> None:  # noqa: C901
        if self._mode is not DisplayMode.ALL_ACTIONS:
            return
        agent_name = normalize_agent_name(agent_name)
        try:
            match event:
                case AgentStarted(name=name, kind=kind):
                    self._enqueue(agent_id, "lifecycle", f"started ({kind})", agent_name=agent_name or name)
                case TurnStarted():
                    self._enqueue(agent_id, "lifecycle", "turn started", agent_name=agent_name)
                case TurnFinished(status=status):
                    detail = f"turn {status.value}"
                    self._enqueue(agent_id, "lifecycle", detail, critical=True, agent_name=agent_name)
                    self._remove_collector(agent_id, suppress_incomplete=True)
                case AgentStopped(status=status):
                    self._enqueue(
                        agent_id,
                        "lifecycle",
                        f"stopped ({status.value})",
                        critical=True,
                        agent_name=agent_name,
                    )
                    self._remove_collector(agent_id, suppress_incomplete=True)
                case ToolUseStart(tool_use_id=tool_use_id, name=name):
                    self._remember_tool(agent_id, tool_use_id, _ToolAction(name=name, agent_name=agent_name))
                case ToolInputDelta(tool_use_id=tool_use_id, partial_json=partial_json):
                    action = self._touch_tool(agent_id, tool_use_id)
                    if action is None or action.call_emitted or not partial_json:
                        return
                    action.arguments.append(partial_json)
                    fragment_bytes = len(partial_json.encode("utf-8"))
                    action.arguments_bytes += fragment_bytes
                    self._collector_bytes += fragment_bytes
                    self._emit_call_if_complete(agent_id, action)
                case ToolOutputDelta(tool_use_id=tool_use_id, key=key, delta=delta):
                    action = self._touch_tool(agent_id, tool_use_id)
                    if action is None:
                        return
                    self._emit_call(agent_id, action, retained=True)
                    action.saw_output = True
                    self._set_output_buffer(action, key, action.output.get(key, "") + delta)
                    self._emit_complete_output(agent_id, action, key)
                case ToolResult(tool_use_id=tool_use_id, name=name, is_error=is_error, content=content):
                    action = self._pop_tool(agent_id, tool_use_id)
                    if action is None:
                        self._enqueue(
                            agent_id,
                            "tool error" if is_error else "tool result",
                            f"{'✗' if is_error else '✓'} {name}\n{content}" if content else name,
                            critical=True,
                            agent_name=agent_name,
                        )
                        return
                    self._emit_call(agent_id, action, retained=False)
                    for key in tuple(action.output):
                        self._flush_output(agent_id, action, key)
                    if is_error:
                        self._enqueue(
                            agent_id,
                            "tool error",
                            f"✗ {action.name}\n{content}",
                            critical=True,
                            agent_name=action.agent_name,
                        )
                    elif action.saw_output:
                        self._enqueue(
                            agent_id,
                            "tool result",
                            f"✓ {action.name} completed",
                            critical=True,
                            agent_name=action.agent_name,
                        )
                    else:
                        suffix = f"\n{content}" if content else ""
                        self._enqueue(
                            agent_id,
                            "tool result",
                            f"✓ {action.name}{suffix}",
                            critical=True,
                            agent_name=action.agent_name,
                        )
                case Error():
                    self._enqueue(
                        agent_id,
                        "error",
                        "agent stream failed",
                        critical=True,
                        agent_name=agent_name,
                    )
                case SessionEndEvent(stop_reason=stop_reason):
                    self._enqueue(
                        agent_id,
                        "lifecycle",
                        f"session ended ({stop_reason.value})",
                        critical=True,
                        agent_name=agent_name,
                    )
                    self._remove_collector(agent_id, suppress_incomplete=True)
                case _:
                    return
        finally:
            self._enforce_retained_limit()

    def adopt_tool(
        self,
        agent_id: str,
        tool_use_id: str,
        name: str,
        *,
        agent_name: str | None = None,
    ) -> None:
        """Collect continuation events for a tool call already shown live."""
        if self._mode is not DisplayMode.ALL_ACTIONS:
            return
        agent_name = normalize_agent_name(agent_name)
        self._remember_tool(
            agent_id,
            tool_use_id,
            _ToolAction(name=name, agent_name=agent_name, call_emitted=True),
        )
        self._enforce_retained_limit()

    def drain(self, *, max_frames: int = 4, max_bytes: int = 16 * 1024) -> list[str]:
        if min(max_frames, max_bytes) <= 0:
            return []
        rendered: list[str] = []
        used_bytes = 0
        if self._suppressed_frames or self._suppressed_state:
            suppression_frame = self._suppression_frame()
            text = suppression_frame.render()
            size = len(text.encode("utf-8"))
            if size <= max_bytes:
                rendered.append(text)
                used_bytes = size
                self._suppressed_frames = 0
                self._suppressed_state = 0
        attempts = 0
        while self._round_robin and len(rendered) < max_frames:
            agent_id = self._round_robin.popleft()
            self._scheduled.discard(agent_id)
            attempts += 1
            frame = self._next_frame(agent_id)
            if frame is None:
                continue
            text = frame.render()
            size = len(text.encode("utf-8"))
            if size > max_bytes - used_bytes:
                self._restore_front(agent_id, frame)
                if attempts >= len(self._round_robin) + 1:
                    break
                continue
            rendered.append(text)
            used_bytes += size
            attempts = 0
            if self._has_pending(agent_id):
                self._schedule(agent_id)
        return rendered

    def _emit_call_if_complete(self, agent_id: str, action: _ToolAction) -> None:
        try:
            value = json.loads("".join(action.arguments))
        except json.JSONDecodeError:
            return
        if isinstance(value, dict):
            self._emit_call(agent_id, action, arguments=value, retained=True)

    def _emit_call(
        self,
        agent_id: str,
        action: _ToolAction,
        *,
        arguments: object | None = None,
        retained: bool,
    ) -> None:
        if action.call_emitted:
            return
        action.call_emitted = True
        if arguments is None and action.arguments:
            raw = "".join(action.arguments)
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = raw
        if arguments is None:
            body = f"▶ {action.name}"
        elif isinstance(arguments, str):
            body = f"▶ {action.name}\narguments: {arguments}"
        else:
            body = f"▶ {action.name}\narguments: {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"
        self._enqueue(agent_id, "tool call", body, agent_name=action.agent_name)
        if retained:
            self._collector_bytes -= action.arguments_bytes
        action.arguments.clear()
        action.arguments_bytes = 0

    def _emit_complete_output(self, agent_id: str, action: _ToolAction, key: str) -> None:
        while buffer := action.output.get(key, ""):
            newline = buffer.rfind("\n", 0, self._output_chunk_chars + 1)
            if newline >= 0:
                end = newline + 1
            elif len(buffer) >= self._output_chunk_chars:
                end = self._output_chunk_chars
            else:
                break
            self._enqueue(agent_id, f"{action.name} {key}", buffer[:end], agent_name=action.agent_name)
            self._set_output_buffer(action, key, buffer[end:])

    def _flush_output(self, agent_id: str, action: _ToolAction, key: str) -> None:
        buffer = action.output.pop(key, "")
        if buffer:
            self._enqueue(agent_id, f"{action.name} {key}", buffer, agent_name=action.agent_name)

    def _set_output_buffer(self, action: _ToolAction, key: str, value: str) -> None:
        old = action.output.get(key)
        old_bytes = 0 if old is None else len(key.encode("utf-8")) + len(old.encode("utf-8"))
        if value:
            new_bytes = len(key.encode("utf-8")) + len(value.encode("utf-8"))
            action.output[key] = value
        else:
            new_bytes = 0
            action.output.pop(key, None)
        delta = new_bytes - old_bytes
        action.output_bytes += delta
        self._collector_bytes += delta

    def _enqueue(
        self,
        agent_id: str,
        kind: str,
        body: object,
        *,
        critical: bool = False,
        agent_name: str | None = None,
    ) -> None:
        agent_id = self._queue_agent_id(agent_id)
        agent_name = normalize_agent_name(agent_name)
        kind = _fit_utf8(sanitize_terminal_text(kind).replace("\n", " "), 40, suffix="…") or "action"
        clean_body = sanitize_terminal_text(body)
        frame = ActionFrame(
            agent_id=agent_id,
            kind=kind,
            body="",
            sequence=0,
            critical=critical,
            agent_name=agent_name,
            powerline=self._powerline,
            theme=self._theme,
        )
        overhead = len(frame.render().encode("utf-8"))
        if overhead >= self._max_frame_bytes:
            while overhead >= self._max_frame_bytes and (agent_name or len(agent_id.encode()) > 1):
                excess = overhead - self._max_frame_bytes + 1
                if agent_name:
                    agent_name = (
                        _fit_utf8(
                            agent_name,
                            max(0, len(agent_name.encode()) - (excess + 1) // 2),
                            suffix="…",
                        ).strip()
                        or None
                    )
                else:
                    agent_id = _fit_utf8(
                        agent_id,
                        max(1, len(agent_id.encode()) - (excess + 1) // 2),
                        suffix="…",
                    )
                frame = ActionFrame(
                    agent_id=agent_id,
                    kind=kind,
                    body="",
                    sequence=0,
                    critical=critical,
                    agent_name=agent_name,
                    powerline=self._powerline,
                    theme=self._theme,
                )
                overhead = len(frame.render().encode("utf-8"))
        clean_body = _fit_utf8(clean_body, max(0, self._max_frame_bytes - overhead))
        self._sequence += 1
        frame = ActionFrame(
            agent_id=agent_id,
            kind=kind,
            body=clean_body,
            sequence=self._sequence,
            critical=critical,
            agent_name=agent_name,
            powerline=self._powerline,
            theme=self._theme,
        )
        queue = self._queues.setdefault(agent_id, deque())
        queue.append(frame)
        self._queued_bytes += self._frame_size(frame)
        self._schedule(agent_id)
        while len(queue) > self._max_frames_per_agent:
            self._drop_front(agent_id)
        while self._frame_count() > self._max_queued_frames or self._queued_bytes > self._max_queued_bytes:
            oldest_agent = self._oldest_agent()
            if oldest_agent is None:
                break
            self._drop_front(oldest_agent)

    def _next_frame(self, agent_id: str) -> ActionFrame | None:
        queue = self._queues.get(agent_id)
        if not queue:
            return None
        frame = queue.popleft()
        self._queued_bytes -= self._frame_size(frame)
        if not queue:
            self._queues.pop(agent_id, None)
        return frame

    def _restore_front(self, agent_id: str, frame: ActionFrame) -> None:
        queue = self._queues.setdefault(agent_id, deque())
        queue.appendleft(frame)
        self._queued_bytes += self._frame_size(frame)
        self._schedule(agent_id)

    def _drop_front(self, agent_id: str) -> None:
        queue = self._queues.get(agent_id)
        if not queue:
            return
        frame = next((candidate for candidate in queue if not candidate.critical), queue[0])
        queue.remove(frame)
        self._queued_bytes -= self._frame_size(frame)
        self._suppressed_frames += 1
        if not queue:
            self._queues.pop(agent_id, None)
            self._unschedule(agent_id)

    def _oldest_agent(self) -> str | None:
        oldest_agent: str | None = None
        oldest_sequence: int | None = None
        critical_agent: str | None = None
        critical_sequence: int | None = None
        for agent_id, queue in self._queues.items():
            for frame in queue:
                if frame.critical:
                    if critical_sequence is None or frame.sequence < critical_sequence:
                        critical_agent = agent_id
                        critical_sequence = frame.sequence
                elif oldest_sequence is None or frame.sequence < oldest_sequence:
                    oldest_agent = agent_id
                    oldest_sequence = frame.sequence
        return oldest_agent or critical_agent

    def _frame_count(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    def _has_pending(self, agent_id: str) -> bool:
        return bool(self._queues.get(agent_id))

    def _remember_tool(self, agent_id: str, tool_use_id: str, action: _ToolAction) -> None:
        # Eviction can remove the agent's last tool and collector, so make room
        # before resolving the collector that will receive the new action.
        self._discard_tool(agent_id, tool_use_id, suppress_incomplete=False)
        while len(self._tool_order) >= self._max_tools:
            oldest_agent, oldest_tool_id = next(iter(self._tool_order))
            self._discard_tool(oldest_agent, oldest_tool_id, suppress_incomplete=True)

        collector = self._collectors.get(agent_id)
        while collector is not None and len(collector.tools) >= self._max_tools_per_agent:
            oldest_tool_id = next(iter(collector.tools))
            self._discard_tool(agent_id, oldest_tool_id, suppress_incomplete=True)
            collector = self._collectors.get(agent_id)

        if collector is None:
            while len(self._collectors) >= self._max_agents:
                oldest_agent = next(iter(self._collectors))
                self._remove_collector(oldest_agent, suppress_incomplete=True)
            collector = _AgentCollector(identity_bytes=len(agent_id.encode("utf-8")))
            self._collectors[agent_id] = collector
            self._collector_bytes += collector.identity_bytes
        else:
            self._collectors.move_to_end(agent_id)

        collector.tools[tool_use_id] = action
        self._tool_order[(agent_id, tool_use_id)] = None
        self._collector_bytes += len(tool_use_id.encode("utf-8")) + action.retained_bytes

    def _touch_tool(self, agent_id: str, tool_use_id: str) -> _ToolAction | None:
        collector = self._collectors.get(agent_id)
        if collector is None:
            return None
        action = collector.tools.get(tool_use_id)
        if action is None:
            return None
        collector.tools.move_to_end(tool_use_id)
        self._collectors.move_to_end(agent_id)
        self._tool_order.move_to_end((agent_id, tool_use_id))
        return action

    def _pop_tool(self, agent_id: str, tool_use_id: str) -> _ToolAction | None:
        collector = self._collectors.get(agent_id)
        if collector is None:
            self._tool_order.pop((agent_id, tool_use_id), None)
            return None
        action = collector.tools.pop(tool_use_id, None)
        self._tool_order.pop((agent_id, tool_use_id), None)
        if action is None:
            return None
        self._collector_bytes -= len(tool_use_id.encode("utf-8")) + action.retained_bytes
        if not collector.tools:
            self._collectors.pop(agent_id, None)
            self._collector_bytes -= collector.identity_bytes
        return action

    def _discard_tool(self, agent_id: str, tool_use_id: str, *, suppress_incomplete: bool) -> None:
        action = self._pop_tool(agent_id, tool_use_id)
        if action is not None and suppress_incomplete:
            self._suppressed_state += 1

    def _remove_collector(self, agent_id: str, *, suppress_incomplete: bool) -> None:
        collector = self._collectors.pop(agent_id, None)
        stale_tool_ids = [tool_use_id for owner, tool_use_id in self._tool_order if owner == agent_id]
        if collector is not None:
            self._collector_bytes -= collector.identity_bytes
            self._collector_bytes -= sum(
                len(tool_use_id.encode("utf-8")) + action.retained_bytes
                for tool_use_id, action in collector.tools.items()
            )
            stale_tool_ids.extend(tool_use_id for tool_use_id in collector.tools if tool_use_id not in stale_tool_ids)
        for tool_use_id in stale_tool_ids:
            self._tool_order.pop((agent_id, tool_use_id), None)
        if suppress_incomplete:
            self._suppressed_state += len(stale_tool_ids)

    def _suppression_body(self) -> str:
        parts: list[str] = []
        if self._suppressed_frames:
            count = self._suppressed_frames
            shown = str(count) if count <= _MAX_DISPLAY_COUNT else f"{_MAX_DISPLAY_COUNT}+"
            parts.append(f"{shown} action frame{'s' if count != 1 else ''} suppressed")
        if self._suppressed_state:
            count = self._suppressed_state
            shown = str(count) if count <= _MAX_DISPLAY_COUNT else f"{_MAX_DISPLAY_COUNT}+"
            parts.append(f"{shown} incomplete action{'s' if count != 1 else ''} discarded")
        return "; ".join(parts)

    def _make_suppression_frame(self, *, sequence: int) -> ActionFrame:
        return ActionFrame(
            agent_id="all agents",
            kind="suppressed",
            body=self._suppression_body(),
            sequence=sequence,
            critical=True,
            powerline=self._powerline,
            theme=self._theme,
        )

    def _suppression_frame(self) -> ActionFrame:
        self._sequence += 1
        return self._make_suppression_frame(sequence=self._sequence)

    def _enforce_retained_limit(self) -> None:
        while self.retained_bytes > self._max_retained_bytes:
            # Incomplete collector payload has no durable presentation value;
            # shed it before queued frames, where critical results are preferred.
            if self._tool_order:
                agent_id, tool_use_id = next(iter(self._tool_order))
                self._discard_tool(agent_id, tool_use_id, suppress_incomplete=True)
                continue
            oldest_agent = self._oldest_agent()
            if oldest_agent is not None:
                self._drop_front(oldest_agent)
                continue
            break

    def _queue_agent_id(self, agent_id: str) -> str:
        return _fit_utf8(sanitize_terminal_text(agent_id).replace("\n", " "), 80, suffix="…") or "unknown"

    def _schedule(self, agent_id: str) -> None:
        if agent_id not in self._scheduled:
            self._scheduled.add(agent_id)
            self._round_robin.append(agent_id)

    def _unschedule(self, agent_id: str) -> None:
        if agent_id not in self._scheduled:
            return
        self._scheduled.discard(agent_id)
        self._round_robin = deque(item for item in self._round_robin if item != agent_id)

    @staticmethod
    def _frame_size(frame: ActionFrame) -> int:
        return len(frame.render().encode("utf-8"))

    def _clear(self) -> None:
        self._collectors.clear()
        self._tool_order.clear()
        self._queues.clear()
        self._round_robin.clear()
        self._scheduled.clear()
        self._queued_bytes = 0
        self._collector_bytes = 0
        self._suppressed_frames = 0
        self._suppressed_state = 0
