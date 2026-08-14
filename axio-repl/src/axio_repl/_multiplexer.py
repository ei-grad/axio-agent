"""Bounded presentation queues for actions from non-active agents."""

from __future__ import annotations

import json
import re
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import StrEnum

from axio.events import Error, SessionEndEvent, ToolInputDelta, ToolOutputDelta, ToolResult, ToolUseStart
from axio_tools_agents.runtime import AgentStarted, AgentStopped, RuntimeEvent, TurnFinished, TurnStarted

RESET = "\033[0m"

_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


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

    def render(self) -> str:
        agent_id = sanitize_terminal_text(self.agent_id).replace("\n", " ")[:80]
        kind = sanitize_terminal_text(self.kind).replace("\n", " ")[:40]
        body = sanitize_terminal_text(self.body).rstrip("\n")
        return f"{RESET}\n── agent {agent_id} · {kind} ──\n{body}\n── /agent {agent_id} ──\n{RESET}\n"


@dataclass(slots=True)
class _ToolAction:
    name: str
    arguments: list[str] = field(default_factory=list)
    call_emitted: bool = False
    output: dict[str, str] = field(default_factory=dict)
    saw_output: bool = False


@dataclass(slots=True)
class _AgentCollector:
    tools: OrderedDict[str, _ToolAction] = field(default_factory=OrderedDict)


def sanitize_terminal_text(value: object) -> str:
    """Remove terminal control sequences while preserving lines and tabs."""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = _OSC.sub("", text)
    text = _CSI.sub("", text)
    return _CONTROL.sub("", text)


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
        max_queued_frames: int = 256,
        max_queued_bytes: int = 256 * 1024,
        max_frames_per_agent: int = 64,
        max_frame_bytes: int = 3072,
        output_chunk_chars: int = 1024,
        max_agents: int = 256,
        max_tools: int = 512,
        max_tools_per_agent: int = 64,
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
            )
            <= 0
        ):
            raise ValueError("multiplexer limits must be positive")
        self._mode = mode
        self._max_queued_frames = max_queued_frames
        self._max_queued_bytes = max_queued_bytes
        self._max_frames_per_agent = max_frames_per_agent
        self._max_frame_bytes = max_frame_bytes
        self._output_chunk_chars = output_chunk_chars
        self._max_agents = max_agents
        self._max_tools = max_tools
        self._max_tools_per_agent = max_tools_per_agent
        self._collectors: OrderedDict[str, _AgentCollector] = OrderedDict()
        self._tool_order: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._queues: dict[str, deque[ActionFrame]] = {}
        self._round_robin: deque[str] = deque()
        self._scheduled: set[str] = set()
        self._queued_bytes = 0
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
        discarded_bytes = self._queued_bytes
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

    def observe(self, agent_id: str, event: RuntimeEvent) -> None:  # noqa: C901
        if self._mode is not DisplayMode.ALL_ACTIONS:
            return
        match event:
            case AgentStarted(name=name, kind=kind):
                self._enqueue(agent_id, "lifecycle", f"started {name} ({kind})")
            case TurnStarted():
                self._enqueue(agent_id, "lifecycle", "turn started")
            case TurnFinished(status=status, error=error):
                detail = f"turn {status.value}"
                if error:
                    detail += f": {error}"
                self._enqueue(agent_id, "lifecycle", detail, critical=True)
                self._remove_collector(agent_id, suppress_incomplete=True)
            case AgentStopped(status=status):
                self._enqueue(agent_id, "lifecycle", f"stopped ({status.value})", critical=True)
                self._remove_collector(agent_id, suppress_incomplete=True)
            case ToolUseStart(tool_use_id=tool_use_id, name=name):
                self._remember_tool(agent_id, tool_use_id, _ToolAction(name=name))
            case ToolInputDelta(tool_use_id=tool_use_id, partial_json=partial_json):
                collector = self._collectors.get(agent_id)
                if collector is None:
                    return
                action = collector.tools.get(tool_use_id)
                if action is None:
                    return
                collector.tools.move_to_end(tool_use_id)
                self._collectors.move_to_end(agent_id)
                self._tool_order.move_to_end((agent_id, tool_use_id))
                action.arguments.append(partial_json)
                self._emit_call_if_complete(agent_id, action)
            case ToolOutputDelta(tool_use_id=tool_use_id, key=key, delta=delta):
                collector = self._collectors.get(agent_id)
                if collector is None:
                    return
                action = collector.tools.get(tool_use_id)
                if action is None:
                    return
                collector.tools.move_to_end(tool_use_id)
                self._collectors.move_to_end(agent_id)
                self._tool_order.move_to_end((agent_id, tool_use_id))
                self._emit_call(agent_id, action)
                action.saw_output = True
                action.output[key] = action.output.get(key, "") + delta
                self._emit_complete_output(agent_id, action, key)
            case ToolResult(tool_use_id=tool_use_id, name=name, is_error=is_error, content=content):
                action = self._pop_tool(agent_id, tool_use_id)
                if action is None:
                    self._enqueue(
                        agent_id,
                        "tool error" if is_error else "tool result",
                        f"{'✗' if is_error else '✓'} {name}\n{content}" if content else name,
                        critical=True,
                    )
                    return
                self._emit_call(agent_id, action)
                for key in tuple(action.output):
                    self._flush_output(agent_id, action, key)
                if is_error:
                    self._enqueue(agent_id, "tool error", f"✗ {action.name}\n{content}", critical=True)
                elif action.saw_output:
                    self._enqueue(agent_id, "tool result", f"✓ {action.name} completed", critical=True)
                else:
                    suffix = f"\n{content}" if content else ""
                    self._enqueue(agent_id, "tool result", f"✓ {action.name}{suffix}", critical=True)
            case Error(exception=exception):
                self._enqueue(agent_id, "error", f"{type(exception).__name__}: {exception}", critical=True)
            case SessionEndEvent(stop_reason=stop_reason):
                self._enqueue(agent_id, "lifecycle", f"session ended ({stop_reason.value})", critical=True)
                self._remove_collector(agent_id, suppress_incomplete=True)
            case _:
                return

    def adopt_tool(self, agent_id: str, tool_use_id: str, name: str) -> None:
        """Collect continuation events for a tool call already shown live."""
        if self._mode is not DisplayMode.ALL_ACTIONS:
            return
        self._remember_tool(agent_id, tool_use_id, _ToolAction(name=name, call_emitted=True))

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
            self._emit_call(agent_id, action, arguments=value)

    def _emit_call(self, agent_id: str, action: _ToolAction, *, arguments: object | None = None) -> None:
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
        self._enqueue(agent_id, "tool call", body)

    def _emit_complete_output(self, agent_id: str, action: _ToolAction, key: str) -> None:
        buffer = action.output[key]
        while buffer:
            newline = buffer.rfind("\n", 0, self._output_chunk_chars + 1)
            if newline >= 0:
                end = newline + 1
            elif len(buffer) >= self._output_chunk_chars:
                end = self._output_chunk_chars
            else:
                break
            self._enqueue(agent_id, f"{action.name} {key}", buffer[:end])
            buffer = buffer[end:]
        action.output[key] = buffer

    def _flush_output(self, agent_id: str, action: _ToolAction, key: str) -> None:
        buffer = action.output.pop(key, "")
        if buffer:
            self._enqueue(agent_id, f"{action.name} {key}", buffer)

    def _enqueue(self, agent_id: str, kind: str, body: object, *, critical: bool = False) -> None:
        agent_id = self._queue_agent_id(agent_id)
        kind = _fit_utf8(sanitize_terminal_text(kind).replace("\n", " "), 40, suffix="…") or "action"
        clean_body = sanitize_terminal_text(body)
        overhead = len(f"{RESET}\n── agent {agent_id} · {kind} ──\n\n── /agent {agent_id} ──\n{RESET}\n".encode())
        if overhead >= self._max_frame_bytes:
            while overhead >= self._max_frame_bytes and len(agent_id.encode()) > 1:
                excess = overhead - self._max_frame_bytes + 1
                agent_id = _fit_utf8(
                    agent_id,
                    max(1, len(agent_id.encode()) - (excess + 1) // 2),
                    suffix="…",
                )
                overhead = len(
                    f"{RESET}\n── agent {agent_id} · {kind} ──\n\n── /agent {agent_id} ──\n{RESET}\n".encode()
                )
        clean_body = _fit_utf8(clean_body, max(0, self._max_frame_bytes - overhead))
        self._sequence += 1
        frame = ActionFrame(
            agent_id=agent_id,
            kind=kind,
            body=clean_body,
            sequence=self._sequence,
            critical=critical,
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
        collector = self._collectors.get(agent_id)
        if collector is None:
            while len(self._collectors) >= self._max_agents:
                oldest_agent = next(iter(self._collectors))
                self._remove_collector(oldest_agent, suppress_incomplete=True)
            collector = _AgentCollector()
            self._collectors[agent_id] = collector
        else:
            self._collectors.move_to_end(agent_id)

        if tool_use_id in collector.tools:
            collector.tools.pop(tool_use_id)
            self._tool_order.pop((agent_id, tool_use_id), None)
        while len(collector.tools) >= self._max_tools_per_agent:
            oldest_tool_id, _ = collector.tools.popitem(last=False)
            self._tool_order.pop((agent_id, oldest_tool_id), None)
            self._suppressed_state += 1
        while len(self._tool_order) >= self._max_tools:
            (oldest_agent, oldest_tool_id), _ = self._tool_order.popitem(last=False)
            oldest_collector = self._collectors.get(oldest_agent)
            if oldest_collector is not None:
                oldest_collector.tools.pop(oldest_tool_id, None)
                if not oldest_collector.tools:
                    self._collectors.pop(oldest_agent, None)
            self._suppressed_state += 1
        collector.tools[tool_use_id] = action
        self._tool_order[(agent_id, tool_use_id)] = None

    def _pop_tool(self, agent_id: str, tool_use_id: str) -> _ToolAction | None:
        collector = self._collectors.get(agent_id)
        if collector is None:
            return None
        action = collector.tools.pop(tool_use_id, None)
        self._tool_order.pop((agent_id, tool_use_id), None)
        if not collector.tools:
            self._collectors.pop(agent_id, None)
        return action

    def _remove_collector(self, agent_id: str, *, suppress_incomplete: bool) -> None:
        collector = self._collectors.pop(agent_id, None)
        if collector is None:
            return
        for tool_use_id in collector.tools:
            self._tool_order.pop((agent_id, tool_use_id), None)
        if suppress_incomplete:
            self._suppressed_state += len(collector.tools)

    def _suppression_frame(self) -> ActionFrame:
        parts: list[str] = []
        if self._suppressed_frames:
            count = self._suppressed_frames
            parts.append(f"{count} action frame{'s' if count != 1 else ''} suppressed")
        if self._suppressed_state:
            count = self._suppressed_state
            parts.append(f"{count} incomplete action{'s' if count != 1 else ''} discarded")
        self._sequence += 1
        return ActionFrame(
            agent_id="all agents",
            kind="suppressed",
            body="; ".join(parts),
            sequence=self._sequence,
            critical=True,
        )

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
        self._suppressed_frames = 0
        self._suppressed_state = 0
