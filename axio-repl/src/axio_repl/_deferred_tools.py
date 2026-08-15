"""Session-owned continuation of tool dispatches released by interrupted turns."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from axio.agent import ToolDispatch
from axio.blocks import AudioBlock, ImageBlock, TextBlock, ToolResultBlock, VideoBlock
from axio_tools_agents.runtime import current_turn_identity


class DeferredToolPhase(StrEnum):
    ACTIVE = "active"
    DEFERRED = "deferred"
    PROTOCOL_CLOSED = "protocol_closed"
    DELIVERED = "delivered"


@dataclass(frozen=True, slots=True)
class DeferredToolNotification:
    agent_id: str
    run_id: str
    tool_use_id: str
    tool_name: str
    text: str
    is_error: bool

    def as_user_text(self) -> str:
        status = "failed" if self.is_error else "completed"
        return f"[Deferred tool {status}: name={self.tool_name}, call_id={self.tool_use_id}]\n\n{self.text}"


@dataclass(frozen=True, slots=True)
class DeferredToolSnapshot:
    agent_id: str
    turn_id: str | None
    tool_use_ids: tuple[str, ...]
    tool_names: tuple[str, ...]
    phase: DeferredToolPhase


@dataclass(slots=True)
class _OwnedDispatch:
    dispatch: ToolDispatch
    agent_id: str
    run_id: str
    turn_id: str | None
    protocol_closed: asyncio.Event
    phase: DeferredToolPhase = DeferredToolPhase.ACTIVE
    watcher: asyncio.Task[None] | None = None
    preemption_requested: bool = False


NotificationHandler = Callable[[DeferredToolNotification], Awaitable[None]]
DispatchStartedHandler = Callable[[str, str | None], None]


class DeferredToolRegistry:
    """Retain deferred tasks and publish their results only after protocol close."""

    def __init__(
        self,
        deliver: NotificationHandler,
        *,
        on_dispatch_started: DispatchStartedHandler | None = None,
    ) -> None:
        self._deliver = deliver
        self._on_dispatch_started = on_dispatch_started
        self._records: dict[asyncio.Task[list[ToolResultBlock]], _OwnedDispatch] = {}

    def set_dispatch_started_handler(self, handler: DispatchStartedHandler | None) -> None:
        self._on_dispatch_started = handler

    def dispatch_started(self, dispatch: ToolDispatch) -> None:
        if dispatch.task in self._records:
            raise RuntimeError("tool dispatch task is already registered")
        identity = current_turn_identity()
        agent_id = identity.agent_id if identity is not None else "main"
        run_id = identity.run_id if identity is not None else "deferred-tool"
        turn_id = identity.turn_id if identity is not None else None
        self._records[dispatch.task] = _OwnedDispatch(
            dispatch=dispatch,
            agent_id=agent_id,
            run_id=run_id,
            turn_id=turn_id,
            protocol_closed=asyncio.Event(),
        )
        if self._on_dispatch_started is not None:
            self._on_dispatch_started(agent_id, turn_id)

    def dispatch_finished(self, dispatch: ToolDispatch) -> None:
        record = self._records.get(dispatch.task)
        if record is None:
            return
        if record.phase is not DeferredToolPhase.ACTIVE:
            raise RuntimeError("a deferred dispatch cannot finish through the active path")
        self._records.pop(dispatch.task)

    def defer(self, dispatch: ToolDispatch) -> None:
        record = self._require(dispatch)
        if record.phase is not DeferredToolPhase.ACTIVE:
            raise RuntimeError(f"cannot defer dispatch from {record.phase.value} state")
        record.phase = DeferredToolPhase.DEFERRED
        record.watcher = asyncio.create_task(
            self._watch(record),
            name=f"axio-repl-deferred-tools-{dispatch.blocks[0].id}",
        )

    def should_defer(self, dispatch: ToolDispatch) -> bool:
        return self._require(dispatch).preemption_requested

    def request_preemption(self, turn_id: str | None) -> bool:
        requested = False
        for record in self._records.values():
            if (
                record.turn_id == turn_id
                and record.phase is DeferredToolPhase.ACTIVE
                and not record.dispatch.task.done()
                and not record.preemption_requested
            ):
                record.preemption_requested = True
                requested = True
        return requested

    def protocol_closed(self, dispatch: ToolDispatch) -> None:
        record = self._require(dispatch)
        if record.phase is not DeferredToolPhase.DEFERRED:
            raise RuntimeError(f"cannot close deferred protocol from {record.phase.value} state")
        record.phase = DeferredToolPhase.PROTOCOL_CLOSED
        record.protocol_closed.set()

    def has_active_dispatch(self, turn_id: str | None) -> bool:
        return any(
            record.turn_id == turn_id and record.phase is DeferredToolPhase.ACTIVE and not record.dispatch.task.done()
            for record in self._records.values()
        )

    def snapshots(self) -> tuple[DeferredToolSnapshot, ...]:
        return tuple(
            DeferredToolSnapshot(
                agent_id=record.agent_id,
                turn_id=record.turn_id,
                tool_use_ids=tuple(block.id for block in record.dispatch.blocks),
                tool_names=tuple(block.name for block in record.dispatch.blocks),
                phase=record.phase,
            )
            for record in self._records.values()
        )

    async def close(self) -> tuple[DeferredToolSnapshot, ...]:
        snapshots = self.snapshots()
        records = tuple(self._records.values())
        for record in records:
            if not record.dispatch.task.done():
                record.dispatch.task.cancel()
            if record.watcher is not None and not record.watcher.done():
                record.watcher.cancel()
        tasks: list[asyncio.Task[object]] = []
        tasks.extend(record.dispatch.task for record in records)
        tasks.extend(record.watcher for record in records if record.watcher is not None)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._records.clear()
        return snapshots

    def _require(self, dispatch: ToolDispatch) -> _OwnedDispatch:
        record = self._records.get(dispatch.task)
        if record is None:
            raise RuntimeError("unknown tool dispatch")
        return record

    async def _watch(self, record: _OwnedDispatch) -> None:
        try:
            try:
                results = await asyncio.shield(record.dispatch.task)
                by_id = {result.tool_use_id: result for result in results}
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                by_id = {
                    block.id: ToolResultBlock(tool_use_id=block.id, content=str(exc), is_error=True)
                    for block in record.dispatch.blocks
                }
            await record.protocol_closed.wait()
            for block in record.dispatch.blocks:
                result = by_id.get(
                    block.id,
                    ToolResultBlock(
                        tool_use_id=block.id,
                        content="Deferred tool ended without a result.",
                        is_error=True,
                    ),
                )
                await self._deliver(
                    DeferredToolNotification(
                        agent_id=record.agent_id,
                        run_id=record.run_id,
                        tool_use_id=block.id,
                        tool_name=block.name,
                        text=_result_text(result),
                        is_error=result.is_error,
                    )
                )
            record.phase = DeferredToolPhase.DELIVERED
        finally:
            self._records.pop(record.dispatch.task, None)


def _result_text(result: ToolResultBlock) -> str:
    if isinstance(result.content, str):
        return result.content
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ImageBlock):
            parts.append(f"[image result: {block.media_type}, {len(block.data)} bytes]")
        elif isinstance(block, AudioBlock):
            parts.append(f"[audio result: {block.media_type}, {len(block.data)} bytes]")
        elif isinstance(block, VideoBlock):
            parts.append(f"[video result: {block.media_type}, {len(block.data)} bytes]")
    return "\n".join(parts)
