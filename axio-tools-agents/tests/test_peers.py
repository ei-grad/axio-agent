from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest
from axio import notify
from axio.agent import Agent
from axio.blocks import TextBlock, ToolResultBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.events import IterationEnd, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from axio.exceptions import HandlerError
from axio.messages import Message
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.tool import Tool
from axio.types import StopReason, Usage

from axio_tools_agents.peers import (
    MAX_MESSAGE_CHARS,
    PeerMessage,
    PeerRecord,
    PeerServer,
    background_agent_state,
    enqueue_local_agent_context,
    enqueue_local_agent_messages,
    enqueue_local_agent_prompt,
    format_message_for_dialog,
    interrupt_agent,
    interrupt_local_agent_turn,
    is_local_background_agent,
    list_peer_records,
    list_peers,
    mark_background_report_delivered,
    peer_context,
    run_agent,
    send_message,
    set_agent_event_handler,
    set_background_input_admitted_handler,
    set_background_outcome_handler,
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
    ExecutionMode,
    ForegroundEntered,
    ForegroundExited,
    MessageCommitted,
    ObservedContextStore,
    OutcomeDelivered,
    SessionEventHub,
    TurnOutcome,
    TurnStatus,
    new_turn_identity,
    observe_agent_turn,
)


@pytest.fixture(autouse=True)
async def peer_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[None]:
    monkeypatch.setenv("AXIO_PEER_DIR", str(tmp_path / "peers"))
    yield
    await stop_local_background_agents()
    set_agent_event_handler(None)
    set_background_outcome_handler(None)
    set_background_input_admitted_handler(None)
    set_run_agent_factory(None)
    set_spawn_agent_factory(None)
    set_session_event_hub(None)


# Owners are only known at run time — a peer id is generated per test — so the
# helpers that hand one out record it here for the fixture to clean up. None is
# always in: it owns everything spawned without a peer identity.
_bus_owners: list[str | None] = [None]


def _watch_bus_owner(owner: str | None) -> str | None:
    _bus_owners.append(owner)
    return owner


@pytest.fixture(autouse=True)
def clean_notification_bus() -> Iterator[None]:
    # Queues and listeners live for the process, so what one test leaves behind
    # would be delivered to the next one.
    yield
    for owner in _bus_owners:
        notify.discard(owner)
    del _bus_owners[1:]


async def _noop_handler(message: PeerMessage) -> None:
    return None


async def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached")


async def test_list_peers_filters_to_current_project_by_default(tmp_path: Path) -> None:
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()

    current = await PeerServer("current", kind="test", handler=_noop_handler, project=str(project_a)).start()
    same_project = await PeerServer(
        "same",
        kind="test",
        handler=_noop_handler,
        project=str(project_a),
    ).start(set_current=False)
    other_project = await PeerServer(
        "other",
        kind="test",
        handler=_noop_handler,
        project=str(project_b),
    ).start(set_current=False)

    try:
        scoped = await list_peers()
        assert same_project.id in scoped
        assert current.id not in scoped
        assert other_project.id not in scoped

        all_projects = await list_peers(all_projects=True)
        assert same_project.id in all_projects
        assert other_project.id in all_projects
    finally:
        await current.close()
        await same_project.close()
        await other_project.close()


async def test_send_message_delivers_by_global_agent_id(tmp_path: Path) -> None:
    received: list[PeerMessage] = []

    async def handler(message: PeerMessage) -> None:
        received.append(message)

    sender = await PeerServer("sender", kind="test", handler=_noop_handler, project=str(tmp_path)).start()
    recipient = await PeerServer(
        "recipient",
        kind="test",
        handler=handler,
        project=str(tmp_path),
    ).start(set_current=False)

    try:
        result = await send_message(agent_id=recipient.id, message="hello")
        assert result.startswith("Delivered message")
        assert len(received) == 1
        assert received[0].from_id == sender.id
        assert received[0].body == "hello"
    finally:
        await sender.close()
        await recipient.close()


async def test_send_message_too_large_raises_handler_error() -> None:
    with pytest.raises(HandlerError, match="too large"):
        await send_message(agent_id="whatever", message="x" * (MAX_MESSAGE_CHARS + 1))


async def test_send_message_unknown_agent_raises_handler_error() -> None:
    with pytest.raises(HandlerError, match="No peer found"):
        await send_message(agent_id="does-not-exist", message="hi")


async def test_send_message_connect_failure_raises_handler_error(tmp_path: Path) -> None:
    import json
    import os
    import time

    peer_dir = tmp_path / "peers"
    peer_dir.mkdir(parents=True, exist_ok=True)
    # A regular file at the socket path stands in for a peer whose process is
    # gone but whose registry entry was never cleaned up: nothing is listening,
    # so connecting to it fails the same way a crashed peer would.
    socket_path = peer_dir / "dead.sock"
    socket_path.write_text("")
    record = PeerRecord(
        id="dead-peer",
        name="dead",
        kind="test",
        project=str(tmp_path),
        pid=os.getpid(),
        cwd=str(tmp_path),
        socket_path=str(socket_path),
        started_at=time.time(),
    )
    (peer_dir / f"{record.id}.json").write_text(json.dumps(record.to_dict()), encoding="utf-8")

    with pytest.raises(HandlerError, match="Failed to connect"):
        await send_message(agent_id=record.id, message="hi")


async def test_send_message_peer_rejection_returns_plain_string(tmp_path: Path) -> None:
    async def failing_handler(message: PeerMessage) -> None:
        raise ValueError("nope")

    sender = await PeerServer("sender", kind="test", handler=_noop_handler, project=str(tmp_path)).start()
    recipient = await PeerServer(
        "recipient",
        kind="test",
        handler=failing_handler,
        project=str(tmp_path),
    ).start(set_current=False)

    try:
        result = await send_message(agent_id=recipient.id, message="hello")
        assert result.startswith(f"Peer {recipient.id} rejected the message:")
    finally:
        await sender.close()
        await recipient.close()


class _MessagingTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.sender: PeerServer | None = None

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        prompt = messages[-1].content[0].text  # type: ignore[attr-defined]
        self.calls.append(prompt)
        if len(self.calls) == 1:
            from axio_tools_agents.peers import list_peer_records

            agent_id = next(record.id for record in await list_peer_records() if record.kind == "spawned-agent")
            assert self.sender is not None
            from axio_tools_agents.peers import peer_context

            with peer_context(self.sender):
                delivered = await send_message(agent_id=agent_id, message="follow-up")
            assert delivered.startswith("Delivered message")
            yield TextDelta(index=0, delta="first")
        else:
            assert "follow-up" in prompt
            yield TextDelta(index=0, delta="second")
        yield IterationEnd(iteration=len(self.calls), stop_reason=StopReason.end_turn, usage=Usage(0, 0))


async def test_spawn_agent_registers_peer_and_processes_inbound_after_turn(tmp_path: Path) -> None:
    transport = _MessagingTransport()
    transport.sender = await PeerServer(
        "sender",
        kind="test",
        handler=_noop_handler,
        project=str(tmp_path),
    ).start(set_current=False)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        assert not inherit_context
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    try:
        result = await spawn_agent(task="initial")
        assert result.startswith("Spawned background agent_id=")
        agent_id = result.split("agent_id=", 1)[1].split(" ", 1)[0]
        await _wait_for(lambda: len(transport.calls) == 2)
        assert transport.calls[0].endswith("initial")
        assert transport.calls[1].endswith(
            format_message_for_dialog(
                PeerMessage(
                    id="unused",
                    from_id=transport.sender.id,
                    from_name=transport.sender.name,
                    to_id="unused",
                    body="follow-up",
                    sent_at=0,
                )
            )
        )
        stop_result = await stop_agent(agent_id=agent_id, reason="done")
        assert stop_result.startswith("Sent stop")
    finally:
        await transport.sender.close()


class _InterruptibleTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.interrupted = asyncio.Event()

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        prompt = messages[-1].content[0].text  # type: ignore[attr-defined]
        self.calls.append(prompt)
        if len(self.calls) == 1:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.interrupted.set()
                raise
        else:
            yield TextDelta(index=0, delta="next")
            yield IterationEnd(iteration=len(self.calls), stop_reason=StopReason.end_turn, usage=Usage(0, 0))


async def test_interrupt_agent_cancels_current_turn_without_stopping(tmp_path: Path) -> None:
    transport = _InterruptibleTransport()
    sender = await PeerServer("sender", kind="test", handler=_noop_handler, project=str(tmp_path)).start()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    try:
        result = await spawn_agent(task="block")
        agent_id = result.split("agent_id=", 1)[1].split(" ", 1)[0]
        await asyncio.wait_for(transport.started.wait(), timeout=1)

        interrupted = await interrupt_agent(agent_id=agent_id, reason="test")
        assert interrupted.startswith("Sent interrupt")
        await asyncio.wait_for(transport.interrupted.wait(), timeout=1)

        delivered = await enqueue_local_agent_prompt(agent_id, "after interrupt", wait=True)
        assert delivered
        assert transport.calls[-1].endswith("after interrupt")

        stopped = await stop_agent(agent_id=agent_id, reason="done")
        assert stopped.startswith("Sent stop")
    finally:
        await sender.close()


class _GenerationTransport:
    def __init__(self) -> None:
        self.calls = 0
        self.started = (asyncio.Event(), asyncio.Event())
        self.cancelled = (asyncio.Event(), asyncio.Event())
        self.release_second = asyncio.Event()

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools, system
        index = self.calls
        self.calls += 1
        self.started[index].set()
        try:
            if index == 0:
                await asyncio.Future()
            else:
                await self.release_second.wait()
                yield TextDelta(index=0, delta="done")
                yield IterationEnd(iteration=2, stop_reason=StopReason.end_turn, usage=Usage(0, 0))
        except asyncio.CancelledError:
            self.cancelled[index].set()
            raise


async def test_local_interrupt_is_generation_checked_and_input_admission_reports_owner() -> None:
    transport = _GenerationTransport()
    admitted: list[tuple[str, str | None]] = []

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    set_background_input_admitted_handler(lambda agent_id, turn_id: admitted.append((agent_id, turn_id)))
    agent_id = _spawn_id(await spawn_agent(task="first"))
    await asyncio.wait_for(transport.started[0].wait(), timeout=1)

    assert await enqueue_local_agent_prompt(agent_id, "second")
    captured_turn_id = admitted[-1][1]
    assert captured_turn_id is not None
    assert interrupt_local_agent_turn(agent_id, captured_turn_id)
    await asyncio.wait_for(transport.cancelled[0].wait(), timeout=1)
    await asyncio.wait_for(transport.started[1].wait(), timeout=1)

    assert not interrupt_local_agent_turn(agent_id, captured_turn_id)
    await asyncio.sleep(0)
    assert not transport.cancelled[1].is_set()

    transport.release_second.set()
    await wait_local_background_agents_idle([agent_id])


async def test_stop_agent_releases_waiting_local_prompt(tmp_path: Path) -> None:
    transport = _InterruptibleTransport()
    sender = await PeerServer("sender", kind="test", handler=_noop_handler, project=str(tmp_path)).start()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    try:
        result = await spawn_agent(task="block")
        agent_id = result.split("agent_id=", 1)[1].split(" ", 1)[0]
        await asyncio.wait_for(transport.started.wait(), timeout=1)

        waiter = asyncio.create_task(enqueue_local_agent_prompt(agent_id, "queued", wait=True))
        await asyncio.sleep(0)
        stopped = await stop_agent(agent_id=agent_id, reason="done")
        assert stopped.startswith("Sent stop")
        assert await asyncio.wait_for(waiter, timeout=1)
    finally:
        await sender.close()


class _BatchCapturingTransport:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(messages)
        yield TextDelta(index=0, delta="done")
        yield IterationEnd(iteration=len(self.calls), stop_reason=StopReason.end_turn, usage=Usage(0, 0))


async def test_local_agent_message_batch_stays_distinct_in_one_turn(tmp_path: Path) -> None:
    transport = _BatchCapturingTransport()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    result = await spawn_agent(task="initial")
    agent_id = result.split("agent_id=", 1)[1].split(" ", 1)[0]
    await wait_local_background_agents_idle([agent_id])
    committed: list[str] = []
    first = Message(role="user", content=[TextBlock(text="first")])
    second = Message(role="user", content=[TextBlock(text="second")])

    async def input_committed() -> None:
        committed.append("yes")

    delivered = await enqueue_local_agent_messages(
        agent_id,
        (first, second),
        wait=True,
        on_input_committed=input_committed,
    )

    assert delivered
    assert committed == ["yes"]
    assert transport.calls[1][-2:] == [first, second]


async def test_local_agent_context_batch_does_not_start_replacement_turn(tmp_path: Path) -> None:
    transport = _BatchCapturingTransport()
    context = MemoryContextStore()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), context

    set_spawn_agent_factory(factory)
    result = await spawn_agent(task="initial")
    agent_id = result.split("agent_id=", 1)[1].split(" ", 1)[0]
    await wait_local_background_agents_idle([agent_id])
    notice = Message(role="user", content=[TextBlock(text="interrupt notice")])

    delivered = await enqueue_local_agent_context(agent_id, (notice,), wait=True)

    assert delivered
    assert len(transport.calls) == 1
    assert (await context.get_history())[-1] == notice


class _DelayedTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        prompt = messages[-1].content[0].text  # type: ignore[attr-defined]
        self.calls.append(prompt)
        self.started.set()
        await self.release.wait()
        yield TextDelta(index=0, delta="done")
        yield IterationEnd(iteration=len(self.calls), stop_reason=StopReason.end_turn, usage=Usage(0, 0))


async def test_wait_local_background_agents_idle_waits_for_current_turn(tmp_path: Path) -> None:
    transport = _DelayedTransport()
    sender = await PeerServer("sender", kind="test", handler=_noop_handler, project=str(tmp_path)).start()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    try:
        result = await spawn_agent(task="slow")
        agent_id = result.split("agent_id=", 1)[1].split(" ", 1)[0]
        await asyncio.wait_for(transport.started.wait(), timeout=1)

        waiter = asyncio.create_task(wait_local_background_agents_idle([agent_id]))
        await asyncio.sleep(0)
        assert not waiter.done()
        transport.release.set()
        await asyncio.wait_for(waiter, timeout=1)
        assert is_local_background_agent(agent_id)

        stopped = await stop_agent(agent_id=agent_id, reason="done")
        assert stopped.startswith("Sent stop")
    finally:
        await sender.close()


class _LoopingTransport:
    """A model that keeps calling a tool and never answers."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        yield ToolUseStart(index=0, tool_use_id=f"c{self.calls}", name="noop")
        yield ToolInputDelta(index=0, tool_use_id=f"c{self.calls}", partial_json="{}")
        yield IterationEnd(iteration=self.calls, stop_reason=StopReason.tool_use, usage=Usage(0, 0))


async def test_running_out_of_iterations_reaches_the_parent(tmp_path: Path) -> None:
    # It does not raise, so the exception handler never sees it. Before it was
    # recorded here, the parent saw the same idle as an agent that answered and
    # the reason existed only as a log line.
    async def noop() -> str:
        return "ok"

    transport = _LoopingTransport()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        agent = Agent(
            system="child",
            tools=[Tool(name="noop", handler=noop)],
            transport=transport,
            max_iterations=2,
        )
        return agent, MemoryContextStore()

    set_spawn_agent_factory(factory)
    result = await spawn_agent(task="go")
    agent_id = result.split("agent_id=", 1)[1].split(" ", 1)[0]
    await _wait_for(lambda: background_agent_state(agent_id)[1] is not None)

    state, error = background_agent_state(agent_id)
    assert state == "idle"
    assert error is not None
    assert "2 iterations" in error


async def test_a_spawned_agent_gets_a_two_word_name() -> None:
    # Named from its task, an agent produced ids like
    # spawn_agent-You-are-analyzing-the-axio-monorepo--4058046-fc69dbe1, which
    # every later call had to carry.
    from axio_tools_agents.names import ADJECTIVES, SURNAMES

    transport = _LoopingTransport()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        agent = Agent(system="child", tools=[], transport=transport, max_iterations=1)
        return agent, MemoryContextStore()

    set_spawn_agent_factory(factory)
    result = await spawn_agent(task="Analyse the whole monorepo and report back in detail")

    name = result.split("name=", 1)[1].strip().strip("'\"").split(".")[0].strip("'\"")
    adjective, _, surname = name.partition("_")
    assert adjective in ADJECTIVES, name
    assert surname in SURNAMES, name


async def test_an_explicit_name_is_kept() -> None:
    transport = _LoopingTransport()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", tools=[], transport=transport, max_iterations=1), MemoryContextStore()

    set_spawn_agent_factory(factory)
    result = await spawn_agent(task="anything", name="docs-audit")

    assert "name='docs-audit'" in result


class _AnsweringTransport:
    """A model that answers every prompt in one iteration."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(messages[-1].content[0].text)  # type: ignore[attr-defined]
        yield TextDelta(index=0, delta="answered")
        yield IterationEnd(iteration=len(self.calls), stop_reason=StopReason.end_turn, usage=Usage(0, 0))


class _OwnerAnsweringTransport(_AnsweringTransport):
    def __init__(self) -> None:
        super().__init__()
        self.owners: list[str | None] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        self.owners.append(notify.current_owner())
        async for event in super().stream(messages, tools, system):
            yield event


class _ClosingMemoryContext(MemoryContextStore):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _spawn_id(result: str) -> str:
    agent_id = result.split("agent_id=", 1)[1].split(" ", 1)[0]
    _watch_bus_owner(agent_id)
    return agent_id


async def _start_parent(tmp_path: Path, name: str = "parent") -> PeerServer:
    peer = await PeerServer(name, kind="test", handler=_noop_handler, project=str(tmp_path)).start()
    _watch_bus_owner(peer.id)
    return peer


async def test_a_finished_child_turn_is_announced_to_its_parent_once(tmp_path: Path) -> None:
    transport = _AnsweringTransport()
    parent = await _start_parent(tmp_path)
    received: list[str] = []
    notify.add_listener(parent.id, received.append)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    try:
        agent_id = _spawn_id(await spawn_agent(task="do the thing"))
        await _wait_for(lambda: len(received) == 1)

        assert agent_id in received[0]
        assert "finished its turn and is idle" in received[0]

        # Stopping is the parent's own doing, so it is not news to report back.
        assert (await stop_agent(agent_id=agent_id, reason="done")).startswith("Sent stop")
        await _wait_for(lambda: not is_local_background_agent(agent_id))
        assert len(received) == 1
    finally:
        await parent.close()


async def test_a_failed_child_turn_carries_the_error_to_its_parent(tmp_path: Path) -> None:
    async def noop() -> str:
        return "ok"

    transport = _LoopingTransport()
    parent = await _start_parent(tmp_path)
    received: list[str] = []
    notify.add_listener(parent.id, received.append)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        agent = Agent(
            system="child",
            tools=[Tool(name="noop", handler=noop)],
            transport=transport,
            max_iterations=2,
        )
        return agent, MemoryContextStore()

    set_spawn_agent_factory(factory)
    try:
        _spawn_id(await spawn_agent(task="go"))
        await _wait_for(lambda: len(received) == 1)

        assert "turn failed" in received[0]
        assert "2 iterations" in received[0]
    finally:
        await parent.close()


async def test_an_interrupted_child_turn_is_not_announced_as_finished(tmp_path: Path) -> None:
    # Nothing raises out of a cancelled turn, so the agent went idle with no
    # error to show: reported as "finished", an interrupt would read as an answer.
    transport = _InterruptibleTransport()
    parent = await _start_parent(tmp_path)
    received: list[str] = []
    notify.add_listener(parent.id, received.append)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    try:
        agent_id = _spawn_id(await spawn_agent(task="block"))
        await asyncio.wait_for(transport.started.wait(), timeout=1)

        assert (await interrupt_agent(agent_id=agent_id, reason="enough")).startswith("Sent interrupt")
        await _wait_for(lambda: len(received) == 1)

        assert "turn was interrupted" in received[0]
        assert "finished its turn" not in received[0]
    finally:
        await parent.close()


async def test_a_turn_already_reported_to_the_parent_is_not_announced_again(tmp_path: Path) -> None:
    transport = _DelayedTransport()
    parent = await _start_parent(tmp_path)
    received: list[str] = []
    notify.add_listener(parent.id, received.append)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    try:
        agent_id = _spawn_id(await spawn_agent(task="do the thing"))
        await asyncio.wait_for(transport.started.wait(), timeout=1)

        mark_background_report_delivered(agent_id, parent.id)
        transport.release.set()
        await asyncio.wait_for(wait_local_background_agents_idle([agent_id]), timeout=1)

        assert received == []
    finally:
        await parent.close()


async def test_a_report_delivered_elsewhere_still_announces_to_the_parent(tmp_path: Path) -> None:
    transport = _DelayedTransport()
    parent = await _start_parent(tmp_path)
    received: list[str] = []
    notify.add_listener(parent.id, received.append)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    try:
        agent_id = _spawn_id(await spawn_agent(task="do the thing"))
        await asyncio.wait_for(transport.started.wait(), timeout=1)

        mark_background_report_delivered(agent_id, "some-other-peer")
        transport.release.set()
        await asyncio.wait_for(wait_local_background_agents_idle([agent_id]), timeout=1)

        assert len(received) == 1
        assert "finished its turn and is idle" in received[0]
    finally:
        await parent.close()


async def test_an_idle_child_takes_a_notification_as_its_next_prompt() -> None:
    transport = _AnsweringTransport()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    agent_id = _spawn_id(await spawn_agent(task="start something detached"))
    await _wait_for(lambda: background_agent_state(agent_id)[0] == "idle")

    notify.post("[background task t-1] shell: finished\nexit 0", owner=agent_id)

    await _wait_for(lambda: len(transport.calls) == 2)
    assert "[background task t-1] shell: finished" in transport.calls[1]


async def test_a_stopped_agent_leaves_no_listener_or_queue_behind() -> None:
    transport = _AnsweringTransport()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_spawn_agent_factory(factory)
    agent_id = _spawn_id(await spawn_agent(task="do the thing"))
    await _wait_for(lambda: background_agent_state(agent_id)[0] == "idle")

    assert (await stop_agent(agent_id=agent_id, reason="done")).startswith("Sent stop")
    await _wait_for(lambda: not is_local_background_agent(agent_id))

    assert notify.drain(agent_id) == []
    # A detached call finishing after the agent died must not wake anything: with
    # the listener gone the text only waits, and monitor(tasks=[...]) still has it.
    notify.post("[background task t-1] shell: finished", owner=agent_id)
    assert notify.drain(agent_id) == ["[background task t-1] shell: finished"]
    assert len(transport.calls) == 1


async def test_run_agent_streams_one_shot_child_with_parent_tool_correlation(tmp_path: Path) -> None:
    transport = _OwnerAnsweringTransport()
    parent = await _start_parent(tmp_path)
    hub = SessionEventHub(session_id="session")
    envelopes: list[AgentEventEnvelope] = []
    child_context = _ClosingMemoryContext()

    async def collect(envelope: AgentEventEnvelope) -> None:
        envelopes.append(envelope)

    hub.subscribe(collect)
    set_session_event_hub(hub)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        assert not inherit_context
        return Agent(system="child", transport=transport), child_context

    set_run_agent_factory(factory)
    tool: Tool[object] = Tool(name="run_agent", handler=run_agent, concurrency=1, detachable=False)
    caller = Agent(system="parent", transport=transport, tools=[tool])
    block = ToolUseBlock(id="parent-call", name="run_agent", input={"task": "inspect"})

    try:
        with peer_context(parent):
            [result] = await caller.dispatch_tools([block], iteration=3)
    finally:
        await parent.close()

    assert not result.is_error
    assert result.content == "answered"
    child_ids = {envelope.agent_id for envelope in envelopes}
    assert len(child_ids) == 1
    [child_id] = child_ids
    assert not is_local_background_agent(child_id)
    assert transport.owners == [child_id]
    assert child_context.closed
    assert all(envelope.execution_mode is ExecutionMode.FOREGROUND for envelope in envelopes)
    assert all(envelope.parent_tool_use_id == "parent-call" for envelope in envelopes)
    events = [envelope.event for envelope in envelopes]
    assert isinstance(events[0], AgentStarted)
    assert isinstance(events[1], ForegroundEntered)
    assert any(isinstance(event, TextDelta) and event.delta == "answered" for event in events)
    assert isinstance(events[-2], ForegroundExited)
    assert isinstance(events[-1], AgentStopped)
    assert not any(isinstance(event, OutcomeDelivered) for event in events)


async def test_foreground_answer_is_delivered_by_one_correlated_parent_commit(tmp_path: Path) -> None:
    parent = await _start_parent(tmp_path)
    hub = SessionEventHub()
    envelopes: list[AgentEventEnvelope] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        envelopes.append(envelope)

    hub.subscribe(collect)
    set_session_event_hub(hub)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=_AnsweringTransport()), MemoryContextStore()

    set_run_agent_factory(factory)
    parent_agent = Agent(
        system="parent",
        transport=StubTransport(
            [
                make_tool_use_response("run_agent", "foreground-call", {"task": "answer"}),
                make_text_response("parent done"),
            ]
        ),
        tools=[Tool(name="run_agent", handler=run_agent, concurrency=1, detachable=False)],
    )
    inner = MemoryContextStore()
    context = ObservedContextStore(inner, hub)
    identity = new_turn_identity(
        agent_id=parent.id,
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        context_id=context.session_id,
    )

    try:
        with peer_context(parent):
            outcome = await observe_agent_turn(
                agent=parent_agent,
                context=context,
                prompt="delegate",
                identity=identity,
                hub=hub,
            )
    finally:
        await parent.close()

    assert outcome.succeeded
    result_commits: list[tuple[int, MessageCommitted]] = []
    for envelope in envelopes:
        event = envelope.event
        if isinstance(event, MessageCommitted) and any(
            isinstance(block, ToolResultBlock) for block in event.message.content
        ):
            result_commits.append((envelope.seq, event))
    assert len(result_commits) == 1
    commit_seq, commit_event = result_commits[0]
    [result] = [block for block in commit_event.message.content if isinstance(block, ToolResultBlock)]
    assert result.tool_use_id == "foreground-call"
    assert result.content == "answered"
    assert not result.is_error
    child_envelopes = [
        envelope
        for envelope in envelopes
        if envelope.agent_id != parent.id and envelope.execution_mode is ExecutionMode.FOREGROUND
    ]
    assert child_envelopes
    assert all(envelope.parent_tool_use_id == result.tool_use_id for envelope in child_envelopes)
    assert commit_seq > max(envelope.seq for envelope in child_envelopes)
    assert not any(
        isinstance(envelope.event, OutcomeDelivered) and envelope.execution_mode is ExecutionMode.FOREGROUND
        for envelope in envelopes
    )


@pytest.mark.parametrize("blocked_event_type", [AgentStarted, ForegroundEntered])
async def test_foreground_startup_cancellation_cleans_resources(
    tmp_path: Path,
    blocked_event_type: type[object],
) -> None:
    parent = await _start_parent(tmp_path)
    hub = SessionEventHub()
    blocked = asyncio.Event()
    envelopes: list[AgentEventEnvelope] = []
    child_context = _ClosingMemoryContext()
    transport = _AnsweringTransport()

    async def block_lifecycle(envelope: AgentEventEnvelope) -> None:
        envelopes.append(envelope)
        if isinstance(envelope.event, blocked_event_type):
            blocked.set()
            await asyncio.Event().wait()

    hub.subscribe(block_lifecycle)
    set_session_event_hub(hub)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), child_context

    set_run_agent_factory(factory)
    with peer_context(parent):
        task = asyncio.create_task(run_agent(task="never starts", name="startup-cancel"))
        await asyncio.wait_for(blocked.wait(), timeout=1)
        [child_id] = {envelope.agent_id for envelope in envelopes}
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)

    assert task.done()
    assert child_context.closed
    assert transport.calls == []
    assert not is_local_background_agent(child_id)
    marker = "owner was discarded"
    notify.post(marker, owner=child_id)
    assert notify.drain(child_id) == [marker]
    await parent.close()


async def test_background_startup_cancellation_cleans_peer_listener_and_context(tmp_path: Path) -> None:
    hub = SessionEventHub()
    blocked = asyncio.Event()
    child_id: str | None = None
    child_context = _ClosingMemoryContext()

    async def block_started(envelope: AgentEventEnvelope) -> None:
        nonlocal child_id
        if isinstance(envelope.event, AgentStarted) and envelope.execution_mode is ExecutionMode.BACKGROUND:
            child_id = envelope.agent_id
            blocked.set()
            await asyncio.Event().wait()

    hub.subscribe(block_started)
    set_session_event_hub(hub)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=_AnsweringTransport()), child_context

    set_spawn_agent_factory(factory)
    task = asyncio.create_task(spawn_agent(task="never starts", name="startup-cancel"))
    await asyncio.wait_for(blocked.wait(), timeout=1)
    assert child_id is not None
    [started_record] = [record for record in await list_peer_records() if record.id == child_id]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert task.done()
    assert child_context.closed
    assert not is_local_background_agent(child_id)
    assert all(record.id != child_id for record in await list_peer_records())
    assert not Path(started_record.socket_path).exists()
    marker = "listener was discarded"
    notify.post(marker, owner=child_id)
    assert notify.drain(child_id) == [marker]


async def test_cancelled_parent_never_records_foreground_outcome_as_delivered(tmp_path: Path) -> None:
    slow_started = asyncio.Event()
    slow_closed = asyncio.Event()

    async def slow_sibling() -> str:
        slow_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            slow_closed.set()
        return "unreachable"

    parent_transport = StubTransport(
        [
            [
                ToolUseStart(index=0, tool_use_id="foreground-call", name="run_agent"),
                ToolInputDelta(
                    index=0,
                    tool_use_id="foreground-call",
                    partial_json='{"task":"answer"}',
                ),
                ToolUseStart(index=1, tool_use_id="slow-call", name="slow_sibling"),
                ToolInputDelta(index=1, tool_use_id="slow-call", partial_json="{}"),
                IterationEnd(iteration=1, stop_reason=StopReason.tool_use, usage=Usage(0, 0)),
            ],
            make_text_response("unreachable"),
        ]
    )
    child_context = _ClosingMemoryContext()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=_AnsweringTransport()), child_context

    set_run_agent_factory(factory)
    hub = SessionEventHub()
    envelopes: list[AgentEventEnvelope] = []
    child_stopped = asyncio.Event()

    async def collect(envelope: AgentEventEnvelope) -> None:
        envelopes.append(envelope)
        if (
            envelope.execution_mode is ExecutionMode.FOREGROUND
            and isinstance(envelope.event, AgentStopped)
            and envelope.event.status is TurnStatus.SUCCEEDED
        ):
            child_stopped.set()

    hub.subscribe(collect)
    set_session_event_hub(hub)
    inner = MemoryContextStore()
    context = ObservedContextStore(inner, hub)
    parent = await _start_parent(tmp_path)
    parent_agent = Agent(
        system="parent",
        transport=parent_transport,
        tools=[
            Tool(name="run_agent", handler=run_agent, concurrency=1, detachable=False),
            Tool(name="slow_sibling", handler=slow_sibling),
        ],
    )
    identity = new_turn_identity(
        agent_id=parent.id,
        parent_agent_id=None,
        execution_mode=ExecutionMode.FOREGROUND,
        context_id=context.session_id,
    )

    with peer_context(parent):
        task = asyncio.create_task(
            observe_agent_turn(
                agent=parent_agent,
                context=context,
                prompt="delegate and wait",
                identity=identity,
                hub=hub,
            )
        )
        await asyncio.wait_for(slow_started.wait(), timeout=1)
        await asyncio.wait_for(child_stopped.wait(), timeout=1)
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)

    await parent.close()
    assert child_context.closed
    assert slow_closed.is_set()
    assert not any(
        isinstance(envelope.event, OutcomeDelivered) and envelope.execution_mode is ExecutionMode.FOREGROUND
        for envelope in envelopes
    )
    committed_results = [
        block
        for envelope in envelopes
        if isinstance(envelope.event, MessageCommitted)
        for block in envelope.event.message.content
        if isinstance(block, ToolResultBlock)
    ]
    foreground_results = [block for block in committed_results if block.tool_use_id == "foreground-call"]
    assert len(foreground_results) == 1
    assert foreground_results[0].is_error
    assert foreground_results[0].content == "[interrupted by user]"
    child_envelopes = [
        envelope
        for envelope in envelopes
        if envelope.agent_id != parent.id and envelope.execution_mode is ExecutionMode.FOREGROUND
    ]
    assert child_envelopes
    assert all(envelope.parent_tool_use_id == "foreground-call" for envelope in child_envelopes)


async def test_run_agent_failure_becomes_one_parent_tool_error(tmp_path: Path) -> None:
    async def noop() -> str:
        return "ok"

    parent = await _start_parent(tmp_path)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return (
            Agent(
                system="child",
                transport=_LoopingTransport(),
                tools=[Tool(name="noop", handler=noop)],
                max_iterations=1,
            ),
            MemoryContextStore(),
        )

    set_run_agent_factory(factory)
    caller = Agent(
        system="parent",
        transport=_AnsweringTransport(),
        tools=[Tool(name="run_agent", handler=run_agent, concurrency=1, detachable=False)],
    )

    try:
        with peer_context(parent):
            results = await caller.dispatch_tools(
                [ToolUseBlock(id="parent-call", name="run_agent", input={"task": "loop"})],
                iteration=1,
            )
    finally:
        await parent.close()

    assert len(results) == 1
    assert results[0].is_error
    assert "1 iterations" in str(results[0].content)


async def test_cancelling_parent_tool_call_closes_foreground_child_and_restores_lifecycle(tmp_path: Path) -> None:
    transport = _InterruptibleTransport()
    parent = await _start_parent(tmp_path)
    hub = SessionEventHub()
    events: list[object] = []

    async def collect(envelope: AgentEventEnvelope) -> None:
        events.append(envelope.event)

    hub.subscribe(collect)
    set_session_event_hub(hub)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_run_agent_factory(factory)
    caller = Agent(
        system="parent",
        transport=_AnsweringTransport(),
        tools=[Tool(name="run_agent", handler=run_agent, concurrency=1, detachable=False)],
    )
    with peer_context(parent):
        task = asyncio.create_task(
            caller.dispatch_tools(
                [ToolUseBlock(id="parent-call", name="run_agent", input={"task": "block"})],
                iteration=1,
            )
        )
        await transport.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    await parent.close()
    await asyncio.wait_for(transport.interrupted.wait(), timeout=1)
    assert ForegroundExited(status=TurnStatus.CANCELLED) in events
    assert AgentStopped(status=TurnStatus.CANCELLED) in events
    assert not any(isinstance(event, OutcomeDelivered) for event in events)


class _ConcurrentForegroundTransport:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(0.02)
            yield TextDelta(index=0, delta="done")
            yield IterationEnd(iteration=1, stop_reason=StopReason.end_turn, usage=Usage(0, 0))
        finally:
            self.active -= 1


async def test_run_agent_tool_serializes_sibling_foreground_runs() -> None:
    transport = _ConcurrentForegroundTransport()

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=transport), MemoryContextStore()

    set_run_agent_factory(factory)
    tool: Tool[object] = Tool(name="run_agent", handler=run_agent, concurrency=1, detachable=False)

    first, second = await asyncio.gather(tool(task="one"), tool(task="two"))
    assert first == "done"
    assert second == "done"
    assert transport.maximum == 1


async def test_background_outcome_delivery_does_not_depend_on_renderer(tmp_path: Path) -> None:
    outcomes: list[TurnOutcome] = []
    parent = await _start_parent(tmp_path)
    notifications: list[str] = []
    notify.add_listener(parent.id, notifications.append)

    async def factory(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
        return Agent(system="child", transport=_AnsweringTransport()), MemoryContextStore()

    async def collect_outcome(outcome: TurnOutcome) -> None:
        outcomes.append(outcome)

    set_spawn_agent_factory(factory)
    set_background_outcome_handler(collect_outcome)
    try:
        with peer_context(parent):
            agent_id = _spawn_id(await spawn_agent(task="answer without a renderer"))
        await _wait_for(lambda: len(outcomes) == 1)

        assert outcomes[0].identity.agent_id == agent_id
        assert outcomes[0].text == "answered"
        assert outcomes[0].status is TurnStatus.SUCCEEDED
        assert notifications == []
    finally:
        await parent.close()


async def test_spawn_agent_not_configured_raises_handler_error() -> None:
    with pytest.raises(HandlerError, match="spawn_agent is not configured"):
        await spawn_agent(task="anything")


async def test_run_agent_not_configured_raises_handler_error() -> None:
    with pytest.raises(HandlerError, match="run_agent is not configured"):
        await run_agent(task="anything")
