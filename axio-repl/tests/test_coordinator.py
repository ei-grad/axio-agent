from __future__ import annotations

import asyncio
from dataclasses import fields, replace

import pytest
from axio.blocks import TextBlock
from axio.messages import Message
from axio_tools_agents.runtime import (
    AgentEventEnvelope,
    ExecutionMode,
    InputBuffered,
    InputClaimed,
    InputDelivered,
    InputRecalled,
    RuntimeEvent,
)

from axio_repl._coordinator import (
    ClaimBatch,
    ContextArrival,
    ForegroundCoordinatorState,
    ForegroundOperation,
    ForegroundPhase,
    PendingInputCoordinator,
    PendingInputState,
    PendingInputStatus,
    PendingUserEntry,
    claim_batch_arrivals,
    ordered_messages,
)
from axio_repl._input import ExitArmingState, InterruptRequested


def _entry(identifier: str, seq: int, text: str, target: str = "main") -> PendingUserEntry:
    return PendingUserEntry(
        id=identifier,
        arrival_seq=seq,
        text=text,
        intended_target_agent_id=target,
    )


def _text(message: Message) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


def test_interrupt_event_captures_identity_but_carries_no_editor_text() -> None:
    event = InterruptRequested(target_agent_id="main", captured_turn_id="turn-1")

    assert event.target_agent_id == "main"
    assert event.captured_turn_id == "turn-1"
    assert "text" not in {field.name for field in fields(event)}


def test_foreground_reducer_makes_repeated_interrupt_idempotent() -> None:
    running = ForegroundCoordinatorState().start("main", "turn-1")

    cancelling, accepted = running.request_interrupt("main", "turn-1")
    duplicate, accepted_again = cancelling.request_interrupt("main", "turn-1")

    assert accepted
    assert cancelling.phase is ForegroundPhase.CANCELLING
    assert not accepted_again
    assert duplicate is cancelling


def test_foreground_reducer_ignores_stale_completion_and_accepts_replacement_interrupt() -> None:
    running = ForegroundCoordinatorState().start("main", "turn-1")

    assert running.complete("main", "stale") is running
    idle = running.complete("main", "turn-1")
    replacement = idle.start("main", "turn-2")
    cancelling, accepted = replacement.request_interrupt("main", "turn-2")

    assert idle.phase is ForegroundPhase.IDLE
    assert accepted
    assert cancelling.active_turn_id("main") == "turn-2"


def test_foreground_reducer_preserves_operation_until_shutdown_settles() -> None:
    running = ForegroundCoordinatorState().start("main", "turn-1")

    stopping = running.request_shutdown("sigterm")
    settled = stopping.complete("main", "turn-1")
    stopped = settled.mark_stopped()

    assert stopping.phase is ForegroundPhase.STOPPING
    assert stopping.active_turn_id("main") == "turn-1"
    assert settled.active_turn_id("main") is None
    assert stopped.phase is ForegroundPhase.STOPPED
    assert stopped.shutdown_reason == "sigterm"


def test_foreground_reducer_rejects_invalid_states_and_transitions() -> None:
    with pytest.raises(ValueError, match="target_agent_id"):
        ForegroundOperation("", "turn")
    with pytest.raises(ValueError, match="turn_id"):
        ForegroundOperation("main", "")
    with pytest.raises(ValueError, match="requires an operation"):
        ForegroundCoordinatorState(phase=ForegroundPhase.RUNNING)
    with pytest.raises(ValueError, match="cannot retain an operation"):
        ForegroundCoordinatorState(operation=ForegroundOperation("main", "turn"))
    with pytest.raises(ValueError, match="requires a shutdown reason"):
        ForegroundCoordinatorState(phase=ForegroundPhase.STOPPING)
    with pytest.raises(ValueError, match="cannot carry a shutdown reason"):
        ForegroundCoordinatorState(shutdown_reason="unexpected")

    running = ForegroundCoordinatorState().start("main", "turn")
    with pytest.raises(RuntimeError, match="cannot start"):
        running.start("main", "replacement")
    with pytest.raises(ValueError, match="target_agent_id"):
        running.request_interrupt("", "turn")
    assert running.request_interrupt("main", None) == (running, True)
    stale, accepted = running.request_interrupt("child", "other-turn")
    assert accepted
    assert stale.phase is ForegroundPhase.RUNNING

    with pytest.raises(ValueError, match="shutdown reason"):
        running.request_shutdown("")
    stopping = running.request_shutdown("sigterm")
    assert stopping.request_shutdown("sigterm") is stopping
    with pytest.raises(RuntimeError, match="cannot change"):
        stopping.request_shutdown("sigint")
    stopped = stopping.complete("main", "turn").mark_stopped()
    assert stopped.request_shutdown("sigterm") is stopped
    with pytest.raises(RuntimeError, match="cannot stop"):
        ForegroundCoordinatorState().mark_stopped()


def test_recall_returns_every_pending_input_as_one_editor_value() -> None:
    state = PendingInputState().admit(_entry("one", 1, "first")).admit(_entry("two", 2, "second"))

    recalled_state, batch = state.recall_all()

    assert batch is not None
    assert batch.source_ids == ("one", "two")
    assert batch.editor_text == "first\n\nsecond"
    assert recalled_state.pending == ()
    assert [entry.status for entry in recalled_state.entries] == [
        PendingInputStatus.RECALLED,
        PendingInputStatus.RECALLED,
    ]


def test_resubmitting_recalled_text_creates_one_new_chronological_entry() -> None:
    state = PendingInputState().admit(_entry("one", 1, "first")).admit(_entry("two", 2, "second"))
    recalled_state, batch = state.recall_all()
    assert batch is not None

    resubmitted = recalled_state.admit(_entry("combined", 3, batch.editor_text))

    assert [entry.id for entry in resubmitted.pending] == ["combined"]
    assert resubmitted.pending[0].text == "first\n\nsecond"


def test_interrupt_claims_all_pending_inputs_for_current_agent_without_joining() -> None:
    state = PendingInputState().admit(_entry("one", 1, "first", "main")).admit(_entry("two", 2, "second", "child"))

    claimed_state, batch = state.claim_all_for_interrupt("child")

    assert batch is not None
    assert claimed_state.pending == ()
    assert [entry.claimed_target_agent_id for entry in batch.entries] == ["child", "child"]
    messages = tuple(arrival.message for arrival in claim_batch_arrivals(batch))
    assert [_text(message) for message in messages] == ["first", "second"]
    assert messages[0] is not messages[1]


def test_normal_claim_does_not_take_another_agents_pending_input() -> None:
    state = (
        PendingInputState()
        .admit(_entry("main", 1, "for main", "main"))
        .admit(_entry("child", 2, "for child", "child"))
    )

    next_state, batch = state.claim_for_target("main")

    assert batch is not None
    assert [entry.id for entry in batch.entries] == ["main"]
    assert [entry.id for entry in next_state.pending] == ["child"]


def test_claimed_inputs_can_be_marked_delivered_exactly_once() -> None:
    state = PendingInputState().admit(_entry("one", 1, "first"))
    claimed_state, batch = state.claim_all_for_interrupt("main")
    assert batch is not None

    delivered = claimed_state.mark_delivered(())

    assert delivered == claimed_state

    delivered = claimed_state.mark_delivered(("one",))
    assert delivered.entries[0].status is PendingInputStatus.DELIVERED
    with pytest.raises(ValueError, match="is not claimed"):
        delivered.mark_delivered(("one",))


def test_ordered_messages_merge_sources_by_sequence_without_joining() -> None:
    interactive = Message(role="user", content=[TextBlock(text="interactive")])
    peer = Message(role="user", content=[TextBlock(text="peer")])
    interrupted = Message(role="user", content=[TextBlock(text="interrupted")])
    arrivals = (
        ContextArrival(3, "main", interrupted, "interrupt"),
        ContextArrival(1, "main", interactive, "interactive"),
        ContextArrival(2, "main", peer, "peer"),
        ContextArrival(1, "child", Message(role="user", content=[TextBlock(text="child")]), "interactive"),
    )

    messages = ordered_messages(arrivals, "main")

    assert messages == (interactive, peer, interrupted)
    assert [_text(message) for message in messages] == ["interactive", "peer", "interrupted"]


def test_ordered_messages_stop_at_interrupt_barrier_watermark() -> None:
    before = Message(role="user", content=[TextBlock(text="before")])
    after = Message(role="user", content=[TextBlock(text="after")])
    arrivals = (
        ContextArrival(4, "main", after, "peer"),
        ContextArrival(2, "main", before, "peer"),
    )

    assert ordered_messages(arrivals, "main", through_seq=3) == (before,)


def test_pending_state_rejects_duplicate_or_non_monotonic_admission() -> None:
    state = PendingInputState().admit(_entry("one", 2, "first"))

    with pytest.raises(ValueError, match="duplicate"):
        state.admit(_entry("one", 3, "again"))
    with pytest.raises(ValueError, match="monotonically"):
        state.admit(_entry("two", 1, "older"))


def test_pending_values_reject_invalid_and_crossed_lifecycle_data() -> None:
    with pytest.raises(ValueError, match="id must not be empty"):
        _entry("", 1, "text")
    with pytest.raises(ValueError, match="arrival_seq"):
        _entry("one", 0, "text")
    with pytest.raises(ValueError, match="text must not be empty"):
        _entry("one", 1, "")
    with pytest.raises(ValueError, match="intended_target_agent_id"):
        _entry("one", 1, "text", "")
    with pytest.raises(ValueError, match="require claimed_target_agent_id"):
        replace(_entry("one", 1, "text"), status=PendingInputStatus.CLAIMED)
    with pytest.raises(ValueError, match="unclaimed entries"):
        replace(_entry("one", 1, "text"), claimed_target_agent_id="main")

    claimed = replace(
        _entry("one", 1, "text"),
        status=PendingInputStatus.CLAIMED,
        claimed_target_agent_id="main",
    )
    with pytest.raises(ValueError, match="must not be empty"):
        ClaimBatch((), "main")
    with pytest.raises(ValueError, match="target_agent_id"):
        ClaimBatch((claimed,), "")
    with pytest.raises(ValueError, match="must be claimed"):
        ClaimBatch((_entry("pending", 2, "text"),), "main")
    with pytest.raises(ValueError, match="share the batch target"):
        ClaimBatch((claimed,), "child")


def test_pending_state_rejects_invalid_snapshots_and_delivery_requests() -> None:
    first = _entry("one", 1, "first")
    with pytest.raises(ValueError, match="ids must be unique"):
        PendingInputState((first, replace(first, arrival_seq=2)))
    with pytest.raises(ValueError, match="sequences must be unique"):
        PendingInputState((first, _entry("two", 1, "second")))
    with pytest.raises(ValueError, match="ordered"):
        PendingInputState((_entry("two", 2, "second"), first))

    claimed = replace(first, status=PendingInputStatus.CLAIMED, claimed_target_agent_id="main")
    with pytest.raises(ValueError, match="only pending"):
        PendingInputState().admit(claimed)
    with pytest.raises(ValueError, match="must be unique"):
        PendingInputState((claimed,)).mark_delivered(("one", "one"))
    with pytest.raises(ValueError, match="unknown"):
        PendingInputState((claimed,)).mark_delivered(("missing",))


def test_empty_pending_transitions_are_explicit_noops() -> None:
    empty = PendingInputState()

    assert empty.recall_all() == (empty, None)
    assert empty.claim_oldest() == (empty, None)
    assert empty.claim_for_target("main") == (empty, None)
    with pytest.raises(ValueError, match="target_agent_id"):
        empty.claim_for_target("")

    one = empty.admit(_entry("one", 1, "first", "child"))
    claimed_state, claimed = one.claim_oldest()
    assert claimed is not None
    assert claimed.target_agent_id == "child"
    assert claimed_state.pending == ()


def test_double_eof_only_shuts_down_inside_two_second_window() -> None:
    state, shutdown = ExitArmingState().press(10.0)
    assert not shutdown
    assert state.deadline == 12.0

    state, shutdown = state.press(11.5)
    assert shutdown
    assert state.deadline is None

    state, shutdown = ExitArmingState().press(20.0)
    assert not shutdown
    expired = state.expire(22.1)
    assert expired.deadline is None
    rearmed, shutdown = expired.press(23.0)
    assert not shutdown
    assert rearmed.deadline == 25.0


def test_context_arrivals_reject_invalid_identity_and_duplicate_sequence() -> None:
    message = Message(role="user", content=[TextBlock(text="text")])
    with pytest.raises(ValueError, match="seq must be positive"):
        ContextArrival(0, "main", message, "input")
    with pytest.raises(ValueError, match="target_agent_id"):
        ContextArrival(1, "", message, "input")
    with pytest.raises(ValueError, match="source"):
        ContextArrival(1, "main", message, "")

    duplicate = (
        ContextArrival(1, "main", message, "input"),
        ContextArrival(1, "main", message, "peer"),
    )
    with pytest.raises(ValueError, match="sequences must be unique"):
        ordered_messages(duplicate, "main")


async def test_pending_coordinator_journals_each_transition_before_exposing_it() -> None:
    events: list[RuntimeEvent] = []

    async def publish(event: RuntimeEvent) -> AgentEventEnvelope:
        events.append(event)
        return AgentEventEnvelope(
            seq=len(events),
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
    first = await coordinator.admit("first", "main")
    second = await coordinator.admit("second", "child")
    recalled = await coordinator.recall_all()

    assert first.arrival_seq == 1
    assert second.arrival_seq == 2
    assert recalled is not None
    assert recalled.source_ids == (first.id, second.id)
    assert isinstance(events[0], InputBuffered)
    assert isinstance(events[1], InputBuffered)
    assert events[2] == InputRecalled(
        input_ids=(first.id, second.id),
        editor_text="first\n\nsecond",
    )
    assert coordinator.pending_count == 0


async def test_cancelled_recall_finishes_published_transition_before_propagating_cancellation() -> None:
    recall_published = asyncio.Event()
    release_publish = asyncio.Event()
    sequence = 0

    async def publish(event: RuntimeEvent) -> AgentEventEnvelope:
        nonlocal sequence
        sequence += 1
        if isinstance(event, InputRecalled):
            recall_published.set()
            await release_publish.wait()
        return AgentEventEnvelope(
            seq=sequence,
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
    entry = await coordinator.admit("draft", "main")
    recall = asyncio.create_task(coordinator.recall_all())
    await asyncio.wait_for(recall_published.wait(), timeout=1)

    recall.cancel()
    release_publish.set()
    with pytest.raises(asyncio.CancelledError):
        await recall

    assert coordinator.pending_count == 0
    assert coordinator.state.entries[0].id == entry.id
    assert coordinator.state.entries[0].status is PendingInputStatus.RECALLED


@pytest.mark.parametrize(
    ("transition", "expected_status"),
    [
        ("admit", PendingInputStatus.PENDING),
        ("claim", PendingInputStatus.CLAIMED),
        ("deliver", PendingInputStatus.DELIVERED),
    ],
)
async def test_cancelled_pending_transition_never_splits_publication_from_state(
    transition: str,
    expected_status: PendingInputStatus,
) -> None:
    blocked = asyncio.Event()
    release = asyncio.Event()
    block_event_name = ""
    sequence = 0

    async def publish(event: RuntimeEvent) -> AgentEventEnvelope:
        nonlocal sequence
        sequence += 1
        if type(event).__name__ == block_event_name:
            blocked.set()
            await release.wait()
        return AgentEventEnvelope(
            seq=sequence,
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
    batch: ClaimBatch | None = None
    operation: asyncio.Task[object]
    if transition == "admit":
        block_event_name = "InputBuffered"
        operation = asyncio.create_task(coordinator.admit("draft", "main"))
    else:
        await coordinator.admit("draft", "main")
        if transition == "claim":
            block_event_name = "InputClaimed"
            operation = asyncio.create_task(coordinator.claim_oldest())
        else:
            batch = await coordinator.claim_oldest()
            assert batch is not None
            block_event_name = "InputDelivered"
            operation = asyncio.create_task(coordinator.mark_delivered(batch))

    await asyncio.wait_for(blocked.wait(), timeout=1)
    operation.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation

    assert len(coordinator.state.entries) == 1
    assert coordinator.state.entries[0].status is expected_status


async def test_pending_coordinator_empty_claim_and_recall_paths_publish_nothing() -> None:
    events: list[RuntimeEvent] = []

    async def publish(event: RuntimeEvent) -> AgentEventEnvelope:
        events.append(event)
        raise AssertionError("empty transitions must not publish")

    coordinator = PendingInputCoordinator(publish)

    assert await coordinator.recall_all() is None
    assert await coordinator.claim_for_target("main", reason="boundary") is None
    assert await coordinator.claim_oldest() is None
    assert events == []


async def test_pending_coordinator_claims_only_the_requested_target() -> None:
    events: list[RuntimeEvent] = []

    async def publish(event: RuntimeEvent) -> AgentEventEnvelope:
        events.append(event)
        return AgentEventEnvelope(
            seq=len(events),
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
    main = await coordinator.admit("main text", "main")
    child = await coordinator.admit("child text", "child")

    batch = await coordinator.claim_for_target("child", reason="focus-boundary")

    assert batch is not None
    assert tuple(entry.id for entry in batch.entries) == (child.id,)
    assert tuple(entry.id for entry in coordinator.state.pending) == (main.id,)
    assert events[-1] == InputClaimed((child.id,), "child", "focus-boundary")


async def test_pending_coordinator_keeps_claimed_messages_distinct_until_delivery() -> None:
    events: list[RuntimeEvent] = []

    async def publish(event: RuntimeEvent) -> AgentEventEnvelope:
        events.append(event)
        return AgentEventEnvelope(
            seq=len(events),
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
    first = await coordinator.admit("first", "main")
    second = await coordinator.admit("second", "child")

    batch = await coordinator.claim_all_for_interrupt("child")
    assert batch is not None
    assert [entry.text for entry in batch.entries] == ["first", "second"]
    claimed_event = events[-1]
    assert isinstance(claimed_event, InputClaimed)
    assert claimed_event == InputClaimed(
        input_ids=(first.id, second.id),
        target_agent_id="child",
        reason="interrupt",
    )

    await coordinator.mark_delivered(batch)
    delivered_event = events[-1]
    assert isinstance(delivered_event, InputDelivered)
    assert delivered_event == InputDelivered(
        input_ids=(first.id, second.id),
        target_agent_id="child",
    )
    assert [entry.status for entry in coordinator.state.entries] == [
        PendingInputStatus.DELIVERED,
        PendingInputStatus.DELIVERED,
    ]
