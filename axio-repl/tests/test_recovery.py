from __future__ import annotations

from pathlib import Path

import pytest
from axio.blocks import TextBlock
from axio.events import ReasoningDelta, TextDelta, ToolInputDelta, ToolOutputDelta, ToolUseStart
from axio.messages import Message
from axio_tools_agents.runtime import (
    EditorSnapshot,
    InputBuffered,
    InputClaimed,
    InputDelivered,
    InputRecalled,
    RecoveryApplied,
    ShutdownRecorded,
    TurnFinished,
    TurnStarted,
    TurnStatus,
)

from axio_repl._journal import SessionJournal
from axio_repl._recovery import RecoveryError, materialize_recovery


async def _publish_event(
    journal: SessionJournal,
    kind: str,
    event: object,
    seq: int,
    turn_id: str | None = None,
) -> None:
    assert await journal.publish(
        kind,
        {"hub_seq": seq, "run_id": "run", "event": event},
        agent_id="main",
        turn_id=turn_id,
        context_id="context",
        execution_mode="foreground",
    )


async def _publish_message(
    journal: SessionJournal,
    message: Message,
    seq: int,
    turn_id: str,
    *,
    source_input_id: str | None = None,
) -> None:
    assert await journal.publish(
        "message_committed",
        {
            "hub_seq": seq,
            "run_id": "run",
            "message": message,
            "source_input_id": source_input_id,
        },
        agent_id="main",
        turn_id=turn_id,
        context_id="context",
        execution_mode="foreground",
    )


async def test_recovery_restores_messages_pending_editor_and_partial_turn(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    assert await journal.publish(
        "message_committed",
        {
            "hub_seq": 1,
            "run_id": "run",
            "message": Message(role="user", content=[TextBlock(text="committed")]),
        },
        agent_id="main",
        turn_id="complete-turn",
        context_id="context",
        execution_mode="foreground",
    )
    await _publish_event(journal, "input_buffered", InputBuffered("one", "pending", "main"), 2)
    await _publish_event(journal, "input_buffered", InputBuffered("two", "recalled", "main"), 3)
    await _publish_event(journal, "input_recalled", InputRecalled(("two",), "draft editor"), 4)
    await _publish_event(journal, "editor_snapshot", EditorSnapshot("draft editor plus edits"), 5)
    await _publish_event(journal, "turn_started", TurnStarted("work"), 6, "unfinished")
    await _publish_event(journal, "stream_event", ReasoningDelta(0, "checking"), 7, "unfinished")
    await _publish_event(journal, "stream_event", TextDelta(0, "partial answer"), 8, "unfinished")
    await _publish_event(
        journal,
        "stream_event",
        ToolUseStart(index=0, tool_use_id="call", name="shell"),
        9,
        "unfinished",
    )
    await _publish_event(
        journal,
        "stream_event",
        ToolInputDelta(index=0, tool_use_id="call", partial_json='{"command":"sleep'),
        10,
        "unfinished",
    )
    await _publish_event(
        journal,
        "stream_event",
        ToolOutputDelta(tool_use_id="call", name="shell", key="stdout", delta="still running"),
        11,
        "unfinished",
    )
    await journal.close()

    recovered = materialize_recovery(journal.events_path)

    assert recovered.source_session_id == "source"
    assert [pending.text for pending in recovered.pending_inputs] == ["pending"]
    assert recovered.editor_text == "draft editor plus edits"
    assert recovered.recovery_ids == ("source:unfinished",)
    assert [message.role for message in recovered.messages] == ["user", "assistant", "user"]
    assert recovered.messages[0].content == [TextBlock(text="committed")]
    assert recovered.messages[1].content == [TextBlock(text="partial answer")]
    notice = recovered.messages[2].content[0]
    assert isinstance(notice, TextBlock)
    assert "Available reasoning fragment:\nchecking" in notice.text
    assert "name=shell, call_id=call" in notice.text
    assert "still running" in notice.text


async def test_recovery_applied_record_prevents_duplicate_partial_materialization(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    await _publish_event(journal, "turn_started", TurnStarted("work"), 1, "unfinished")
    await _publish_event(journal, "stream_event", TextDelta(0, "partial"), 2, "unfinished")
    await _publish_event(
        journal,
        "recovery_applied",
        RecoveryApplied(source_session_id="source", recovery_ids=("source:unfinished",)),
        3,
    )
    await journal.close()

    recovered = materialize_recovery(journal.events_path)

    assert recovered.messages == ()
    assert recovered.recovery_ids == ()


async def test_recovery_ignores_only_unterminated_tail(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    await journal.close()
    with journal.events_path.open("ab") as output:
        output.write(b'{"torn":')

    recovered = materialize_recovery(journal.events_path)

    assert recovered.discarded_tail_bytes == len(b'{"torn":')


async def test_cancelled_turn_recovers_notice_without_duplicating_committed_partial(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    await _publish_event(journal, "turn_started", TurnStarted("work"), 1, "cancelled")
    await _publish_event(journal, "stream_event", TextDelta(0, "partial"), 2, "cancelled")
    await _publish_event(
        journal,
        "turn_finished",
        TurnFinished(status=TurnStatus.CANCELLED, stop_reason=None, error="turn cancelled"),
        3,
        "cancelled",
    )
    await _publish_message(
        journal,
        Message(role="assistant", content=[TextBlock(text="partial")]),
        4,
        "cancelled",
    )
    await _publish_event(
        journal,
        "shutdown_recorded",
        ShutdownRecorded(
            reason="terminal_failure",
            pending_input_ids=(),
            deferred_tool_use_ids=(),
            interrupted_turn_id="cancelled",
            partial_text="partial",
        ),
        5,
    )
    await journal.close()

    recovered = materialize_recovery(journal.events_path)

    assert [message.role for message in recovered.messages] == ["assistant", "user"]
    assert recovered.messages[0].content == [TextBlock(text="partial")]
    notice = recovered.messages[1].content[0]
    assert isinstance(notice, TextBlock)
    assert "Recorded reason: terminal_failure" in notice.text


async def test_committed_escape_notice_closes_cancelled_turn_for_recovery(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    await _publish_event(journal, "turn_started", TurnStarted("work"), 1, "cancelled")
    await _publish_event(journal, "stream_event", TextDelta(0, "partial"), 2, "cancelled")
    await _publish_event(
        journal,
        "turn_finished",
        TurnFinished(status=TurnStatus.CANCELLED, stop_reason=None, error="turn cancelled"),
        3,
        "cancelled",
    )
    await _publish_message(
        journal,
        Message(role="assistant", content=[TextBlock(text="partial")]),
        4,
        "cancelled",
    )
    await _publish_message(
        journal,
        Message(role="user", content=[TextBlock(text="[Turn cancelled was interrupted by Escape.]")]),
        5,
        "cancelled",
    )
    await journal.close()

    recovered = materialize_recovery(journal.events_path)

    assert [message.role for message in recovered.messages] == ["assistant", "user"]
    assert recovered.recovery_ids == ()


async def test_claimed_input_is_recovered_until_delivery_is_durable(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    await _publish_event(journal, "input_buffered", InputBuffered("one", "pending", "main"), 1)
    await _publish_event(
        journal,
        "input_claimed",
        InputClaimed(input_ids=("one",), target_agent_id="main", reason="normal"),
        2,
    )
    await journal.close()

    recovered = materialize_recovery(journal.events_path)
    assert [entry.text for entry in recovered.pending_inputs] == ["pending"]

    delivered = await SessionJournal.open(session_id="delivered", root=tmp_path)
    await _publish_event(delivered, "input_buffered", InputBuffered("one", "pending", "main"), 1)
    await _publish_event(
        delivered,
        "input_claimed",
        InputClaimed(input_ids=("one",), target_agent_id="main", reason="normal"),
        2,
    )
    await _publish_event(
        delivered,
        "input_delivered",
        InputDelivered(input_ids=("one",), target_agent_id="main"),
        3,
    )
    await delivered.close()

    assert materialize_recovery(delivered.events_path).pending_inputs == ()


async def test_committed_claimed_input_is_not_requeued_after_crash_before_delivery_event(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    await _publish_event(journal, "input_buffered", InputBuffered("one", "pending", "main"), 1)
    await _publish_event(
        journal,
        "input_claimed",
        InputClaimed(input_ids=("one",), target_agent_id="main", reason="boundary"),
        2,
    )
    await _publish_message(
        journal,
        Message(role="user", content=[TextBlock(text="pending")]),
        3,
        "turn-1",
        source_input_id="one",
    )
    await journal.close()

    recovered = materialize_recovery(journal.events_path)

    assert recovered.pending_inputs == ()
    assert recovered.messages == (Message(role="user", content=[TextBlock(text="pending")]),)


async def test_unrelated_equal_message_does_not_confirm_claimed_input_delivery(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    await _publish_event(journal, "input_buffered", InputBuffered("one", "same", "main"), 1)
    await _publish_event(
        journal,
        "input_claimed",
        InputClaimed(input_ids=("one",), target_agent_id="main", reason="boundary"),
        2,
    )
    await _publish_message(
        journal,
        Message(role="user", content=[TextBlock(text="same")]),
        3,
        "peer-turn",
    )
    await journal.close()

    recovered = materialize_recovery(journal.events_path)

    assert recovered.messages == (Message(role="user", content=[TextBlock(text="same")]),)
    assert len(recovered.pending_inputs) == 1
    assert recovered.pending_inputs[0].source_id == "one"
    assert recovered.pending_inputs[0].text == "same"

    delivered = await SessionJournal.open(session_id="delivered", root=tmp_path)
    await _publish_event(delivered, "input_buffered", InputBuffered("one", "same", "main"), 1)
    await _publish_event(
        delivered,
        "input_claimed",
        InputClaimed(input_ids=("one",), target_agent_id="main", reason="boundary"),
        2,
    )
    equal_message = Message(role="user", content=[TextBlock(text="same")])
    await _publish_message(delivered, equal_message, 3, "peer-turn")
    await _publish_message(delivered, equal_message, 4, "input-turn", source_input_id="one")
    await delivered.close()

    delivered_recovery = materialize_recovery(delivered.events_path)
    assert delivered_recovery.pending_inputs == ()
    assert delivered_recovery.messages == (equal_message, equal_message)


async def test_correlated_commit_rejects_content_mismatch_instead_of_losing_input(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    await _publish_event(journal, "input_buffered", InputBuffered("one", "expected", "main"), 1)
    await _publish_event(
        journal,
        "input_claimed",
        InputClaimed(input_ids=("one",), target_agent_id="main", reason="boundary"),
        2,
    )
    await _publish_message(
        journal,
        Message(role="user", content=[TextBlock(text="different")]),
        3,
        "turn-1",
        source_input_id="one",
    )
    await journal.close()

    with pytest.raises(RecoveryError, match="content does not match"):
        materialize_recovery(journal.events_path)


async def test_delivery_event_after_committed_claim_confirmation_is_idempotent(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    await _publish_event(journal, "input_buffered", InputBuffered("one", "pending", "main"), 1)
    await _publish_event(
        journal,
        "input_claimed",
        InputClaimed(input_ids=("one",), target_agent_id="main", reason="boundary"),
        2,
    )
    await _publish_message(
        journal,
        Message(role="user", content=[TextBlock(text="pending")]),
        3,
        "turn-1",
        source_input_id="one",
    )
    await _publish_event(
        journal,
        "input_delivered",
        InputDelivered(input_ids=("one",), target_agent_id="main"),
        4,
    )
    await journal.close()

    assert materialize_recovery(journal.events_path).pending_inputs == ()


async def test_recovery_materializes_unfinished_child_turn_for_main_context(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    assert await journal.publish(
        "turn_started",
        {"hub_seq": 1, "run_id": "run", "event": TurnStarted("child work")},
        agent_id="child",
        turn_id="child-turn",
        context_id="child-context",
        execution_mode="background",
    )
    assert await journal.publish(
        "stream_event",
        {"hub_seq": 2, "run_id": "run", "event": TextDelta(0, "child partial")},
        agent_id="child",
        turn_id="child-turn",
        context_id="child-context",
        execution_mode="background",
    )
    await journal.close()

    recovered = materialize_recovery(journal.events_path)

    assert recovered.recovery_ids == ("source:agent:child:child-turn",)
    assert len(recovered.messages) == 1
    notice = recovered.messages[0].content[0]
    assert isinstance(notice, TextBlock)
    assert "original_agent=child" in notice.text
    assert "child partial" in notice.text


async def test_recovery_rejects_unknown_input_transition(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    await _publish_event(
        journal,
        "input_delivered",
        InputDelivered(input_ids=("missing",), target_agent_id="main"),
        1,
    )
    await journal.close()

    with pytest.raises(RecoveryError, match="got unknown"):
        materialize_recovery(journal.events_path)


async def test_recovery_materializes_shutdown_cancelled_deferred_tool_as_user_notice(tmp_path: Path) -> None:
    journal = await SessionJournal.open(session_id="source", root=tmp_path)
    await _publish_event(
        journal,
        "shutdown_recorded",
        ShutdownRecorded(
            reason="sigterm",
            pending_input_ids=(),
            deferred_tool_use_ids=("call-1",),
            deferred_tool_agent_ids=("child-1",),
            deferred_tool_turn_ids=("turn-1",),
            deferred_tool_phases=("protocol_closed",),
        ),
        1,
    )
    await journal.close()

    recovered = materialize_recovery(journal.events_path)

    assert recovered.recovery_ids == ("source:deferred:child-1:turn-1:call-1",)
    notice = recovered.messages[0].content[0]
    assert isinstance(notice, TextBlock)
    assert "call_id=call-1" in notice.text
    assert "original_agent=child-1" in notice.text
    assert "last_phase=protocol_closed" in notice.text


async def test_recovery_chain_copies_materialized_partial_exactly_once(tmp_path: Path) -> None:
    source = await SessionJournal.open(session_id="source", root=tmp_path / "source")
    await _publish_event(source, "turn_started", TurnStarted("work"), 1, "unfinished")
    await _publish_event(source, "stream_event", TextDelta(0, "partial"), 2, "unfinished")
    await source.close()
    first = materialize_recovery(source.events_path)

    resumed = await SessionJournal.open(session_id="resumed", root=tmp_path / "resumed")
    for seq, message in enumerate(first.messages, start=1):
        await _publish_message(resumed, message, seq, "recovery-copy")
    await _publish_event(
        resumed,
        "recovery_applied",
        RecoveryApplied(source_session_id="source", recovery_ids=first.recovery_ids),
        len(first.messages) + 1,
    )
    await resumed.close()

    second = materialize_recovery(resumed.events_path)

    assert second.messages == first.messages
    assert second.recovery_ids == ()
