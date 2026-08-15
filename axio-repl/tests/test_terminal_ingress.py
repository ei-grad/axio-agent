from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from axio_repl._terminal_ingress import (
    IngressDestination,
    IngressPhase,
    OutputFrame,
    TerminalIngress,
)


def _drain_active(ingress: TerminalIngress) -> str:
    chunks: list[str] = []
    ingress.wake_delivered()
    while (chunk := ingress.next_batch()) is not None:
        chunks.append(chunk)
        ingress.finish_batch()
    return "".join(chunks)


def test_ingress_separates_pre_barrier_late_and_post_close_output() -> None:
    ingress = TerminalIngress()

    accepted = ingress.submit(OutputFrame("before\n"))
    assert accepted.destination is IngressDestination.ACTIVE
    assert accepted.wake_consumer
    assert ingress.seal() is False

    late = ingress.submit(OutputFrame("late\n", "stderr"))
    assert late.destination is IngressDestination.LATE
    assert _drain_active(ingress) == "before\n"
    assert ingress.consumer_should_stop

    late_drain = ingress.close_late()
    assert late_drain.frames == (OutputFrame("late\n", "stderr"),)
    assert ingress.phase is IngressPhase.CLOSED
    assert ingress.submit(OutputFrame("fallback\n")).destination is IngressDestination.FALLBACK


def test_ingress_bounds_active_and_late_output_with_explicit_markers() -> None:
    ingress = TerminalIngress(max_pending_chars=8, max_batch_chars=8, max_late_chars=5)

    ingress.submit(OutputFrame("1234"))
    ingress.submit(OutputFrame("5678"))
    ingress.submit(OutputFrame("drop"))
    assert ingress.pending_char_count == 8
    active = _drain_active(ingress)
    assert active.startswith("12345678")
    assert "terminal output skipped: 1 frame(s), 4 character(s)" in active

    assert ingress.seal()
    ingress.wake_delivered()
    ingress.submit(OutputFrame("12345"))
    ingress.submit(OutputFrame("lost"))
    late = ingress.close_late()
    assert late.frames == (OutputFrame("12345"),)
    assert (late.dropped_frames, late.dropped_chars) == (1, 4)


def test_fail_discards_active_output_but_retains_subsequent_late_writes() -> None:
    ingress = TerminalIngress()
    ingress.submit(OutputFrame("active"))

    ingress.fail()

    assert ingress.phase is IngressPhase.SEALED
    assert ingress.pending_char_count == 0
    assert ingress.submit(OutputFrame("after failure")).destination is IngressDestination.LATE
    assert ingress.close_late().frames == (OutputFrame("after failure"),)


def test_close_late_requires_an_explicit_barrier() -> None:
    with pytest.raises(RuntimeError, match="sealed"):
        TerminalIngress().close_late()


def test_ingress_validates_limits_and_close_operations_are_idempotent() -> None:
    with pytest.raises(ValueError, match="limits must be positive"):
        TerminalIngress(max_pending_chars=0)

    ingress = TerminalIngress()
    empty = ingress.submit(OutputFrame(""))
    assert empty.destination is IngressDestination.ACTIVE
    assert not empty.wake_consumer
    ingress.seal()
    ingress.fail()
    assert ingress.close_late().frames == ()
    assert ingress.close_late().frames == ()
    ingress.fail()
    assert ingress.phase is IngressPhase.CLOSED


def test_concurrent_producers_keep_whole_frames_and_the_budget() -> None:
    ingress = TerminalIngress(max_pending_chars=100_000, max_batch_chars=32)
    frames = [f"frame-{index:04d}\n" for index in range(500)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda content: ingress.submit(OutputFrame(content)), frames))

    rendered = _drain_active(ingress)
    assert ingress.pending_char_count == 0
    assert all(rendered.count(frame) == 1 for frame in frames)
