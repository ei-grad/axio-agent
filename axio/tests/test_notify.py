"""The notification bus: addressing, delivery paths, and the turn boundary."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import pytest

from axio import notify
from axio.agent import Agent
from axio.blocks import TextBlock
from axio.context import MemoryContextStore
from axio.events import StreamEvent, ToolResult
from axio.messages import Message
from axio.testing import StubTransport, make_ephemeral_context, make_text_response, make_tool_use_response
from axio.tool import Tool

OWNERS = (None, "owner-a", "owner-b", "agent-1")


@pytest.fixture(autouse=True)
async def clean_bus() -> AsyncGenerator[None, None]:
    for owner in OWNERS:
        notify.discard(owner)
    yield
    notify.set_owner_resolver(None)
    for owner in OWNERS:
        notify.discard(owner)


def _user_texts(history: list[Message]) -> list[str]:
    return [b.text for m in history if m.role == "user" for b in m.content if isinstance(b, TextBlock)]


def test_each_owner_only_sees_its_own_notifications() -> None:
    notify.post("for a", "owner-a")
    notify.post("for b", "owner-b")
    notify.post("for nobody in particular", None)

    assert notify.drain("owner-a") == ["for a"]
    assert notify.drain("owner-b") == ["for b"]
    assert notify.drain(None) == ["for nobody in particular"]
    assert notify.drain("owner-a") == []


def test_a_long_notification_is_cut_and_says_so() -> None:
    notify.post("x" * (notify.NOTIFY_MAX_CHARS * 2), "owner-a")

    [text] = notify.drain("owner-a")
    assert len(text) == notify.NOTIFY_MAX_CHARS
    assert text.endswith("[truncated]")
    assert text.startswith("x" * 100)


def test_a_notification_that_fits_is_left_alone() -> None:
    notify.post("y" * notify.NOTIFY_MAX_CHARS, "owner-a")

    [text] = notify.drain("owner-a")
    assert text == "y" * notify.NOTIFY_MAX_CHARS


def test_an_idle_owner_is_handed_the_text_straight_away() -> None:
    received: list[str] = []
    notify.add_listener("owner-a", received.append)

    notify.post("wake up", "owner-a")

    assert received == ["wake up"]
    assert notify.drain("owner-a") == []


def test_a_working_owner_queues_even_with_a_listener() -> None:
    received: list[str] = []
    notify.add_listener("owner-a", received.append)

    with notify.turn_scope("owner-a"):
        notify.post("mid-turn", "owner-a")
        assert received == []
        assert notify.drain("owner-a") == ["mid-turn"]

    assert received == []


def test_what_a_turn_left_behind_is_flushed_to_the_listener_once() -> None:
    received: list[str] = []
    notify.add_listener("owner-a", received.append)

    with notify.turn_scope("owner-a"):
        notify.post("first", "owner-a")
        notify.post("second", "owner-a")
        assert received == []

    assert received == ["first", "second"]
    assert notify.drain("owner-a") == []


def test_nested_turn_scopes_flush_only_when_the_outermost_ends() -> None:
    received: list[str] = []
    notify.add_listener("owner-a", received.append)

    with notify.turn_scope("owner-a"):
        with notify.turn_scope("owner-a"):
            notify.post("inner", "owner-a")
        assert received == []

    assert received == ["inner"]


def test_a_turn_without_a_listener_keeps_its_queue() -> None:
    with notify.turn_scope("owner-a"):
        notify.post("unread", "owner-a")

    assert notify.drain("owner-a") == ["unread"]


def test_a_retracted_notification_is_never_delivered() -> None:
    notify.post("keep", "owner-a", tag="k")
    notify.post("drop", "owner-a", tag="d")

    notify.retract("owner-a", "d")
    notify.retract("owner-a", "unknown-tag")
    notify.retract("owner-b", "d")

    assert notify.drain("owner-a") == ["keep"]


def test_retracting_a_delivered_notification_does_nothing() -> None:
    received: list[str] = []
    notify.add_listener("owner-a", received.append)

    notify.post("gone already", "owner-a", tag="t")
    notify.retract("owner-a", "t")

    assert received == ["gone already"]


def test_discard_forgets_the_owner_entirely() -> None:
    received: list[str] = []
    notify.add_listener("owner-a", received.append)
    with notify.turn_scope("owner-a"):
        notify.post("stale", "owner-a")
        notify.discard("owner-a")

    assert received == []
    assert notify.drain("owner-a") == []

    notify.post("after death", "owner-a")
    assert received == []
    assert notify.drain("owner-a") == ["after death"]


def test_the_owner_comes_from_the_resolver() -> None:
    assert notify.current_owner() is None
    notify.set_owner_resolver(lambda: "agent-1")
    assert notify.current_owner() == "agent-1"
    notify.set_owner_resolver(None)
    assert notify.current_owner() is None


async def test_a_notification_posted_mid_turn_joins_the_next_iteration() -> None:
    notify.set_owner_resolver(lambda: "agent-1")

    async def ping(msg: str) -> str:
        notify.post(f"news about {msg}", notify.current_owner())
        return "pong"

    transport = StubTransport(
        [
            make_tool_use_response("ping", tool_input={"msg": "hi"}),
            make_text_response("done"),
        ]
    )
    agent = Agent(system="t", tools=[Tool[Any](name="ping", handler=ping)], transport=transport)
    context: MemoryContextStore = make_ephemeral_context()

    assert await agent.run("go", context) == "done"

    history = await context.get_history()
    assert "news about hi" in _user_texts(history)
    assert notify.drain("agent-1") == []


class _PostingStub(StubTransport):
    """Stub transport that posts a notification while the answer is produced."""

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        notify.post("late news", notify.current_owner())
        return super().stream(messages, tools, system)


async def test_a_turn_that_ends_hands_its_leftovers_to_the_listener() -> None:
    notify.set_owner_resolver(lambda: "agent-1")
    received: list[str] = []
    notify.add_listener("agent-1", received.append)

    agent = Agent(system="t", transport=_PostingStub([make_text_response("done")]))
    context: MemoryContextStore = make_ephemeral_context()

    assert await agent.run("go", context) == "done"

    # Nothing drained it — the turn was already producing its last answer.
    assert received == ["late news"]
    assert notify.drain("agent-1") == []


async def test_an_abandoned_turn_still_flushes_its_leftovers() -> None:
    notify.set_owner_resolver(lambda: "agent-1")
    received: list[str] = []
    notify.add_listener("agent-1", received.append)

    async def ping(msg: str) -> str:
        notify.post(f"news about {msg}", notify.current_owner())
        return "pong"

    transport = StubTransport(
        [
            make_tool_use_response("ping", tool_input={"msg": "hi"}),
            make_text_response("done"),
        ]
    )
    agent = Agent(system="t", tools=[Tool[Any](name="ping", handler=ping)], transport=transport)
    context: MemoryContextStore = make_ephemeral_context()

    stream = agent.run_stream("go", context)
    async for event in stream:
        if isinstance(event, ToolResult):
            break
    # Closing the stream mid-turn must still end the scope and flush.
    await stream.aclose()

    assert received == ["news about hi"]
    assert notify.drain("agent-1") == []
