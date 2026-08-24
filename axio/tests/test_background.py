import asyncio
from collections.abc import AsyncGenerator
from types import MappingProxyType
from typing import Any

import pytest

from axio import background, notify
from axio.agent import Agent
from axio.blocks import ToolUseBlock
from axio.testing import StubTransport
from axio.tool import BACKGROUND_PARAM, CURRENT_TOOL_CALL, Tool, ToolCallContext


@pytest.fixture(autouse=True)
async def clean_registry() -> AsyncGenerator[None, None]:
    yield
    await background.cancel_all()
    notify.set_owner_resolver(None)
    notify.discard(None)


async def slow_echo(text: str, delay: float = 0.0) -> str:
    await asyncio.sleep(delay)
    return f"echo: {text}"


def _agent(tool: Tool[Any]) -> Agent:
    return Agent(system="test", tools=[tool], transport=StubTransport())


def test_every_tool_advertises_the_argument() -> None:
    tool = Tool[Any](name="slow_echo", handler=slow_echo)
    schema = tool.input_schema
    assert BACKGROUND_PARAM in schema["properties"]
    # The tool's own arguments must survive alongside it.
    assert "text" in schema["properties"]


def test_non_detachable_tool_omits_the_argument() -> None:
    tool = Tool[Any](name="slow_echo", handler=slow_echo, detachable=False)
    schema = tool.input_schema
    assert BACKGROUND_PARAM not in schema["properties"]
    assert "text" in schema["properties"]


def test_non_detachable_tool_removes_argument_from_explicit_schema() -> None:
    async def handler(**kwargs: object) -> str:
        return str(kwargs)

    schema = MappingProxyType(
        {
            "type": "object",
            "properties": {BACKGROUND_PARAM: {"type": "boolean"}, "text": {"type": "string"}},
            "required": [BACKGROUND_PARAM, "text"],
        }
    )
    tool: Tool[Any] = Tool(name="explicit", handler=handler, schema=schema, detachable=False)
    advertised = tool.input_schema
    assert BACKGROUND_PARAM not in advertised["properties"]
    assert advertised["required"] == ["text"]


def test_the_argument_is_not_stored_on_the_tool_fields() -> None:
    # It is consumed by the dispatcher; a handler must never receive it.
    tool = Tool[Any](name="slow_echo", handler=slow_echo)
    assert BACKGROUND_PARAM not in tool._fields


@pytest.mark.asyncio
async def test_ordinary_call_is_unaffected() -> None:
    agent = _agent(Tool[Any](name="slow_echo", handler=slow_echo))
    block = ToolUseBlock(id="1", name="slow_echo", input={"text": "hi"})
    [result] = await agent.dispatch_tools([block], 0)
    assert result.content == "echo: hi"
    assert not background.snapshot()


@pytest.mark.asyncio
async def test_detached_call_returns_a_handle_and_finishes_later() -> None:
    agent = _agent(Tool[Any](name="slow_echo", handler=slow_echo))
    block = ToolUseBlock(id="1", name="slow_echo", input={"text": "later", "delay": 0.05, BACKGROUND_PARAM: True})
    [result] = await agent.dispatch_tools([block], 0)
    assert "runs detached as" in str(result.content)

    [call] = background.snapshot()
    assert not call.task.done()
    assert await call.task == "echo: later"
    assert call.state == "done"
    assert call.output() == "echo: later"


@pytest.mark.asyncio
async def test_detached_handler_inherits_tool_call_correlation() -> None:
    seen: list[ToolCallContext] = []

    async def handler() -> str:
        await asyncio.sleep(0)
        seen.append(CURRENT_TOOL_CALL.get())
        return "done"

    agent = _agent(Tool[Any](name="detached", handler=handler))
    outer = ToolCallContext(tool_use_id="outer", tool_name="outer", iteration=0)
    token = CURRENT_TOOL_CALL.set(outer)
    try:
        block = ToolUseBlock(id="bg-call", name="detached", input={BACKGROUND_PARAM: True})
        [result] = await agent.dispatch_tools([block], 5)
        assert CURRENT_TOOL_CALL.get() is outer
    finally:
        CURRENT_TOOL_CALL.reset(token)

    assert not result.is_error
    [call] = background.snapshot()
    assert await call.task == "done"
    assert seen == [ToolCallContext(tool_use_id="bg-call", tool_name="detached", iteration=5)]


@pytest.mark.asyncio
async def test_non_detachable_call_rejects_forced_background() -> None:
    called = False

    async def handler() -> str:
        nonlocal called
        called = True
        return "called"

    agent = _agent(Tool[Any](name="foreground_only", handler=handler, detachable=False))
    block = ToolUseBlock(id="1", name="foreground_only", input={BACKGROUND_PARAM: True})
    [result] = await agent.dispatch_tools([block], 4)

    assert result.is_error
    assert "does not support background execution" in str(result.content)
    assert not called
    assert not background.snapshot()


@pytest.mark.asyncio
async def test_detached_failure_is_kept_not_swallowed() -> None:
    async def boom() -> str:
        raise RuntimeError("nope")

    agent = _agent(Tool[Any](name="boom", handler=boom))
    block = ToolUseBlock(id="1", name="boom", input={BACKGROUND_PARAM: True})
    await agent.dispatch_tools([block], 0)

    [call] = background.snapshot()
    await asyncio.gather(call.task, return_exceptions=True)
    assert call.state == "failed"
    assert "nope" in call.output()


@pytest.mark.asyncio
async def test_streaming_tool_detaches_instead_of_streaming() -> None:
    async def streamer(text: str) -> str:
        return f"done: {text}"

    async def _stream(text: str) -> AsyncGenerator[tuple[str, str], None]:
        yield "stdout", f"chunk:{text}"

    streamer.stream = _stream  # type: ignore[attr-defined]
    tool: Tool[Any] = Tool(name="streamer", handler=streamer)
    assert tool.supports_streaming
    agent = _agent(tool)

    queue: asyncio.Queue[object] = asyncio.Queue()
    block = ToolUseBlock(id="1", name="streamer", input={"text": "x", BACKGROUND_PARAM: True})
    [result] = await agent._dispatch_tools_streaming([block], 0, queue)  # type: ignore[arg-type]
    # Intercepted before the streaming decision, so nothing was streamed.
    assert "runs detached as" in str(result.content)
    assert queue.empty()

    [call] = background.snapshot()
    assert await call.task == "done: x"


@pytest.mark.asyncio
async def test_non_detachable_streaming_tool_rejects_forced_background() -> None:
    called = False

    async def streamer() -> str:
        nonlocal called
        called = True
        return "done"

    async def _stream() -> AsyncGenerator[tuple[str, str], None]:
        nonlocal called
        called = True
        yield "stdout", "chunk"

    streamer.stream = _stream  # type: ignore[attr-defined]
    tool: Tool[Any] = Tool(name="foreground_stream", handler=streamer, detachable=False)
    agent = _agent(tool)

    queue: asyncio.Queue[object] = asyncio.Queue()
    block = ToolUseBlock(id="1", name="foreground_stream", input={BACKGROUND_PARAM: True})
    [result] = await agent._dispatch_tools_streaming([block], 7, queue)  # type: ignore[arg-type]

    assert result.is_error
    assert "does not support background execution" in str(result.content)
    assert not called
    assert queue.empty()
    assert not background.snapshot()


@pytest.mark.asyncio
async def test_the_result_reaches_the_owner_that_started_the_call() -> None:
    notify.set_owner_resolver(lambda: "owner-a")
    try:
        handle = background.start("slow_echo", slow_echo("hi"))
    finally:
        notify.set_owner_resolver(None)

    call = background.get(handle)
    assert call is not None
    assert call.owner == "owner-a"
    await call.task

    [text] = notify.drain("owner-a")
    notify.discard("owner-a")
    assert handle in text
    assert "slow_echo" in text
    assert "done" in text
    assert f'monitor(tasks=["{handle}"])' in text
    assert text.endswith("echo: hi")
    assert not notify.drain("owner-a")


@pytest.mark.asyncio
async def test_a_cancelled_call_notifies_nobody() -> None:
    handle = background.start("slow_echo", slow_echo("never", 30))
    call = background.get(handle)
    assert call is not None
    call.task.cancel()
    await asyncio.gather(call.task, return_exceptions=True)
    await asyncio.sleep(0)
    assert notify.drain(None) == []


@pytest.mark.asyncio
async def test_cancel_all_repeats_cancellation_for_a_handler_stuck_in_cleanup() -> None:
    cleanup_started = asyncio.Event()
    finalized = asyncio.Event()

    async def resists_first_cancellation() -> str:
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            cleanup_started.set()
            try:
                await asyncio.Future[None]()
            finally:
                finalized.set()
        return "unreachable"

    background.start("resistant", resists_first_cancellation())
    await asyncio.sleep(0)
    await background.cancel_all(grace_seconds=0.01)

    assert cleanup_started.is_set()
    assert finalized.is_set()
    assert background.snapshot() == []


@pytest.mark.asyncio
async def test_a_failed_call_reports_its_error() -> None:
    async def boom() -> str:
        raise RuntimeError("nope")

    handle = background.start("boom", boom())
    call = background.get(handle)
    assert call is not None
    await asyncio.gather(call.task, return_exceptions=True)
    await asyncio.sleep(0)

    [text] = notify.drain(None)
    assert "failed" in text
    assert "RuntimeError: nope" in text


@pytest.mark.asyncio
async def test_a_result_collected_by_hand_is_not_delivered_again() -> None:
    handle = background.start("slow_echo", slow_echo("hi"))
    call = background.get(handle)
    assert call is not None
    await call.task

    assert "echo: hi" in background.describe(handle)
    assert notify.drain(None) == []


@pytest.mark.asyncio
async def test_streaming_tool_still_streams_when_not_detached() -> None:
    async def streamer(text: str) -> str:
        return f"done: {text}"

    async def _stream(text: str) -> AsyncGenerator[tuple[str, str], None]:
        yield "stdout", f"chunk:{text}"

    streamer.stream = _stream  # type: ignore[attr-defined]
    agent = _agent(Tool[Any](name="streamer", handler=streamer))

    queue: asyncio.Queue[object] = asyncio.Queue()
    block = ToolUseBlock(id="1", name="streamer", input={"text": "x"})
    [result] = await agent._dispatch_tools_streaming([block], 0, queue)  # type: ignore[arg-type]
    assert "chunk:x" in str(result.content)
    assert not queue.empty()
