from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from axio.events import IterationEnd, ReasoningDelta, StreamEvent, ToolInputDelta, ToolUseStart
from axio.messages import Message
from axio.models import Capability, ModelRegistry, ModelSpec
from axio.tool import Tool
from axio.types import StopReason, Usage

import axio_repl

STARTED = Path(os.environ["AXIO_EXIT_HARNESS_STARTED"])
FINALIZED = Path(os.environ["AXIO_EXIT_HARNESS_FINALIZED"])
JOURNAL_ROOT = Path(os.environ["AXIO_EXIT_HARNESS_JOURNAL_ROOT"])
CONFIG_ROOT = Path(os.environ["AXIO_EXIT_HARNESS_CONFIG_ROOT"])
PEER_ROOT = Path(os.environ["AXIO_EXIT_HARNESS_PEER_ROOT"])
PROMPT_COUNT = Path(os.environ["AXIO_EXIT_HARNESS_PROMPT_COUNT"])
CALL_COUNT = Path(os.environ["AXIO_EXIT_HARNESS_CALL_COUNT"])
MODE = os.environ["AXIO_EXIT_HARNESS_MODE"]


def record_transport_call() -> int:
    try:
        count = int(CALL_COUNT.read_text())
    except FileNotFoundError:
        count = 0
    count += 1
    CALL_COUNT.write_text(str(count))
    return count


class ExitTransport:
    name = "never-ending"

    def __init__(self, **kwargs: object) -> None:
        del kwargs
        capabilities = {Capability.text, Capability.tool_use} if MODE in {"tool", "detached"} else {Capability.text}
        self.model = ModelSpec(id="stub/exit", capabilities=frozenset(capabilities))
        self.models = ModelRegistry([self.model])
        self.temperature: float | None = None
        self.max_output_tokens: int | None = None
        self.debug = False

    async def fetch_models(self) -> None:
        pass

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[object]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools, system
        call = record_transport_call()
        if MODE != "detached" and call > 1:
            raise AssertionError("EOF shutdown started another provider iteration")
        if MODE in {"tool", "detached"} and call == 1:
            yield ToolUseStart(index=0, tool_use_id="blocking-call", name="blocking_tool")
            partial_json = '{"background":true}' if MODE == "detached" else "{}"
            yield ToolInputDelta(index=0, tool_use_id="blocking-call", partial_json=partial_json)
            yield IterationEnd(
                iteration=1,
                stop_reason=StopReason.tool_use,
                usage=Usage(input_tokens=1, output_tokens=1),
            )
            return
        if MODE == "detached":
            if call > 2:
                raise AssertionError("EOF shutdown started another provider iteration")
            await asyncio.Future[None]()
            return
        STARTED.touch()
        try:
            yield ReasoningDelta(index=0, delta="provider stream is active")
            await asyncio.Future()
        finally:
            try:
                if MODE == "provider_resistant":
                    await asyncio.Future[None]()
            finally:
                FINALIZED.touch()


async def blocking_tool() -> str:
    STARTED.touch()
    try:
        await asyncio.Future[None]()
        return "unreachable"
    finally:
        FINALIZED.touch()


async def build_tools(*args: object, **kwargs: object) -> tuple[list[Tool[object]], str, Path, str]:
    del args, kwargs
    tools: list[Tool[object]] = []
    if MODE in {"tool", "detached"}:
        tools.append(Tool(name="blocking_tool", handler=blocking_tool))
    return tools, "none", Path.cwd(), ""


def select_transport(name: str | None, credential_override: bool = False) -> tuple[Callable[..., Any], str]:
    del name, credential_override
    return ExitTransport, ""


def effective_username() -> str:
    return "exit-test"


def main() -> None:
    os.environ["AXIO_PEER_DIR"] = str(PEER_ROOT)
    axio_repl._select_transport = select_transport
    axio_repl._sandbox.build_tools = build_tools
    setattr(axio_repl, "resolve_effective_username", effective_username)
    original_make_session = axio_repl._panel.make_session

    def make_session(*args: Any, **kwargs: Any) -> Any:
        session = original_make_session(*args, **kwargs)
        original_prompt_async = session.prompt_async

        async def prompt_async(*prompt_args: Any, **prompt_kwargs: Any) -> Any:
            try:
                count = int(PROMPT_COUNT.read_text())
            except FileNotFoundError:
                count = 0
            PROMPT_COUNT.write_text(str(count + 1))
            return await original_prompt_async(*prompt_args, **prompt_kwargs)

        session.prompt_async = prompt_async
        return session

    axio_repl._panel.make_session = make_session
    sys.argv = [
        "axio-repl",
        "--transport",
        "exit",
        "--sandbox",
        "none",
        "--no-powerline",
        "--config-dir",
        str(CONFIG_ROOT),
        "--session-log-dir",
        str(JOURNAL_ROOT),
    ]
    axio_repl.main_sync()


if __name__ == "__main__":
    main()
