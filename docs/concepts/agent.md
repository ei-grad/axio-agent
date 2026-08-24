# Agent & the Agentic Loop

The {class}`Agent` is the central orchestrator. It connects a transport, a set of
tools, and a context store into a single loop that streams LLM responses and
dispatches tool calls until the model signals it is done.

## The Agent dataclass

<!--
name: test_agent_dataclass
-->
```python
from dataclasses import dataclass, field
from axio import PatchLineFraming, Tool, CompletionTransport, ToolSelector
from axio.agent import DeferredToolSink
from axio.messages import Message


@dataclass(slots=True)
class Agent:
    system: str
    transport: CompletionTransport
    tools: list[Tool] = field(default_factory=list)
    selector: ToolSelector | None = field(default=None)
    max_iterations: int = field(default=50)
    last_iteration_message: Message | None = field(default=None)
    deferred_tool_sink: DeferredToolSink | None = field(default=None)
    patch_line_framing: PatchLineFraming = field(default="auto")
```

`system`
: The system prompt sent with every request.

`transport`
: Any object satisfying the {ref}`CompletionTransport <protocols>` protocol.

`tools`
: Available tools. The agent searches this list by name when the model
  issues a tool call. Defaults to an empty list.

`selector`
: An optional {ref}`ToolSelector <tool-selector>` that filters the active tool
  list before each iteration. When `None`, all tools are passed to the
  transport on every iteration.

`max_iterations`
: Safety limit preventing runaway loops. The agent emits a
  `SessionEndEvent` with an error if this limit is reached. Defaults to 50.

`last_iteration_message`
: An optional `Message` appended to the effective history **only** on the
  final iteration (when `max_iterations` is about to be exceeded). Useful for
  injecting a stop instruction such as "you must answer now without calling
  more tools" to coerce a final response before the loop terminates.

`deferred_tool_sink`
: An advanced host hook for transferring ownership of an in-flight tool batch
  when its originating turn is cancelled. The default is `None`, which keeps
  ordinary cancellation semantics. REPL-like hosts can use the hook to close
  the original tool protocol with a placeholder and deliver the eventual
  result later through a new user message.

`patch_line_framing`
: Controls the legacy visible-line protocol used to protect leading whitespace
  in `patch_file` content. `auto` (the default) advertises that protocol only
  when the active transport does not provide a general `tool_argument_codec`;
  `on` forces it and `off` suppresses it. The setting is evaluated per agent
  immediately before each provider request and does not mutate shared tools.

## How the loop works

```{mermaid}
flowchart TD
    A[User message] --> B[Append to context]
    B --> C[Get history from context]
    C --> D[Stream from transport]
    D --> E{Tool calls?}
    E -- Yes --> F[Dispatch tools concurrently]
    F --> G[Append results to context]
    G --> C
    E -- No --> H{Stop reason?}
    H -- end_turn --> I[SessionEndEvent]
    H -- max_tokens / error --> J[Error event]
```

1. The user message is appended to the context store.
2. The agent retrieves the full conversation history and streams it to the
   transport along with the tool definitions and system prompt.
3. As `StreamEvent` values arrive, the agent accumulates text deltas and
   buffers pending tool calls.
4. When the transport yields an `IterationEnd` event:
   - If tool-use blocks were collected, the agent dispatches **all tool calls
     concurrently** via `asyncio.gather`, appends the assistant message and
     tool results to context, and loops back to step 2.
   - If only text was produced and the stop reason is `end_turn`, the agent
     emits a `SessionEndEvent` and returns.
5. If `max_iterations` is exceeded, the loop terminates with an error.

## Streaming API

`Agent` exposes four run methods:

`run_stream(user_message, context) -> AgentStream`
: Returns an `AgentStream` - an async iterator over `StreamEvent` values.
  Use this when you need per-token streaming or want to observe tool calls
  as they happen.

`run(user_message, context) -> str`
: Convenience wrapper that consumes the stream and returns the final text.

`run_stream_messages(messages, context, *, on_input_committed=None) -> AgentStream`
: Appends an ordered batch of distinct `Message` objects, then starts one model
  operation. Messages are never joined. Use this when several chronological
  inputs must become visible at the same provider boundary. The input sequence
  is deep-copied before the asynchronous run starts.

`run_messages(messages, context, *, on_input_committed=None) -> str`
: Convenience wrapper for `run_stream_messages()`.

Both batch methods call `ContextStore.append_many()` once. Persistent stores
should implement that operation atomically. If `on_input_committed` is supplied,
the agent awaits it immediately after `append_many()` succeeds and before the
first transport request; hosts use this barrier to mark queued input as
delivered without guessing from equal message text.

## Concurrent tool dispatch

When the model requests multiple tool calls in a single response, the agent
runs them all concurrently via `asyncio.gather`. The public method signature is:

```python
from axio.blocks import ToolResultBlock, ToolUseBlock


async def dispatch_tools(
    self,
    blocks: list[ToolUseBlock],
    iteration: int,
) -> list[ToolResultBlock]: ...
```

Each tool call goes through the full guard chain before execution. If a tool
raises an exception, the agent catches it and wraps it in a `ToolResultBlock`
with `is_error=True` - the model sees the error and can react accordingly.

If a tool's JSON arguments could not be parsed from the stream, the agent
returns a `ToolResultBlock` with `is_error=True` and a message asking the
model to retry with valid JSON, rather than passing malformed input to the
handler.

### Cancellation and deferred tool dispatch

Tool execution starts before the assistant tool-use message is appended to
context. This prevents cancellation from leaving an orphaned `ToolUseBlock`
without a matching result.

With the default `deferred_tool_sink=None`, cancelling a turn also cancels an
unfinished tool batch. The agent appends the assistant tool-use message and one
interrupted `ToolResultBlock` per call before propagating cancellation. Already
completed calls keep their real results.

When a host supplies a `DeferredToolSink`, it is notified when a dispatch starts
and may request deferral for a particular cancellation. The agent then:

1. transfers the still-running task to the sink;
2. appends a protocol-safe placeholder result stating that the tool continues;
3. notifies the sink only after that original protocol has been closed in
   context; and
4. propagates cancellation of the originating turn.

The sink owns eventual result delivery. It must not append another
`ToolResultBlock` for the closed call; a REPL host delivers the result as a new,
labelled user message instead.

## ToolSelector

(tool-selector)=

The `ToolSelector` protocol lets you trim the active tool list before each
iteration. This is useful for reducing noise in the model's context, enforcing
capability restrictions, or implementing dynamic tool routing.

```python
from typing import Protocol, runtime_checkable
from collections.abc import Iterable
from axio.messages import Message
from axio import Tool


@runtime_checkable
class ToolSelector(Protocol):
    async def select(
        self, messages: Iterable[Message], tools: Iterable[Tool]
    ) -> Iterable[Tool]: ...
```

Pass a `ToolSelector` via the `selector` field when constructing an `Agent`.
On each iteration the agent calls `selector.select(history, tools)` and passes
only the returned subset of tools to the transport.

When `selector` is `None` (the default) all tools are passed on every
iteration.

## Copying an Agent

`Agent.copy(**overrides)` returns a new `Agent` with selected fields replaced.
Because `Agent` uses `slots=True`, this is the correct way to derive a
modified agent without mutating the original:

<!-- name: test_agent_copy -->
```python
import asyncio
from axio import Agent
from axio.testing import StubTransport, make_text_response

transport = StubTransport([make_text_response("ok")])
agent = Agent(system="You are helpful.", transport=transport)

# Derive an agent with a different system prompt
strict_agent = agent.copy(system="Be concise. Answer in one sentence.")
assert strict_agent.system == "Be concise. Answer in one sentence."
assert strict_agent.transport is agent.transport  # shared by default
```
