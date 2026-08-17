# AGENTS.md

Guidance for AI coding agents working in this repository.

---

## Repository layout

This is a **uv workspace** monorepo. Each subdirectory is an independent Python package with its own `pyproject.toml`, `src/` layout, and `tests/`. They share a single `uv.lock` and a single `.venv`.

- `axio/` - core library; start here when unsure
- `axio-repl/` - POSIX inline REPL, chronological input coordinator, journal recovery
- `axio-tui/` - TUI application and plugin discovery
- `axio-transport-*/` - transport implementations
- `axio-tools-*/` - tool providers
- `axio-context-sqlite/` - SQLite context store
- `axio-tui-guards/` - permission guard plugins
- `docs/` - Sphinx sources and markdown-pytest doc tests
- `examples/` - runnable example scripts

| Package | Purpose |
|---|---|
| `axio` | Agent loop, Tool, Transport protocol, ContextStore, PermissionGuard, events, blocks, types |
| `axio-transport-openai` | OpenAI-compatible transport (OpenAI, Nebius, OpenRouter, custom) |
| `axio-transport-anthropic` | Anthropic Claude transport with prompt caching |
| `axio-transport-codex` | ChatGPT via OAuth Responses API |
| `axio-repl` | POSIX primary-buffer REPL, renderer, input coordination, session journal and recovery |
| `axio-tools-agents` | Local agent lifecycle, messaging, monitoring, and ordered runtime events |
| `axio-tools-local` | File, shell, Python execution tools |
| `axio-tools-mcp` | MCP server bridge |
| `axio-tools-docker` | Docker sandbox tool provider |
| `axio-context-sqlite` | SQLite-backed persistent context store |
| `axio-tui` | Textual TUI, SQLite context store, plugin discovery |
| `axio-tui-guards` | PathGuard + LLMGuard plugins |

---

## Commands

Always use `make`. Never invoke `uv run pytest`, `ruff`, or `mypy` directly at the repo root.

```bash
make              # lint + type-check + tests for all packages + doc tests
make linter       # ruff check + ruff format --check on all packages
make typing       # mypy --strict on all packages
make pytest       # pytest on all packages
make test-docs    # markdown-pytest on docs/
```

Run a single package's tests or checks:

```bash
make PACKAGES=axio-transport-anthropic
```

Run a specific test file inside a package:

```bash
uv run --directory axio pytest tests/test_agent_run.py -v
```

Run doc tests for a single file:

```bash
uv run --directory docs pytest -v guides/best-practices.md
```

---

## Development setup

```bash
git clone https://github.com/mosquito/axio-agent.git
cd axio-agent
uv sync --all-packages   # installs all workspace members + dev deps into .venv
```

After sync, all local packages resolve to their workspace sources via `[tool.uv.sources]` - no `pip install -e` or PYTHONPATH hacks needed.

---

## Code style

- **Formatter / linter**: [ruff](https://docs.astral.sh/ruff/), `line-length = 119`, `target-version = "py312"`
- **Type checker**: [mypy](https://mypy.readthedocs.io/) strict mode (`--strict`), `python_version = "3.12"`
- Enabled ruff rules: `E`, `F`, `I`, `UP`
- All new code must pass `mypy --strict` with zero errors
- Use `from __future__ import annotations` at the top of every module
- Prefer `dataclass(frozen=True, slots=True)` for value types

Always run `make linter` and `make typing` before considering a task done.

---

## Architecture

### Public API (`axio/__init__.py`)

Common symbols are importable directly from `axio`:

```python
from axio import Agent, Tool, Field, PermissionGuard, IterationEnd
from axio import StopReason, Usage, GuardError, GuardCrash, HandlerError, HandlerCrash
from axio import ContextStore, MemoryContextStore, CompletionTransport
```

Submodule-only (not re-exported at top level):

```python
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.schema import build_tool_schema
from axio.agent_loader import AgentSpec, load_agents
from axio.compaction import AutoCompactStore
```

### Types (`axio/types.py`)

Primitive type aliases and enums:

```python
type ToolName = str          # Unique tool identifier
type ToolCallID = str         # Opaque ID for a tool invocation

class StopReason(StrEnum):
    end_turn = "end_turn"     # Assistant finished responding
    tool_use = "tool_use"     # Assistant wants to call a tool
    max_tokens = "max_tokens" # Output truncated
    error = "error"           # Something went wrong

@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    # Supports arithmetic: usage1 + usage2
```

### Messages (`axio/messages.py`)

The fundamental unit of conversation history:

```python
@dataclass(slots=True)
class Message:
    role: Literal["user", "assistant", "system"]
    content: list[ContentBlock]  # List of Text/Image/ToolUse/ToolResult blocks

    # Serialization
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message: ...
```

Messages are serialized for transport to LLM providers and for persistence in context stores.

### Content Blocks (`axio/blocks.py`)

Four block types represent all content in messages:

| Block | Fields | Purpose |
|---|---|---|
| `TextBlock` | `text: str` | Plain text content |
| `ImageBlock` | `media_type: Literal[image/*]`, `data: bytes` | Image attachments (base64 encoded in serialization) |
| `ToolUseBlock` | `id: ToolCallID`, `name: ToolName`, `input: dict[str, Any]` | A tool call request |
| `ToolResultBlock` | `tool_use_id: ToolCallID`, `content: str | list[Block]`, `is_error: bool` | Result of tool execution |

```python
# Serialization helpers
def to_dict(block: ContentBlock) -> dict[str, Any]: ...
def from_dict(data: dict[str, Any]) -> ContentBlock: ...
```

Both functions handle the full round-trip including nested blocks in `ToolResultBlock`.

### Events (`axio/events.py`)

All events are `dataclass(frozen=True, slots=True)`. `StreamEvent` is a type union of all event types:

| Event | Fields | When |
|---|---|---|
| `TextDelta(index, delta)` | `index: int`, `delta: str` | Streamed text chunk |
| `ReasoningDelta(index, delta)` | `index: int`, `delta: str` | Streamed reasoning/thinking chunk |
| `ToolUseStart(index, tool_use_id, name)` | `index: int`, `tool_use_id: ToolCallID`, `name: ToolName` | Tool call begins |
| `ToolInputDelta(index, tool_use_id, partial_json)` | `index: int`, `tool_use_id: ToolCallID`, `partial_json: str` | Streaming tool arguments (JSON string) |
| `ToolResult(tool_use_id, name, is_error, content, input)` | Various | Tool execution result (emitted by agent, not transport) |
| `IterationEnd(iteration, stop_reason, usage)` | `iteration: int`, `stop_reason: StopReason`, `usage: Usage` | One LLM call complete |
| `SessionEndEvent(stop_reason, total_usage)` | `stop_reason: StopReason`, `total_usage: Usage` | Full agent run complete |
| `Error(exception)` | `exception: Exception` | Unhandled exception in the stream |

**Important**: The stream **must** end with exactly one `IterationEnd` event per LLM call.

### Transport protocol (`axio/transport.py`)

`CompletionTransport` is a `@runtime_checkable` Protocol. Implement one method:

```python
@runtime_checkable
class CompletionTransport(Protocol):
    def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
    ) -> AsyncIterator[StreamEvent]: ...
```

Contract:
- The stream **must** end with exactly one `IterationEnd` event
- Do not suppress exceptions - let them propagate as `Error` events or raise naturally
- `messages` contains the full conversation history including the current user input or ordered input batch, which the `Agent` appends before calling the transport
- `tools` is the Tool definitions from the Agent's registry

### Agent loop (`axio/agent.py`)

`Agent` is a `dataclass(slots=True)` with fields:

```python
@dataclass(slots=True)
class Agent:
    system: str                              # System prompt
    transport: CompletionTransport          # LLM backend
    tools: list[Tool]                        # Available tools
    selector: ToolSelector | None = None     # Optional tool selection logic
    max_iterations: int = 50                # Iteration limit
    last_iteration_message: Message | None = None  # Optional final-iteration hint
    deferred_tool_sink: DeferredToolSink | None = None  # Optional host-owned continuation of cancelled tools
```

Public API:
- `agent.run_stream(user_message, context) -> AgentStream` - streaming entry point, returns an async iterator that yields `StreamEvent` plus `SessionEndEvent`
- `await agent.run(user_message, context) -> str` - convenience wrapper, returns final text
- `agent.run_stream_messages(messages, context, *, on_input_committed=None) -> AgentStream` - append an ordered batch of distinct `Message` objects and start one model operation
- `await agent.run_messages(messages, context, *, on_input_committed=None) -> str` - convenience wrapper for the batch API

The batch APIs deep-copy the input sequence and call `ContextStore.append_many()` once. Persistent stores should make
that operation atomic. `on_input_committed`, when supplied, runs after the append succeeds and before the first
transport request; use it to correlate durable queue transitions with the exact input IDs rather than matching text.

**Loop per iteration**:
1. Call `transport.stream(history, tools, system)` → `AsyncIterator[StreamEvent]`
2. Accumulate `TextDelta` and `ToolUseStart`/`ToolInputDelta` from events
3. On `IterationEnd(stop_reason=tool_use)`:
   - Parse fully accumulated tool calls
   - Dispatch all pending tool calls **concurrently** via `asyncio.gather()`
   - Append results as `ToolResultBlock` messages to context
   - Loop
4. On `IterationEnd(stop_reason=end_turn)`:
   - Append assistant message with text content
   - Emit `SessionEndEvent`
   - Return

Tool dispatch happens **before** appending to context. This prevents orphaned `ToolUseBlock`s if a task is cancelled
mid-loop. Without a `deferred_tool_sink`, cancellation stops unfinished tools and records interrupted results. A host
may install a sink to retain an in-flight task: the agent first closes the old tool protocol with a continuation
placeholder, then the host delivers the eventual result as a new user message. Never append a second `ToolResultBlock`
for the closed call.

### axio-repl interactive architecture

`axio-repl` has contracts that are easy to break with a locally reasonable terminal or concurrency change:

- It is POSIX-only and stays on the primary screen buffer. Do not enable the alternate screen or use `DECSTBM` scroll
  regions; completed output must remain in ordinary terminal scrollback.
- `prompt_toolkit` owns only editor/history/key handling and the temporary status lines. `_terminal.TerminalUI` is the
  sole interactive terminal writer. Application and background code may use the wrapped stdout/stderr, but must not
  write directly to the prompt output backend.
- `_prompt_terminal.PromptToolkitInlineOutput` is the only compatibility boundary allowed to use private
  `prompt_toolkit` redraw APIs. Keep the dependency upper bound and fail-fast compatibility validation; do not spread
  those private calls through renderer or coordinator code.
- Output ingress is serialized and bounded. Preserve explicit suppression markers, late-write fallback, startup
  rollback, terminal-failure propagation, and restoration of cursor, autowrap, streams, logging handlers, and input
  state.
- `SessionEventHub` defines one monotonic order across user input, peer messages, lifecycle events, context commits,
  and tool events. A non-empty Enter reserves its sequence synchronously before the prompt accept handler returns;
  every exit path after accept must complete that reservation.
- Enter is the only operation that queues editor text. Up recalls every still-pending message and joins text only in
  the editor with `"\n\n"`; a later Enter creates one new message. Escape never submits, clears, or modifies the editor.
- Persistent scrollback and model context are two views of the same semantic conversation stream. Render UI-only
  state such as startup details, command feedback, queue warnings, lifecycle summaries, and interruption causes in
  the temporary panel; do not mix it into conversation output.
- Slash commands belong to the UI plane. Discard their reserved ingress sequence, show their bounded feedback only in
  the temporary panel, and queue unsafe commands separately until a turn boundary. Do not publish them as
  `InputReceived`, put them in pending user input, append them to model context, or print their output into scrollback.
  Durable effects such as a model or configuration change still get their typed runtime event.
- Escape claims **all** pending messages and sends them to the focused agent as distinct `Message` objects. With no
  pending input it records an interrupt only and starts no replacement turn. Repeated Escape for the same captured
  turn is idempotent; stale interrupts must not cancel a replacement turn.
- Agent-visible arrivals must be exposed at the earliest safe provider boundary in hub order. If a foreground tool
  blocks that boundary, request deferral, close its old protocol with a placeholder, and deliver its real result later
  as a labelled user message exactly once.
- Journal events for input transitions, interruption, editor snapshots, context mutation, recovery, and shutdown are
  durability boundaries. Recovery must preserve available partial output plus the nature and identity of the
  interruption; do not correlate inputs by equal text when a source input ID exists.
- Every coloured physical line must establish and close its own SGR state. Reapply reasoning/tool styles after each
  newline and reset before another semantic block so asynchronous redraw cannot leak colours.

When changing these paths, run the focused `axio-repl` and `axio-tools-agents` suites, including PTY, repeated-Escape,
cancellation, recovery, ordering, and renderer-boundary tests, before the normal repository-wide checks.

### Tool system (`axio/tool.py`)

A tool handler is a plain `async def` function. Parameters become the input JSON schema; the docstring becomes the description. Use `Annotated` + `Field` from `axio.field` to add per-parameter descriptions, defaults, or numeric bounds.

```python
async def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path."""
    Path(path).write_text(content)
    return f"wrote {len(content)} bytes"
```

`Tool` is a `dataclass(frozen=True, slots=True)`:

```python
@dataclass(frozen=True, slots=True)
class Tool[T]:
    name: ToolName                         # Unique identifier
    handler: Callable[..., Awaitable[str]] # Plain async function
    description: str = ""                  # Defaults to handler.__doc__
    guards: tuple[PermissionGuard, ...] = ()  # Run sequentially; any GuardError denies
    concurrency: int | None = None         # Optional per-tool semaphore limit
    context: T = ...                       # Runtime state for CONTEXT.get()
```

`Tool.__call__(**kwargs)` pipeline:
1. Acquire semaphore (if `concurrency` is set)
2. Field validation from type hints / `FieldInfo`
3. Guards run **sequentially**: each receives `(tool, **kwargs)` and returns modified kwargs or raises `GuardError`
4. `await handler(**kwargs)` - execute handler; an unexpected exception from either step is
   wrapped in `HandlerCrash` / `GuardCrash`

### ContextStore (`axio/context.py`)

`ContextStore` is an ABC. Custom stores must implement the two abstract methods:

```python
class ContextStore(ABC):
    @abstractmethod
    async def append(self, message: Message) -> None: ...

    @abstractmethod
    async def get_history(self) -> list[Message]: ...
```

The base class also provides default implementations for `append_many()`, `fork()`, `clear()`, token accounting,
session listing, and `close()`. Override `append_many()` with one atomic transaction in persistent stores. Override
the other optional operations when the backend supports them; `clear()` raises `NotImplementedError` by default,
while `fork()` returns an independent `MemoryContextStore` copy.

**Implementations**:
- `MemoryContextStore` - in-process, ephemeral storage
- `axio-context-sqlite` - `SQLiteContextStore` for persistence across sessions

### PermissionGuard (`axio/permission.py`)

`PermissionGuard` is an ABC. Implement:

```python
class PermissionGuard(ABC):
    @abstractmethod
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        """Return (possibly modified) kwargs to allow, raise GuardError to deny."""
        ...
```

Guards are not limited to access control. Because `check()` receives the `Tool` object
and the raw kwargs before the handler executes, guards are also the right place for
**logging, auditing, and display**:

```python
class AuditGuard(PermissionGuard):
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        logger.info("tool=%s args=%s", tool.name, kwargs)
        return kwargs  # always allow; raise GuardError to deny
```

See `examples/agent_swarm/agent_swarm/__main__.py` (`RoleGuard`) for a production example.

**ConcurrentGuard**: Use as base when the guard itself must be rate-limited (e.g. LLM-based approval). Provides internal semaphore management.

### Exceptions (`axio/exceptions.py`)

Full exception hierarchy:

```python
class AxioError(Exception):
    """Base exception for all axio errors."""

class ToolError(AxioError):
    """Base for tool-related errors."""

class GuardError(ToolError):
    """Guard denied the tool call."""

class GuardCrash(GuardError):
    """A guard implementation crashed, as opposed to deliberately denying."""

class HandlerError(ToolError):
    """Expected tool failure, reported to the model."""

class HandlerCrash(HandlerError):
    """An unexpected exception escaped a tool handler."""

class StreamError(AxioError):
    """Error during stream collection."""
```

`Tool` raises the `*Crash` variants itself when an exception escapes a handler or a guard;
a `HandlerError`/`GuardError` raised deliberately propagates unchanged, and so does the
`HandlerError` produced by input validation. Agent tool dispatch turns every one of them
into `ToolResultBlock(is_error=True)`, logging expected failures at `INFO` and crashes at
`ERROR` with a traceback.

### Testing (`axio/testing.py`)

Helper classes and functions for testing:

| Function | Returns | Purpose |
|---|---|---|
| `StubTransport(responses)` | `StubTransport` | Yields pre-configured event sequences per `stream()` call |
| `make_text_response(text, iteration, usage)` | `list[StreamEvent]` | Build a simple end_turn response |
| `make_tool_use_response(tool_name, tool_id, tool_input, iteration, usage)` | `list[StreamEvent]` | Build a tool_use response sequence |
| `make_stub_transport()` | `StubTransport` | Pre-configured with "Hello world" text response |
| `make_ephemeral_context()` | `MemoryContextStore` | Fresh empty context |
| `make_echo_tool()` | `Tool` | Tool with a plain async handler that returns its `msg` arg as JSON |

```python
# Example: StubTransport with multiple responses
async def example(my_tool: Tool) -> str:
    transport = StubTransport([
        make_tool_use_response("my_tool", tool_input={"x": 1}),
        make_text_response("Done"),
    ])
    agent = Agent(system="...", transport=transport, tools=[my_tool])
    return await agent.run("go", make_ephemeral_context())
```

`StubTransport` pops the next event sequence on each `stream()` call. If there are fewer sequences than calls, it repeats the last one.

### Plugin system (entry points)

Plugins register via `pyproject.toml` entry points and are discovered by `axio-tui` at startup:

| Group | Registers |
|---|---|
| `axio.tools` | plain async handler functions |
| `axio.tools.settings` | `ToolsPlugin` (dynamic tool sets, e.g. MCP) |
| `axio.transport` | `CompletionTransport` implementations |
| `axio.transport.settings` | TUI settings screens (Textual `Screen` subclasses) |
| `axio.guards` | `PermissionGuard` subclasses |

---

## Testing

### Unit tests

Each package has `tests/` with pytest. Test files follow the pattern `test_<module>.py`.

Use helpers from `axio.testing`:

```python
from axio.testing import (
    StubTransport,          # pre-configured event sequences
    make_text_response,     # build an end_turn event list
    make_tool_use_response, # build a tool_use event list
    make_stub_transport,    # StubTransport with a single "Hello world" response
    make_ephemeral_context, # fresh MemoryContextStore
    make_echo_tool,         # Tool(name="echo", handler=MsgInput)
)
```

`StubTransport` pops the next event sequence on each `stream()` call. If there are fewer sequences than calls, it repeats the last one.

```python
async def example_with_tool_use(my_tool: Tool) -> str:
    transport = StubTransport([
        make_tool_use_response("my_tool", tool_input={"x": 1}),
        make_text_response("Done"),
    ])
    agent = Agent(system="...", transport=transport, tools=[my_tool])
    return await agent.run("go", make_ephemeral_context())
```

### Doc tests

Documentation in `docs/` is tested with [markdown-pytest](https://github.com/mosquito/markdown-pytest). Annotate code blocks with HTML comments:

```markdown
<!-- name: test_my_example -->
```python
import asyncio
from axio.agent import Agent
# ... asyncio.run() for async code
```
```

Hidden setup (stubs that must not appear in rendered docs):

```markdown
<!--
name: test_my_example
```python
# This block is invisible in docs but runs before the named block
from axio.testing import StubTransport, make_text_response
```
-->
```

Run doc tests:

```bash
make test-docs
# or for a single file:
uv run --directory docs pytest -v guides/writing-transports.md
```

---

## Adding a new package

1. Create `axio-<name>/` with a `src/axio_<name>/` layout and a `pyproject.toml` matching the style of existing packages (hatchling build backend, ruff + mypy + pytest dev deps, `asyncio_mode = "auto"`).
2. Add to `[tool.uv.workspace] members` and `[tool.uv.sources]` in the root `pyproject.toml`.
3. Add to `PACKAGES` in `Makefile`.
4. Run `uv sync --all-packages` to update `uv.lock`.

---

## What not to do

- **Do not** run `uv run pytest` or `ruff` from the repo root - use `make`.
- **Do not** add dependencies to the root `pyproject.toml` - it is a workspace manifest only.
- **Do not** edit `uv.lock` manually - it is generated by `uv sync`.
- **Do not** use `asyncio_mode = "auto"` in ad-hoc scripts - it is only for pytest.
- **Do not** add a guard's blocking I/O in the hot path without subclassing `ConcurrentGuard`.
