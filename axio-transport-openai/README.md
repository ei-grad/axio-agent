# axio-transport-openai

[![PyPI](https://img.shields.io/pypi/v/axio-transport-openai)](https://pypi.org/project/axio-transport-openai/)
[![Python](https://img.shields.io/pypi/pyversions/axio-transport-openai)](https://pypi.org/project/axio-transport-openai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

OpenAI Responses and OpenAI-compatible Chat Completions transports for
[axio](https://github.com/mosquito/axio-agent).

`OpenAITransport` always uses OpenAI's `/v1/responses` endpoint.
`ChatCompletionsTransport` and the Nebius, OpenRouter, llama.cpp, and custom
provider transports always use `/v1/chat/completions`. There is no endpoint
auto-detection or model-based routing.

## Features

- **Full SSE streaming** - parses `data:` chunks incrementally; no waiting for full responses
- **Automatic retry** - configurable backoff on 429 and 5xx responses; honours `Retry-After` header
- **Tool calling** - streams tool-use JSON fragments as `ToolInputDelta` events; parallel tool calls supported
- **Reasoning support** - `<think>...</think>` blocks emitted as `ReasoningDelta` events
- **Embeddings** - `embed()` method for models that support `/v1/embeddings`
- **Sub-transports** - provider-specific transports for Nebius, OpenRouter, llama.cpp, and custom endpoints
- **aiohttp-based** - zero blocking I/O
- **Optional TUI settings screen** - install with `[tui]` extra for a Textual configuration UI

## Installation

```bash
pip install axio-transport-openai
```

With TUI settings screens:

```bash
pip install "axio-transport-openai[tui]"
```

## Usage

```python
import asyncio
import aiohttp
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import TextDelta
from axio_transport_openai import OpenAITransport, OPENAI_MODELS

async def main() -> None:
    async with aiohttp.ClientSession() as session:
        transport = OpenAITransport(
            api_key="sk-...",
            model=OPENAI_MODELS["gpt-4.1-mini"],
            session=session,
        )
        agent = Agent(system="You are a helpful assistant.", tools=[], transport=transport)
        ctx = MemoryContextStore()
        async for event in agent.run_stream("What is 2 + 2?", ctx):
            if isinstance(event, TextDelta):
                print(event.delta, end="", flush=True)
        print()

asyncio.run(main())
```

The `session` parameter is **required** for streaming. Create an `aiohttp.ClientSession` in an async context and pass it to the transport.

### Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Default API key if not passed to the constructor |
| `OPENAI_BASE_URL` | Default base URL (falls back to `https://api.openai.com/v1`) |
| `LLAMA_API_KEY` | Optional llama.cpp API key (empty by default) |
| `LLAMA_CPP_BASE_URL` | llama.cpp OpenAI-compatible base URL (defaults to `http://127.0.0.1:8080/v1`) |

### Local models (Ollama, vLLM, LM Studio)

```python
from axio.models import ModelSpec, Capability
from axio_transport_openai import ChatCompletionsTransport

transport = ChatCompletionsTransport(
    api_key="ollama",                        # any non-empty string
    model=ModelSpec(id="llama3.2", capabilities=frozenset({Capability.text})),
    base_url="http://localhost:11434/v1",
)
```

### Streaming events

```python
import asyncio
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response
from axio.events import TextDelta, SessionEndEvent

agent = Agent(
    system="",
    tools=[],
    transport=StubTransport([make_text_response("Why did the chicken cross the road?")]),
)

async def main() -> None:
    ctx = MemoryContextStore()
    async for event in agent.run_stream("Tell me a joke", ctx):
        match event:
            case TextDelta(delta=text):
                print(text, end="", flush=True)
            case SessionEndEvent(total_usage=usage):
                print(f"\n[{usage.input_tokens}in / {usage.output_tokens}out tokens]")

asyncio.run(main())
```

### Cost accounting

`IterationEnd.usage.cost_usd` is populated only when the configured provider
reports a validated, non-negative USD cost for that operation. OpenRouter's
terminal `usage.cost` is such a value, so `OpenRouterTransport` exposes it with
`cost_source == CostSource.provider`. It is not recomputed from catalogue token
prices and is recorded only once even if a compatible endpoint sends more than
one usage chunk. OpenRouter's [usage-accounting documentation](https://openrouter.ai/docs/cookbook/administration/usage-accounting)
describes `usage.cost` as the total charged and says detailed usage is included
automatically in the final streaming message; it also marks
`usage: {"include": true}` and `stream_options: {"include_usage": true}` as
deprecated no-ops. Its
[currency FAQ](https://openrouter.ai/docs/faq) states that API pricing is
denominated in US dollars.

The documented Chat Completions usage schemas for
[OpenAI](https://developers.openai.com/api/reference/resources/chat) and
[Nebius](https://docs.tokenfactory.nebius.com/api-reference/inference/create-chat-completion)
expose token counts but no monetary total. Their `cost_usd` and `cost_source`
fields therefore remain `None` unless the configured endpoint supplies a cost
under a provider-specific contract that this transport recognizes. Hosts may
estimate from `ModelSpec.input_cost` and `ModelSpec.output_cost`; `axio-repl`
labels that fallback as `est.` and labels provider totals as `reported`. The
estimate is not a billing statement and may differ when cached tokens,
long-context or service tiers, request fees, images, or hosted tools have
separate prices.

Usage normally arrives in the terminal SSE chunk. If a stream is interrupted
before that chunk, the transport does not fabricate token usage or cost.

## Configuration reference

`OpenAITransport` and `ChatCompletionsTransport` share the following configuration fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"OpenAI"` | Display name (used by TUI) |
| `api_key` | `str` | `$OPENAI_API_KEY` | API key |
| `base_url` | `str` | `$OPENAI_BASE_URL` or `https://api.openai.com/v1` | API base URL |
| `model` | `ModelSpec` | `OPENAI_MODELS["gpt-5.6-terra"]` | Active model |
| `models` | `ModelRegistry` | all `OPENAI_MODELS` | Available models |
| `session` | `aiohttp.ClientSession \| None` | `None` | HTTP session (required for streaming) |
| `max_retries` | `int` | `10` | Maximum retry attempts on 429 / 5xx |
| `retry_base_delay` | `float` | `5.0` | Base delay in seconds for exponential backoff |

## Models

| Model ID | Context | Max output | Capabilities | Price (in/out per M tokens) |
|---|---|---|---|---|
| `gpt-5.4` | 1,050,000 | 128,000 | text, vision, tool use | $10 / $40 |
| `gpt-5.4-mini` | 400,000 | 128,000 | text, vision, tool use | $1.50 / $6 |
| `gpt-5.4-nano` | 400,000 | 128,000 | text, tool use | $0.30 / $1.20 |
| `gpt-5.1` | 400,000 | 128,000 | text, vision, tool use | $5 / $20 |
| `gpt-5` | 400,000 | 128,000 | text, vision, tool use | $5 / $20 |
| `gpt-5-mini` | 400,000 | 128,000 | text, vision, tool use | $1.25 / $5 |
| `gpt-5-nano` | 400,000 | 128,000 | text, tool use | $0.25 / $1 |
| `o4-mini` | 200,000 | 100,000 | text, reasoning, tool use | $1.10 / $4.40 |
| `o3` | 200,000 | 100,000 | text, reasoning, tool use | $10 / $40 |
| `o3-mini` | 200,000 | 100,000 | text, reasoning, tool use | $1.10 / $4.40 |
| `gpt-4.1` | 1,047,576 | 32,768 | text, vision, tool use | $2 / $8 |
| `gpt-4.1-mini` | 1,047,576 | 32,768 | text, vision, tool use | $0.40 / $1.60 |
| `gpt-4.1-nano` | 1,047,576 | 32,768 | text, tool use | $0.10 / $0.40 |
| `gpt-4o` | 128,000 | 16,384 | text, vision, tool use | $2.50 / $10 |
| `gpt-4o-mini` | 128,000 | 16,384 | text, vision, tool use | $0.15 / $0.60 |

The default model is `gpt-5.6-terra`.

## `fetch_models()`

`await transport.fetch_models()` resets `transport.models` to the built-in `OPENAI_MODELS` registry. It does not make a network request. Override `model` directly to switch the active model.

## Serialisation

`OpenAITransport` supports JSON round-trip for storing and restoring configuration:

```python
# Serialise
data = transport.to_dict()   # -> {"name": ..., "base_url": ..., "model": ..., "models": [...]}

# Restore
transport = OpenAITransport.from_dict(data, session=session)
```

`from_dict` falls back to `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables if the stored values are empty.

## Sub-transports

### NebiusTransport

`NebiusTransport` connects to [Nebius AI Studio](https://studio.nebius.com/) (`https://api.tokenfactory.nebius.com/v1`). It uses the Chat Completions protocol and shares retry and streaming behaviour with `ChatCompletionsTransport`.

```python
from axio_transport_openai.nebius import NebiusTransport

transport = NebiusTransport(
    api_key="...",          # or set NEBIUS_API_KEY
    session=session,
)
```

| Field | Default |
|---|---|
| `name` | `"Nebius AI Studio"` |
| `api_key` | `$NEBIUS_API_KEY` |
| `base_url` | `https://api.tokenfactory.nebius.com/v1` |
| `model` | `deepseek-ai/DeepSeek-V3-0324` |

`fetch_models()` queries `/v1/models?verbose=true` and populates `transport.models` with all models returned by the API, including their context windows, output limits, capabilities (text, vision, tool use, embedding), and pricing.

### OpenRouterTransport

`OpenRouterTransport` connects to [OpenRouter](https://openrouter.ai/) (`https://openrouter.ai/api/v1`), which provides a unified API over hundreds of models from many providers.

```python
from axio_transport_openai.openrouter import OpenRouterTransport

transport = OpenRouterTransport(
    api_key="...",          # or set OPENROUTER_API_KEY
    session=session,
)
```

| Field | Default |
|---|---|
| `name` | `"OpenRouter"` |
| `api_key` | `$OPENROUTER_API_KEY` |
| `base_url` | `https://openrouter.ai/api/v1` |
| `model` | `google/gemini-2.5-pro-preview` |

`fetch_models()` queries `/v1/models` and populates `transport.models` with all models returned by the API, including their context windows, output limits, capabilities (text, vision, tool use, reasoning, embedding), and pricing.

OpenRouter's response-level `usage.cost` takes precedence over those catalogue
prices because it includes the actual route and applicable non-token charges.

Model ids follow `[<lab>/]<model>[:tier][@<provider>]`:

| Id | Means |
|---|---|
| `z-ai/glm-4.7` | OpenRouter picks the provider |
| `z-ai/glm-4.7:nitro` | OpenRouter's own routing tier, sent as part of the model name |
| `z-ai/glm-4.7@Cerebras` | served by Cerebras only, via `provider.only` |
| `z-ai/glm-4.7:nitro@DeepInfra` | both |

The `@<provider>` part is not an OpenRouter model name: it is stripped off when the request is built and sent as `provider: {"only": [...]}`. Tier and provider suffixes stay in `model.id`, and the metadata of the base model applies to them.

### CustomChatCompletionsTransport

`CustomChatCompletionsTransport` is the explicit Chat Completions transport for
user-defined providers. Instances are created by the TUI hub screen and
persisted to `~/.local/share/axio/openai-custom.json`. You can also instantiate
them directly:

`OpenAICompatibleTransport` remains a public compatibility alias for this
class and is also the target of the `openai-custom` plugin entry point.

```python
from axio.models import ModelSpec, ModelRegistry, Capability
from axio_transport_openai.custom import CustomChatCompletionsTransport

transport = CustomChatCompletionsTransport(
    name="localai",
    base_url="http://localhost:8080/v1",
    api_key="",
    models=ModelRegistry([
        ModelSpec(
            id="llama3.2",
            context_window=131_072,
            max_output_tokens=4_096,
            capabilities=frozenset({Capability.text, Capability.tool_use}),
        )
    ]),
    session=session,
)
```

`fetch_models()` is a no-op for this transport - the model list is provided at construction time.

The JSON configuration format used by the TUI is:

```json
[
  {
    "name": "localai",
    "base_url": "http://localhost:8080/v1",
    "api_key": "",
    "model": "llama3.2",
    "models": [
      {
        "id": "llama3.2",
        "context_window": 131072,
        "max_output_tokens": 4096,
        "capabilities": ["text", "tool_use"],
        "input_cost": 0.0,
        "output_cost": 0.0
      }
    ]
  }
]
```

### LlamaCppTransport

`LlamaCppTransport` keeps generation on llama.cpp's OpenAI-compatible
`/v1/chat/completions` endpoint and uses the native read-only endpoints only for
model discovery:

```python
from axio_transport_openai.llamacpp import LlamaCppTransport

transport = LlamaCppTransport(session=session)
await transport.fetch_models()
```

The default URL is `http://127.0.0.1:8080/v1`. `LLAMA_CPP_BASE_URL` and the
optional `LLAMA_API_KEY` override the defaults. An explicit empty key stays
empty and does not fall back to `OPENAI_API_KEY`.

Discovery reads `/props` first and uses its explicit `role: "router"` marker,
with `/models` status objects as a compatibility fallback, to distinguish the
two server modes. In single-model mode, model metadata comes from
`/v1/models`. Use `--alias <stable-name>` when starting `llama-server`;
otherwise the model ID is normally its GGUF path. Runtime `n_ctx` from `/props`
takes priority over the training context published in model metadata. A
positive server `n_predict` limits the advertised output size.

In router mode, discovery first reads native `/models`, then reads
`/props?model=<id>&autoload=false` only for entries already in `loaded` or
`sleeping` state. Unloaded, loading, downloading, and failed entries are not
published in Axio's registry. This avoids presenting a model whose first use
would silently load it. Refresh is side-effect-free: it never sends
`reload=1`, calls load/unload/download endpoints, or changes llama.cpp's model
cache.

Tool use is advertised only when `/props.chat_template_caps` explicitly reports
both `supports_tools` and `supports_tool_calls`. Run current llama.cpp with
Jinja chat templating enabled (normally the default; `--jinja` is explicit and
portable across older versions) and use a model template that supports tool
calls. Vision, audio, video, and embedding capabilities likewise come only
from published modality metadata. Text requires explicit text output or
published chat-template metadata; missing output metadata is not treated as
proof of completion support. `supports_preserve_reasoning` is not treated as
proof that a model reasons; streamed `reasoning_content` is still emitted as
`ReasoningDelta` by the inherited parser.

## Plugin registration

When installed, this package registers five completion transports via entry points so `axio-tui` discovers them automatically:

```toml
[project.entry-points."axio.transport"]
openai         = "axio_transport_openai:OpenAITransport"
nebius         = "axio_transport_openai.nebius:NebiusTransport"
openrouter     = "axio_transport_openai.openrouter:OpenRouterTransport"
openai-custom  = "axio_transport_openai.custom:OpenAICompatibleTransport"
llama-cpp      = "axio_transport_openai.llamacpp:LlamaCppTransport"
```

## Migration from 0.2.3

`OpenAITransport` now means the official OpenAI Responses API exclusively.
Use `ChatCompletionsTransport` for generic compatible endpoints. The former
`OpenAIResponsesTransport` and the `openai-responses` entry point were removed
without aliases. `OpenAICompatibleTransport` remains as an alias for the
explicit `CustomChatCompletionsTransport`. Persisted TUI role bindings named
`openai-responses:<model>` are migrated once to `openai:<model>`; command-line
selections must use `openai`.

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-transport-codex](https://github.com/mosquito/axio-agent) · [axio-transport-anthropic](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent)

## License

MIT
