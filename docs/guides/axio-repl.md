# axio-repl

`axio-repl` is a terminal coding assistant that runs an Axio agent with file and
shell tools. It auto-detects your LLM backend from environment variables and
streams every token and tool call directly to the terminal.

## Install

```bash
uv tool install axio-repl
```

Add optional transports:

```bash
# Anthropic Claude
uv tool install axio-repl --with axio-transport-anthropic

# Google Gemini
uv tool install axio-repl --with axio-transport-google
```

Within the monorepo workspace:

```bash
uv run axio-repl
```

## Start the REPL

```bash
axio-repl
```

The REPL picks the first transport whose environment variable is set:

| Transport | Env Variable | Package |
|---|---|---|
| `google` | `GEMINI_API_KEY` | `axio-transport-google` |
| `google-vertex` | `GOOGLE_GENAI_USE_VERTEXAI` | `axio-transport-google` |
| `anthropic` | `ANTHROPIC_API_KEY` | `axio-transport-anthropic` |
| `openai` | `OPENAI_API_KEY` | `axio-transport-openai` |
| `nebius` | `NEBIUS_API_KEY` | `axio-transport-openai` |
| `openrouter` | `OPENROUTER_API_KEY` | `axio-transport-openai` |

Override with `--transport <name>`:

```bash
axio-repl --transport anthropic --model claude-sonnet-4-20250514
axio-repl --transport google --model gemini-3.1-pro-preview
```

## Single-prompt mode

Pass a prompt as an argument for non-interactive use:

```bash
axio-repl "list the files in this project"
axio-repl --transport openai "write tests for src/auth.py"
```

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--transport` | auto | Transport name (see table above) |
| `--model` | transport default | Model name |
| `--temperature` | transport default | Sampling temperature |
| `--effort` | `default` | Effort: `none`, `low`, `medium`, `high`, `xhigh`, or `max` |
| `--max-tokens` | transport default | Max output tokens |
| `--max-iterations` | 1000 | Max agent iterations |
| `--debug` | off | Log raw request/response bodies |
| `--agent-actions` | off | Show framed actions from non-active agents (`on` or `off`) |
| `--session-log-dir` | XDG state directory | Root for session JSONL journals |
| `--no-session-log` | off | Disable the default session journal |
| `--sandbox` | auto | Run file and shell tools in a container: `auto`, `docker`, `none` |
| `--sandbox-image` | `python:3.12-slim` | Image for `--sandbox docker` |
| `--sandbox-network` | none | User-defined internal Docker network for restricted service access |
| `--sandbox-memory` | `256m` | Container memory limit |
| `--sandbox-cpus` | `1.0` | Container CPU limit |
| `--sandbox-proxy` | none | HTTP(S) policy proxy URL |
| `--sandbox-no-proxy` | none | Comma-separated internal proxy bypass hostnames |
| `--sandbox-pypi-index` | none | Internal PyPI/simple index URL |
| `--sandbox-npm-registry` | none | Internal npm registry URL |
| `--sandbox-cargo-index` | none | Internal Cargo registry index URL |
| `--sandbox-go-proxy` | none | Internal Go module proxy URL |
| `--sandbox-go-sumdb` | Go default | Explicit `GOSUMDB` setting for an internal checksum database/proxy |
| `--sandbox-datasets` | none | Host directory mounted read-only at `/datasets` |
| `--sandbox-ca-cert` | none | Full CA bundle (system roots plus interception CA) mounted read-only for common clients |

## Session journals

The REPL writes an append-only JSONL journal by default. Each invocation creates:

```text
${XDG_STATE_HOME:-~/.local/state}/axio/sessions/YYYY/MM/DD/<session-id>/events.jsonl
```

The printed `Session log:` line gives the exact path. It is sent to stderr in
single-prompt mode so stdout remains the agent's streamed answer. Choose another
root with `--session-log-dir <directory>`, or disable the journal explicitly
with `--no-session-log`.

The log is independent of terminal focus and display filtering. It includes the
main agent and hidden foreground/background subagents: user input, stream
events, successfully committed context messages, lifecycle events, configuration
changes, and the correlation between child outcomes and their delivery routes.
Media bytes are stored by content hash in an adjacent `attachments/` directory.

`session_start` is written and fsynced before normal session work begins. The
hot streaming path only admits records to a bounded in-memory queue; admission
does not claim that an individual record is durable. The journal drains and
fsyncs at successfully committed context mutations, completed turns, outcome
delivery, agent shutdown, and clean session shutdown. This preserves live
streaming while giving completed work explicit durability boundaries.

If the process stops abruptly, every record through the last successful
boundary is retained. Every record accepted after that boundary may
be lost; alternatively, some prefix can reach disk. The on-disk result is valid
newline-delimited JSON through all complete records, with at most one final
unterminated line from an interrupted write. `read_journal()` ignores only that
tail, and `recover_journal_tail()` validates the complete prefix before
truncating it. Malformed newline-terminated or earlier records are reported as
corruption rather than silently skipped.

This recovery rule does not mask storage-media corruption or broader filesystem
failures; those remain hard errors.

Session directories use mode `0700` and journal or attachment files use `0600`.
Known secret-shaped fields and common token formats are redacted recursively,
but a secret embedded in arbitrary prose or tool output cannot always be
recognized. Treat journals as sensitive local data.

## REPL commands

| Command | Description |
|---|---|
| `/model` | Show current model and list available models |
| `/model <query>` | Switch to a model matching the query |
| `/effort [level]` | Show or set effort; `default` restores the provider/model default |
| `/temperature [val]` | Show or set sampling temperature |
| `/max-tokens [val]` | Show or set max output tokens |
| `/iterations [val]` | Show or set max agent iterations |
| `/debug [on\|off]` | Toggle request/response debug logging |
| `/agent-actions [on\|off]` | Show or toggle actions from non-active agents |
| `/agents` | List local background agents |
| `/agent-focus <id>` | Change the input target without changing execution mode |
| `/help` | List all tools and commands |
| `/quit`, `/exit`, `/q` | Exit the REPL |

`/effort` reports the requested level, the effective mechanism, and valid values. Native controls are shown as
`native-effort` or `native-budget`; transports without verified granular control use `prompt-fallback`. The fallback
adds one replaceable system-prompt overlay describing observable analysis and verification behavior. It does not add
a conversation message and does not claim control over provider reasoning tokens, latency, or cost. `/effort default`
removes the explicit native setting or prompt overlay and restores the provider/model default. For `native-effort`,
the requested level must exactly match a level advertised for the selected model; unsupported levels are rejected
instead of being mapped or routed through prompt fallback. A model switch reapplies the requested level, or resets it
to `default` with an explicit message when the new model does not support that exact level.

## Foreground delegation and agent actions

The `run_agent` tool runs a one-shot child in the foreground. The parent tool
call waits, while the child's reasoning, text, tool arguments, output, media,
and errors stream through the same immediate renderer path as the active parent.
The user's input target remains unchanged. When the child finishes, its full
answer is returned to the parent as a single tool result and is not printed a
second time.

When a sibling parent tool streams while `run_agent` owns the foreground, the
REPL queues that sibling action separately and inserts it at the child's nearest
safe boundary. These frames remain visible with `/agent-actions off` because
they belong to the active parent turn, not a background agent.

Each live-streamed turn starts with a source header. The root answer in
single-prompt mode is the exception: it keeps its plain stdout projection
without a header. When an agent has a human name, headers, action frames,
summaries, errors, and incoming reports identify it as `name (agent_id)`;
otherwise they show the authoritative agent id.

`spawn_agent` creates a persistent background agent. `/agent-actions on` makes
its tool and lifecycle activity visible without mixing its free-form prose or
reasoning into the active answer. Every background action is a labelled,
newline-terminated frame. Frames are inserted only at safe boundaries: after a
complete active paragraph, after reasoning or media, or after all parallel
active tool calls complete. They never split streamed tool JSON or active tool
output.

The queues are bounded and drained round-robin, so a noisy agent cannot block
the active stream or monopolize a boundary. Overflow produces an explicit
suppression marker. Switching the mode off discards queued presentation frames;
switching it on does not replay old activity. Display mode is independent of
input focus, scheduling, parent delivery, context, and the JSONL session log.

For a single-prompt invocation, spawned background agents are joined before the
process exits. Their final reports use the normal incoming-outcome path exactly
once; the REPL does not temporarily focus an agent and replay its hidden prose.

## Tools

| Tool | Description |
|---|---|
| `read_file` | Read file contents; images and videos are returned as vision blocks |
| `write_file` | Create or overwrite files |
| `patch_file` | Replace line ranges (1-indexed, inclusive) |
| `list_files` | List directory contents |
| `search_files` | Text or regex search across files |
| `shell` | Run shell commands with streaming stdout/stderr |
| `generate_image` | Generate images via Gemini (Google transport only) |
| `generate_video` | Generate videos via Veo (Google transport only) |

## Sandbox

`--sandbox` decides where those tools act. The default is `auto`: a container is
used whenever `aiodocker` is installed and `/var/run/docker.sock` exists, so a
machine with Docker running gets a sandbox without asking for one. The startup
banner states which it is — `Tools: docker — …` or `Tools: host — …`.

In a container the working directory is bind-mounted at `/workspace`, and that
is the path the system prompt gives the model. Host paths are meaningless to it.

The default image is `python:3.12-slim` and networking is off, which together
decide what the agent can and cannot do. The repository also contains a locally
buildable standard image and fail-closed support for an internal Docker network,
policy proxy, registry caches, and read-only dataset snapshots. See
[Docker Sandbox](docker-sandbox.md#from-axio-repl).

## AGENTS.md

Place an `AGENTS.md` file in the working directory to inject workspace-specific
instructions into the system prompt:

```markdown
# My project

- Always run `make test` after editing Python files
- The main entry point is `src/app.py`
- Use the `dev` branch for all changes
```

The file is loaded at startup and on every `/model` switch. It is optional -
if absent, the default system prompt is used unchanged.

## Capability-aware system prompt

The system prompt adapts to the selected model's capabilities. Switching models
with `/model` recalculates capabilities and rewrites the prompt automatically:

- **Vision** - `read_file` on images (PNG, JPG, GIF, WebP) returns pixel data.
  Screenshot-based UI review loops are unlocked.
- **Reasoning** - Extended thinking (chain-of-thought) is available.
- **Image generation** - Inline image generation via `generate_image`.
- **Video** - `read_file` on video files returns vision blocks.

## Multiline input

Paste multi-line text directly into the prompt. The REPL detects continuation
lines and joins them before sending:

```
You> Refactor this function:
...   def old(x):
...     return x+1
```

## Interrupting the agent

Press **Ctrl-C** to cancel the running agent loop. Partial tool output (stdout
already captured, files already written) is preserved in conversation context
so the model sees what happened and can resume cleanly.
