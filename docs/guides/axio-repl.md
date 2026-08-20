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

## Platform and terminal model

`axio-repl` is POSIX-only. Interactive mode requires a terminal with POSIX
`termios` behavior; Windows is not supported.

The interactive UI stays on the terminal's primary screen buffer. It does not
enter the alternate screen and does not reserve a scroll region, so completed
REPL output remains in the terminal's normal scrollback. `prompt_toolkit` owns
only the editor and status lines. Before asynchronous output is written, those
temporary lines are erased; one serialized terminal owner writes the output and
then redraws the editor at its previous cursor position.

All interactive stdout, stderr, logging, background-agent output, and prompt
redraws pass through that owner. Its queues are bounded; sustained overload is
reported with an explicit `terminal output skipped` marker instead of growing
memory without limit. Terminal write failures stop the session rather than
silently losing later output. On shutdown the REPL restores the cursor,
autowrap, process streams, and terminal input state.

Reasoning, answer text, tool fields, and tool output establish and reset their
own style on every physical terminal line. A later block therefore cannot
inherit an earlier block's colour after a newline or asynchronous redraw.
Untrusted model text, tool arguments, streaming tool channels, incoming reports,
and tool results pass through an incremental terminal-control filter before
those styles are applied. CSI, OSC, DCS, APC, cursor, screen, scrollback, and
clipboard controls are removed even when a sequence spans multiple chunks;
ordinary text, newlines, and tabs remain.

The bottom panel continuously reports the active agent phase: idle, waiting for
the model, reasoning, responding, or the names and counts of active tools. REPL
startup details, slash-command help/results, queue warnings, and interruption
causes are temporary panel feedback, not conversation output. Background
lifecycle summaries use the panel too; the actual background report remains in
the conversation log and model context. Accepting a slash command therefore
creates neither an `InputReceived` journal record nor a model message. A command
that changes durable configuration still records the resulting
`ConfigurationChanged` event.

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

## Persistent configuration and agent bundles

The REPL reads optional global defaults from
`${AXIO_CONFIG_DIR:-${XDG_CONFIG_HOME:-~/.config}/axio}/config.yaml`. Named
agents are self-contained directories below `agents/` and are selected with
`--agent`. A minimal layout uses these three paths:

```text
~/.config/axio/config.yaml
~/.config/axio/agents/local/agent.yaml
~/.config/axio/agents/local/instructions.md
```

List bundles without initializing a transport or Docker:

```bash
axio-repl --list-agents
```

The global file is optional. It supplies defaults shared by every invocation:

```yaml
version: 1
defaults:
  runtime:
    max_iterations: 100
    theme: default
    session_log: true
  sandbox:
    backend: docker
    image: axio-agent-sandbox:standard
    memory: 4g
    cpus: "2"
```

An agent manifest overlays those defaults. This bundle replaces the long
llama.cpp plus devpi command with `axio-repl --agent local`:

```yaml
# ~/.config/axio/agents/local/agent.yaml
version: 1
description: Local llama.cpp coding agent
model_context: |-
  Network access is routed through the configured local policy proxy.
  Denied requests are policy outcomes, not transient connectivity failures.
instructions:
  - instructions.md

transport:
  name: llama-cpp
  base_url: http://127.0.0.1:18080/v1

runtime:
  temperature: 0.2
  effort: high
  max_tokens: 16384
  max_iterations: 1000
  debug: false
  agent_actions: "off"
  theme: default
  powerline: false
  session_log: true

sandbox:
  backend: docker
  image: axio-agent-sandbox:standard
  network: axio-agent-egress
  memory: 4g
  cpus: "2"
  no_proxy: devpi
  registries:
    pypi: http://devpi:3141/root/pypi/+simple/

tools:
  - read_file
  - write_file
  - patch_file
  - list_files
  - search_files
  - shell
  - run_agent
  - spawn_agent
  - interrupt_agent
  - stop_agent
  - list_peers
  - send_message
  - monitor
```

`model` is optional; omit it to retain the transport default. The same applies
to every nested setting. `tools: [all]` enables the complete built-in set and
`tools: [none]` disables it. A named list preserves the REPL's canonical tool
order. Docker-only `run_python` and environment-dependent `ast_grep` can be
named when that backend actually provides them. Unknown or unavailable names,
duplicates, and mixing `all` or `none` with named tools are errors.

`model_context` is an optional trusted operator-policy description accepted
only in the selected `agent.yaml`; it is not a layered global default and has
no environment or CLI override. The exact block is supplied once to the main
agent and its local children. It describes policy but does not enforce it, so
do not put credentials, mutable guard state, or untrusted external content in
it. `description` remains catalog/UI text and is not sent to the model.

For an authenticated LLM endpoint, store only the environment-variable name:

```yaml
transport:
  name: openai
  base_url: https://llm.internal.example/v1
  api_key_env: AXIO_INTERNAL_LLM_TOKEN
```

Set `AXIO_INTERNAL_LLM_TOKEN` in the process environment or a service-manager
secret store. A missing reference is a startup error. The secret is passed to
the transport constructor; it is not retained in the resolved profile or
emitted as a configuration event. PyPI and Cargo URLs containing credentials
are rejected. No registry field is a secret channel; use an internal
download-only frontend when registry authentication is required.

The complete sandbox form is:

```yaml
sandbox:
  backend: docker              # auto, docker, or none
  image: axio-agent-sandbox:standard
  network: axio-agent-egress   # must be a user-defined Internal=true network
  memory: 4g
  cpus: "2"
  proxy: http://mitmania:3128
  no_proxy: nexus.internal,artifactory.internal
  registries:
    pypi: https://nexus.internal/repository/pypi-all/simple
    npm: https://nexus.internal/repository/npm-all/
    cargo: sparse+https://nexus.internal/repository/cargo-all/
    go: https://nexus.internal/repository/go-all/
    go_sumdb: sum.golang.org https://nexus.internal/repository/go-sumdb/
  datasets: /srv/axio-datasets
  ca_certificate: /srv/axio-pki/egress-ca-bundle.pem
```

The network/cache architecture and concrete devpi, Nexus, and Artifactory
recipes are documented in [Docker sandbox](docker-sandbox.md#restricted-packages-and-datasets).

### Resolution and validation

Configuration resolves in this order, from lowest to highest precedence:

1. built-in defaults;
2. global `config.yaml` defaults;
3. the selected `agents/<name>/agent.yaml`;
4. `AXIO_REPL_*` environment overrides;
5. explicitly supplied CLI flags.

Transport-native variables such as `LLAMA_CPP_BASE_URL` remain constructor
fallbacks only when no resolved `transport.base_url` is present. Select a bundle
with `AXIO_REPL_AGENT` instead of `--agent`, or relocate the whole configuration
root with `AXIO_CONFIG_DIR`/`--config-dir`.

The environment layer maps directly to manifest fields:

| Area | Variables |
|---|---|
| Agent/transport | `AXIO_REPL_AGENT`, `AXIO_REPL_TRANSPORT`, `AXIO_REPL_TRANSPORT_BASE_URL`, `AXIO_REPL_TRANSPORT_API_KEY_ENV`, `AXIO_REPL_MODEL` |
| Runtime | `AXIO_REPL_TEMPERATURE`, `AXIO_REPL_EFFORT`, `AXIO_REPL_MAX_TOKENS`, `AXIO_REPL_MAX_ITERATIONS`, `AXIO_REPL_DEBUG`, `AXIO_REPL_AGENT_ACTIONS`, `AXIO_REPL_THEME`, `AXIO_REPL_POWERLINE`, `AXIO_REPL_SESSION_LOG`, `AXIO_REPL_SESSION_LOG_DIR` |
| Sandbox | `AXIO_REPL_SANDBOX`, `AXIO_REPL_SANDBOX_IMAGE`, `AXIO_REPL_SANDBOX_NETWORK`, `AXIO_REPL_SANDBOX_MEMORY`, `AXIO_REPL_SANDBOX_CPUS`, `AXIO_REPL_SANDBOX_PROXY`, `AXIO_REPL_SANDBOX_NO_PROXY`, `AXIO_REPL_SANDBOX_DATASETS`, `AXIO_REPL_SANDBOX_CA_CERT` |
| Registries | `AXIO_REPL_SANDBOX_PYPI_INDEX`, `AXIO_REPL_SANDBOX_NPM_REGISTRY`, `AXIO_REPL_SANDBOX_CARGO_INDEX`, `AXIO_REPL_SANDBOX_GO_PROXY`, `AXIO_REPL_SANDBOX_GO_SUMDB` |
| Tools | `AXIO_REPL_TOOLS` as a comma-separated list, `all`, or `none` |

Both files require integer `version: 1`. Unknown or duplicate YAML keys are
errors. Instruction files must be relative paths confined to their bundle and
must exist; their combined UTF-8 content is limited to 1 MiB. Relative dataset,
CA, and session-log paths are confined to the file that declares them; absolute
paths are allowed. `agent.yaml` instructions are loaded before project
`AGENTS.md` instructions, and both are included in the system prompt. Mutable
session state and secrets are never written into an agent bundle.

Long CLI options must be written in full. Abbreviations such as `--temp` are
rejected so an abbreviated option cannot bypass CLI-over-config precedence.

## Single-prompt mode

Pass a prompt as an argument for non-interactive use:

```bash
axio-repl "list the files in this project"
axio-repl --transport openai "write tests for src/auth.py"
```

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--agent` | `AXIO_REPL_AGENT` or none | Named bundle below `CONFIG_DIR/agents` |
| `--config-dir` | XDG config directory | Configuration root; overrides `AXIO_CONFIG_DIR` |
| `--list-agents` | off | List named bundles and exit without starting a transport |
| `--version` | off | Show distribution, launcher, module, interpreter, and local Git provenance, then exit |
| `--transport` | auto | Transport name (see table above) |
| `--transport-base-url` | transport default | Transport API base URL |
| `--transport-api-key-env` | none | Environment variable containing the transport API key |
| `--model` | transport default | Model name |
| `--temperature` | transport default | Sampling temperature |
| `--effort` | `default` | Effort: `none`, `low`, `medium`, `high`, `xhigh`, or `max` |
| `--max-tokens` | transport default | Max output tokens |
| `--max-iterations` | 1000 | Max agent iterations |
| `--debug` | off | Log raw request/response bodies |
| `--no-debug` | off | Explicitly disable debug logging after config resolution |
| `--agent-actions` | off | Show framed actions from non-active agents (`on` or `off`) |
| `--theme` | `default` | Terminal palette: `default` or `monochrome` |
| `--powerline` | off | Use Powerline segments for the prompt, tool names, and agent frames |
| `--no-powerline` | — | Explicitly disable Powerline presentation after config resolution |
| `--session-log-dir` | XDG state directory | Root for session JSONL journals |
| `--no-session-log` | off | Disable the default session journal |
| `--session-log` | on | Explicitly enable the journal after config resolution |
| `--resume EVENTS_JSONL` | none | Resume a stopped interactive session into a new journal |
| `--sandbox` | auto | Run file and shell tools in a container: `auto`, `docker`, `none` |
| `--sandbox-image` | `axio-agent-sandbox:standard` | Locally built image for `--sandbox docker` |
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
| `--tools` | all | `all`, `none`, or a comma-separated tool whitelist |

Powerline presentation uses filled colour segments with `U+E0B0` (``) separators. The terminal font must provide
that glyph; the REPL does not substitute a plain-text fallback.

`NO_COLOR` disables application-owned ANSI styling and Powerline regardless of
the selected `--theme`. A one-shot invocation also selects this plain
presentation whenever stdout is not a TTY. Interactive `prompt_toolkit` still
uses terminal cursor controls for editor redraws; `NO_COLOR` governs colour and
presentation styling, not those required input-control sequences.

The active editor prompt contains only the effective-UID username: plain mode
renders `username>`, while Powerline mode renders a filled ` username ` segment
followed by ``. After Enter accepts a user message, the persistent scrollback
line gains a local `HH:MM` timestamp captured synchronously at that keypress.
Queueing, admission, and model-start delays do not change it; recalled text gets
a new timestamp when resubmitted. The username comes from the system account
database, not `USER`; terminal recordings and provider-visible runtime metadata
expose it, while recordings also expose submission times.

`axio-repl --version` exits before agent configuration, sandbox, or provider
initialization. Its local-only report identifies the distribution version,
resolved launcher, imported module source, Python interpreter, and the short Git
revision when the module is inside a checkout. Installed wheels without checkout
metadata report the revision as unavailable.

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
delivery, pending-input buffer/recall/claim/delivery transitions, interruption
barriers, editor snapshots, recovery application, agent shutdown, and clean
session shutdown. This preserves live streaming while making every state
transition needed for deterministic recovery an explicit durability boundary.

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

To resume, pass the stopped session's exact journal path:

```bash
axio-repl --resume ~/.local/state/axio/sessions/2026/08/14/<session>/events.jsonl
```

The new session replays main-agent context as distinct messages, restores
pending Enter submissions and the last durable editor snapshot, and
materializes available partial text, reasoning, tool arguments, and tool output
from unfinished main or background-agent turns. A background agent cannot be
recreated after process death, so its available partial output becomes a
labelled user notice in the restored main context. Cancelled deferred tools are
also materialized with their original agent, turn, and call IDs. The new session
writes `RecoveryApplied` records; use that new journal for a later resume.
Recovery is interactive-only and cannot be combined with `--no-session-log`.

Session directories use mode `0700` and journal or attachment files use `0600`.
Known secret-shaped fields and common token formats are redacted recursively,
but a secret embedded in arbitrary prose or tool output cannot always be
recognized. Treat journals as sensitive local data.

## REPL commands

Command output is shown in the temporary bottom panel and disappears when the
next editor submission is accepted. Commands entered during an unsafe active
turn are queued in the UI plane and applied at the next turn boundary; they are
not converted into pending user messages.

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
| `/agent-interrupt [id]` | Interrupt a background agent's current turn; defaults to the focused agent |
| `/agent-stop [id]` | Stop a background agent; defaults to the focused agent |
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

### Chronological arrivals and deferred tools

User submissions, peer messages, background outcomes, interruption events, and
deferred-tool results share one monotonic session order. A non-empty Enter
reserves its position before the prompt accepts another event, so a peer message
that arrives immediately afterwards cannot overtake it. Ordered batches are
appended as distinct `Message` objects; they are not joined into one text block.

An arrival cannot be inserted into a provider response that is already
streaming, so the coordinator exposes it at the earliest safe model boundary.
If the active turn is blocked in a foreground tool dispatch, the REPL requests
preemption instead of waiting for the tool to finish. The interrupted tool-use
protocol is closed with a placeholder saying that the call continues, while the
session retains ownership of the actual task. Its eventual result is delivered
once as a labelled user message in session order; it is never emitted as a
second `ToolResult` for the closed call. On process shutdown, unfinished
deferred calls are cancelled and their identities are preserved for recovery.

## Tools

| Tool | Description |
|---|---|
| `read_file` | Read file contents; images and videos are returned as vision blocks |
| `write_file` | Create or overwrite files with UTF-8 text |
| `patch_file` | Replace line ranges in UTF-8 text files (1-indexed, inclusive) |
| `list_files` | List directory contents |
| `search_files` | Text or regex search across files |
| `shell` | Run shell commands with streaming stdout/stderr |
| `generate_image` | Generate images via Gemini (Google transport only) |
| `generate_video` | Generate videos via Veo (Google transport only) |
| `list_peers` | List running local agents |
| `send_message` | Send a message to a local agent by global id |
| `run_agent` | Run one foreground child turn and return its answer |
| `spawn_agent` | Start a persistent background child agent |
| `monitor` | Wait inside a tool call for agents, tasks, paths, processes, or messages |
| `interrupt_agent` | Interrupt a background agent's current turn |
| `stop_agent` | Stop a background agent |

## Sandbox

`--sandbox` decides where those tools act. The default is `auto`: a container is
used whenever `aiodocker` is installed and `/var/run/docker.sock` exists, so a
machine with Docker running gets a sandbox without asking for one. The startup
banner states which it is — `Tools: docker — …` or `Tools: host — …`.

In a container the project is bind-mounted at the same absolute path used on
the host, and that is the path the system prompt gives the model. The container
runs with the invoking numeric UID/GID and supplementary groups. Read-only host
`/etc/passwd` and `/etc/group` mounts provide file-based name resolution, while
an automatically removed temporary bind mount provides a writable isolated
`HOME`; the real host home is not exposed.

The default image is the locally built `axio-agent-sandbox:standard`, and
networking is off. Run `make sandbox-image` before its first use; the REPL does
not try to pull that local-only tag. Explicit alternative image names retain
pull-on-missing behavior. Fail-closed support for an internal Docker network,
policy proxy, registry caches, and read-only dataset snapshots is described in
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

Bracketed multi-line paste is preserved in one `prompt_toolkit` editor value.
Pressing Enter submits the complete value as one user message; the REPL does not
split or rejoin it through a separate continuation-line protocol. Displayed
continuation and wrapping are controlled by the terminal and `prompt_toolkit`.
There is no continuation prefix, so each explicit continuation starts at column
zero; any indentation shown is part of the editor content:

```
12:41 username> Refactor this function:
def old(x):
    return x + 1
```

## Interactive input and interruption

Only **Enter** submits editor text. While an agent turn is running, each Enter
durably records a separate pending user message before opening a fresh editor.
The pending queue is bounded. At its limit the prompt remains active for Escape
and Up, and another submitted value is restored to the editor with a warning
instead of being dropped.

Press **Up** to take every user message that has not yet been claimed by an
agent back out of the pending buffer. Their text is placed in the editor in
submission order, with one empty line between the original messages. Sending
that edited text with Enter creates one new pending message. If no pending input
exists, Up falls back to normal prompt history or movement within a multi-line
editor value.

Press **Escape** to interrupt the turn that was active when the keypress was
accepted. Escape never submits, clears, or changes the editor. It claims every
user message previously submitted with Enter and delivers each as a separate
conversation message to the currently focused agent. Peer messages, background
outcomes, tool events, and the interruption notice retain their chronological
order around the interrupt barrier.

If no user message is pending, Escape is an interrupt only. Partial output and
queued events are committed to context, the editor stays unchanged, and no
replacement model turn starts.

A lone Escape is distinguished from an escape-prefixed key sequence with a
200 ms timeout. Escape is bound eagerly so interruption remains responsive;
standard Alt-key word-motion bindings such as Alt+B and Alt+F are therefore not
available in this prompt.

On an empty editor, the first **Ctrl-D** warns that exit is armed for two
seconds. A second Ctrl-D during that window closes the input source: no new
editor prompt is created, while the active turn, already submitted input, ready
claims, peer arrivals, and admission work drain in chronological order before
shutdown. With text in the editor, Ctrl-D keeps its normal forward-delete
behavior.

**Ctrl-C** starts graceful REPL shutdown and neither submits nor clears the
editor. Its current text is included in the shutdown snapshot and restored by
`--resume`. Use Escape to interrupt only the captured agent turn while keeping
the session running.
