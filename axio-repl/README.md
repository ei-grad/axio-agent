# axio-repl

Interactive REPL coding assistant powered by the [axio](../axio) agent framework.
Works with any LLM backend via pluggable transports — bring your own API key.

## Philosophy

axio-repl is an opinionated terminal agent that **actually verifies its work**.
The system prompt encodes hard-won lessons from watching models cut corners:

- **Stream everything, hide nothing.** Every piece of information shown to the
  user — tool arguments, stdout/stderr, images, exit codes — must also be
  faithfully presented to the model. The model should see exactly what the user
  sees, so it can reason about the same reality. No summarizing, no truncating,
  no dropping context between what's displayed and what's sent back.
- **Not tested — not done.** The agent must run tests, re-read edited files,
  and observe actual results instead of assuming success from exit codes.
- **Iterative UI review.** When building or modifying UI, the agent captures
  real screenshots at multiple viewport sizes (desktop, tablet, mobile) via
  Playwright/Puppeteer, reads every screenshot through the model's vision, and
  critically lists visual defects. It repeats the screenshot → fix → re-screenshot
  loop until zero defects — no premature "looks good".
- **Ground everything in project context.** Read before editing. List the
  directory before guessing. Never refuse a safe request.
- **Minimal edits.** Don't reformat surrounding code, don't narrate tool calls.
  The user sees the full tool output in the terminal.

## Features

- **Pluggable transports** — auto-detected from API keys via
  `axio.transport` entry points. Ships with support for OpenAI, Anthropic,
  Google (Gemini API & Vertex AI), Nebius, OpenRouter, and Codex.
- **Runtime model switching** — `/model <query>` to switch models mid-session
  without restarting. Capabilities (vision, reasoning, image generation) are
  re-evaluated and the system prompt adapts automatically.
- **Streaming tool arguments** — tool call fields appear incrementally as the
  model generates them, so you see what's happening before execution starts.
- **Streaming tool output** — shell command stdout/stderr streams line-by-line
  in real time instead of buffering until completion.
- **Vision** — `read_file` on images (PNG, JPG, GIF, WebP) and videos returns
  multimodal content blocks. The model sees the actual pixels, not a description.
- **Image & video generation** — when the Google transport is installed,
  `generate_image` and `generate_video` tools are available for Gemini Nano
  Banana / Veo models.
- **AGENTS.md** — workspace-level instructions loaded into the system prompt from
  an `AGENTS.md` file in the working directory.
- **Multiline paste** — pasting multi-line text into the prompt is handled
  gracefully with continuation markers (`...`).
- **Graceful interruption** — Ctrl-C cancels the running agent loop, preserving
  partial tool output in conversation context so the model knows what happened.
- **Readline history** — persisted across sessions in `~/.axio_repl_history`.
- **Session journals** — every main, foreground, and background agent event is
  written to a private JSONL journal for later inspection.
- **Single-prompt mode** — pass a prompt as argument for scripting and non-interactive use.

## Install

```bash
uv tool install axio-repl
```

To add optional transports:

```bash
uv tool install axio-repl --with axio-transport-anthropic
uv tool install axio-repl --with axio-transport-google
```

Or within the monorepo workspace:

```bash
uv run axio-repl
```

## Usage

```bash
# Interactive REPL (auto-detects transport from API keys)
axio-repl

# Single prompt (non-interactive)
axio-repl "list the files in this project"

# Explicit transport and model
axio-repl --transport anthropic --model claude-sonnet-4-20250514

# Google Gemini
axio-repl --transport google --model gemini-3.1-pro-preview

# Custom temperature and iteration limit
axio-repl --temperature 0.5 --max-iterations 100

# Choose another journal root, or explicitly opt out
axio-repl --session-log-dir ./axio-session-logs
axio-repl --no-session-log

# Show framed tool and lifecycle actions from background agents
axio-repl --agent-actions on
```

## Session journals and privacy

Session journaling is enabled by default. Each invocation creates one journal at:

```text
${XDG_STATE_HOME:-~/.local/state}/axio/sessions/YYYY/MM/DD/<session-id>/events.jsonl
```

The REPL prints the exact path when the session starts. In single-prompt mode it
prints the path to stderr, leaving the streamed answer on stdout. Use
`--session-log-dir <directory>` to select another root or `--no-session-log` to
disable journaling.

The append-only journal contains user input, model stream events, committed
context messages, configuration changes, agent lifecycle events, subagent
output, and outcome-delivery correlation. Binary media is stored by hash under
the session's `attachments/` directory. Directories are created with mode
`0700` and files with `0600`.

The initial `session_start` record is fsynced before the REPL starts normal
session work. Streaming records are admitted to a bounded memory queue without
an fsync per token. Successfully committed context mutations, completed turns,
outcome deliveries, and stopped agents are durability boundaries: each drains
and fsyncs every earlier accepted record. A clean shutdown also drains the queue,
writes `session_end`, and fsyncs it.

After abrupt process termination, all records through the last successful
durability boundary are retained. Records accepted after that
boundary are explicitly best-effort: all of them may be lost, or a prefix may
be present. Any present data remains valid newline-delimited JSON except for at
most one final unterminated line interrupted during `write(2)`. Journal readers
may ignore that final tail; `recover_journal_tail()` validates the earlier
prefix before truncating it. Corruption in a newline-terminated or non-final
record is not treated as a crash tail.

Storage-media corruption and filesystem failures outside this interrupted-tail
model are reported as corruption; the reader does not silently skip them.

Known secret-shaped fields and common token formats are redacted before writing,
but arbitrary secrets embedded in prose or tool output cannot be identified
reliably. Treat the journal as sensitive local session data and apply an
appropriate retention policy.

## REPL Commands

| Command              | Description                                    |
|----------------------|------------------------------------------------|
| `/help`              | Show available tools and commands               |
| `/agent-actions`     | Show whether other agents' actions are visible  |
| `/agent-actions on\|off` | Toggle framed actions from other agents    |
| `/agents`            | List local background agents                    |
| `/agent-focus <id>`  | Change the input target                         |
| `/model`             | Show current model and list available models    |
| `/model <query>`     | Switch to a model matching the query            |
| `/quit` `/exit` `/q` | Exit the REPL                                   |

## Foreground and background agents

`run_agent` executes one bounded child turn in the foreground. Its reasoning,
text, tool arguments, and streaming tool output use the same immediate terminal
path as the parent. The input target does not change, and the child's final text
is returned to the parent exactly once as the `run_agent` tool result.
If another parent tool streams concurrently with the foreground child, its
labelled output is inserted at the child's next safe boundary. This active
parent work remains visible even when agent actions are off, without splitting
the child's paragraph, reasoning, arguments, or tool output.

`spawn_agent` starts a persistent background peer. By default, its prose and
actions stay out of the active stream and the REPL prints a completion summary.
Use `/agent-actions on` (or start with `--agent-actions on`) to show its complete
tool calls, line-grouped tool output, results, errors, and lifecycle changes.
These labelled frames appear only after a complete active paragraph, after
reasoning closes, after media, or after every active parallel tool call has
finished. Background prose and reasoning remain hidden.

The action toggle changes terminal presentation only. It neither changes which
agent receives input nor affects execution, outcome delivery, context, or the
session journal. Queues are bounded; overload is represented by a labelled
suppression frame rather than delaying the active stream. Enabling the mode does
not replay actions that occurred while it was off.

In single-prompt mode, the REPL waits for spawned background agents and displays
each final report through normal incoming-outcome delivery. It does not focus a
background agent or replay that agent's hidden prose before displaying the
report.

## Tools

| Tool            | Description                                                    |
|-----------------|----------------------------------------------------------------|
| `read_file`     | Read file contents; images and videos returned as vision blocks |
| `write_file`    | Create or overwrite files                                       |
| `patch_file`    | Replace line ranges in files (1-indexed, inclusive)              |
| `list_files`    | List directory contents                                         |
| `search_files`  | Text/regex search across files                                  |
| `shell`         | Run shell commands with streaming output and process-group cleanup |
| `generate_image` | Generate images via Gemini Nano Banana (Google transport only) |
| `generate_video` | Generate videos via Veo (Google transport only)                |

## Transports

Transports are discovered via the `axio.transport` entry point group.
The REPL picks the first transport whose required environment variable is set:

| Transport        | Env Variable                   | Package                    |
|------------------|--------------------------------|----------------------------|
| `google`         | `GEMINI_API_KEY`               | `axio-transport-google`    |
| `google-vertex`  | `GOOGLE_GENAI_USE_VERTEXAI`    | `axio-transport-google`    |
| `anthropic`      | `ANTHROPIC_API_KEY`            | `axio-transport-anthropic` |
| `openai`         | `OPENAI_API_KEY`               | `axio-transport-openai`    |
| `nebius`         | `NEBIUS_API_KEY`               | `axio-transport-openai`    |
| `openrouter`     | `OPENROUTER_API_KEY`           | `axio-transport-openai`    |
| `codex`          | *(API key varies)*             | `axio-transport-codex`     |

Use `--transport <name>` to force a specific transport regardless of env vars.

## Capability-Aware System Prompt

The system prompt adapts based on the selected model's declared capabilities:

- **Vision** — unlocks instructions to `read_file` images and do screenshot-based
  UI review.
- **Reasoning** — notes that extended thinking is available.
- **Image generation** — enables inline image generation guidance.
- **Video** — enables `read_file` for video content.
- **Tool use** — gates all tool-related rules (edit workflow, testing, verification).

This means switching from a text-only model to a vision model mid-session
(via `/model`) automatically updates what the agent is instructed to do.

## Architecture

```
┌─────────────┐     ┌───────────┐     ┌──────────────────┐
│  axio-repl  │────▶│   axio    │────▶│    transport      │
│  (terminal  │     │  (agent   │     │  (anthropic /     │
│   UI, I/O)  │     │   loop)   │     │   google / openai │
└─────────────┘     └───────────┘     │   / nebius / ...)  │
                          │           └──────────────────┘
                    ┌─────┴─────┐
                    │ tools     │
                    │ (local fs │
                    │  + shell) │
                    └───────────┘
```

- **axio-repl** owns the terminal UI: readline, event rendering, REPL commands.
- **axio** runs the agent loop: dispatch tools, manage conversation context,
  handle cancellation.
- **transports** handle LLM communication — message conversion, streaming,
  model registries.
- **axio-tools-local** provides the file and shell tools.
