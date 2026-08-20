# TODO

Open design questions. Each entry states the problem and what makes it
non-obvious, not a chosen solution.

## Host identity passthrough into the sandbox

An agent working in a bind-mounted project reaches the point where it needs the
host's identity to finish the job. `git commit` is the first case: the container
has no `user.name`/`user.email`, `git` refuses, and the agent is left to invent
something. Guessing the identity from the last commit's author - which is what
happens in practice - fabricates authorship.

What is unclear is the boundary. Committing needs a name and an address, pushing
needs an SSH agent or a token, signing needs a key. Passing all of it through
turns the sandbox into a thin wrapper around the host account, and a mounted SSH
agent socket is a general-purpose credential the agent can use for anything.
Passing none of it means every commit either fails or is attributed to a
placeholder.

Points to settle:

- Which identity is passed - the host's git identity, a dedicated agent identity,
  or one declared per agent bundle.
- Whether the agent commits at all, or only stages and leaves the commit to the
  host side of the boundary.
- How agent authorship is recorded. A `Co-Authored-By` trailer naming the model
  is the cheap option and needs the transport and model id, which the REPL knows.
- Whether a credential is mounted for the whole session or brokered per
  operation.

## Split the session journal into a semantic log and a replay log

One JSONL file currently carries both the conversation record and the raw
streaming deltas. A 14-minute session produces ~8 MB, roughly three quarters of
it reasoning deltas. The file is simultaneously too noisy to read as a
transcript and too lossy to replay as a session: it has no keystrokes and no
frame timing.

Two artifacts with different jobs:

- A semantic JSONL comparable to what Codex and Claude Code write: user
  messages, committed context, tool calls and results, configuration changes,
  lifecycle. Readable, greppable, diffable, and the input for `--resume`.
- A compressed binary replay log with an explicit schema, in the spirit of
  asciinema, for reproducing a session as it was experienced: terminal frames
  with timing, user keystrokes, and granular input events from any frontend, not
  just the terminal one. Useful as a regression fixture for interface work,
  where the current journal cannot help because it never recorded the input.

Points to settle: whether the semantic log is derived from the replay log or
written independently; the schema and its versioning; retention, given that the
replay log records keystrokes and therefore anything typed, including secrets
the redactor cannot recognize.
