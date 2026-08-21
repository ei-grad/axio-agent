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
