"""Addressed, in-process notification bus.

A notification is text addressed to an owner — an agent, named by whatever id
the peer layer gives it, or ``None`` for an agent with no peer identity. Where
the text goes depends on what that owner is doing: an owner inside a turn picks
its notifications up with :func:`drain` at the top of its next iteration, while
an idle owner has them handed to its listener, which is what starts a new turn.

Delivery happens exactly once. Text handed to a listener leaves the queue, text
the owner already collected another way is removed by :func:`retract`, and the
queue is flushed to the listener when a turn ends so nothing waits for a turn
that may never come.

Everything here is synchronous and non-blocking, assumes a single event loop,
and is therefore safe to call from a task done-callback.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

NOTIFY_MAX_CHARS = 4000

_TRUNCATION_MARKER = "... [truncated]"

OwnerResolver = Callable[[], str | None]
Listener = Callable[[str], None]


@dataclass(slots=True)
class _Entry:
    text: str
    tag: str | None = None


@dataclass(slots=True)
class _Bucket:
    queue: list[_Entry] = field(default_factory=list)
    depth: int = 0
    listener: Listener | None = None

    @property
    def disposable(self) -> bool:
        return not self.queue and self.depth == 0 and self.listener is None


_buckets: dict[str | None, _Bucket] = {}
_resolver: OwnerResolver | None = None


def set_owner_resolver(resolver: OwnerResolver | None) -> None:
    """Install the callable naming the owner of the current execution context."""
    global _resolver
    _resolver = resolver


def current_owner() -> str | None:
    """Owner of the current execution context, ``None`` when nothing resolves it."""
    return _resolver() if _resolver is not None else None


def _bucket(owner: str | None) -> _Bucket:
    bucket = _buckets.get(owner)
    if bucket is None:
        bucket = _Bucket()
        _buckets[owner] = bucket
    return bucket


def _prune(owner: str | None) -> None:
    bucket = _buckets.get(owner)
    if bucket is not None and bucket.disposable:
        del _buckets[owner]


def _truncate(text: str) -> str:
    if len(text) <= NOTIFY_MAX_CHARS:
        return text
    return text[: NOTIFY_MAX_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def post(text: str, owner: str | None, *, tag: str | None = None) -> None:
    """Send *text* to *owner*: to its running turn, its listener, or its queue.

    A *tag* names the notification so :func:`retract` can take it back while it
    is still queued — used when the owner collects the same result by hand.
    """
    entry = _Entry(text=_truncate(text), tag=tag)
    bucket = _buckets.get(owner)
    if bucket is not None and bucket.depth == 0 and bucket.listener is not None:
        bucket.listener(entry.text)
        return
    _bucket(owner).queue.append(entry)


def retract(owner: str | None, tag: str) -> None:
    """Drop still-queued notifications carrying *tag*; delivered ones stay delivered."""
    bucket = _buckets.get(owner)
    if bucket is None:
        return
    bucket.queue[:] = [entry for entry in bucket.queue if entry.tag != tag]
    _prune(owner)


def drain(owner: str | None) -> list[str]:
    """Take everything queued for *owner*, in arrival order."""
    bucket = _buckets.get(owner)
    if bucket is None:
        return []
    texts = [entry.text for entry in bucket.queue]
    bucket.queue.clear()
    _prune(owner)
    return texts


def add_listener(owner: str | None, cb: Listener) -> None:
    """Route notifications for an idle *owner* to *cb*, which must not block."""
    _bucket(owner).listener = cb


def remove_listener(owner: str | None) -> None:
    bucket = _buckets.get(owner)
    if bucket is None:
        return
    bucket.listener = None
    _prune(owner)


@contextmanager
def turn_scope(owner: str | None) -> Iterator[None]:
    """Mark *owner* as working, so notifications wait for its next iteration.

    Nesting is counted, and leaving the outermost scope flushes whatever the
    turn did not pick up to the listener: the turn is over, nothing else will
    drain it.
    """
    _bucket(owner).depth += 1
    try:
        yield
    finally:
        bucket = _buckets.get(owner)
        if bucket is not None:
            bucket.depth = max(bucket.depth - 1, 0)
            if bucket.depth == 0:
                if bucket.queue and bucket.listener is not None:
                    pending, bucket.queue = bucket.queue, []
                    for entry in pending:
                        bucket.listener(entry.text)
                _prune(owner)


def discard(owner: str | None) -> None:
    """Forget *owner* entirely: queue, turn mark and listener. For dead owners."""
    _buckets.pop(owner, None)
