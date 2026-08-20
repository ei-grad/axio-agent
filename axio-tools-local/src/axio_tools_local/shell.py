from __future__ import annotations

import asyncio
import codecs
import contextlib
import os
import re
import shutil
import signal
import tempfile
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TextIO

from axio.exceptions import HandlerError
from axio.field import Field, StrictStr

type OutputRecord = tuple[float, str, str]

_MAX_INLINE_OUTPUT_CHARS = 4096
_INLINE_HEAD_LINES = 5
_INLINE_TAIL_LINES = 5
_PREFIX_TRUNCATION_MARKER = "\n...[truncated]\n"
_SUFFIX_TRUNCATION_MARKER = "[truncated]...\n"
_LIMIT_RE = re.compile(r"within (?P<limit>\d+) chars")
_PIPE_READ_BYTES = 4096
_PIPE_QUEUE_ITEMS = 32
_SHELL_PREFERENCE = ("bash", "sh", "zsh", "dash")


class _LargeOutputNotice(str):
    pass


class _ShellControl(str):
    pass


@dataclass(frozen=True, slots=True)
class _ReaderDone:
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class _ShellExecutable:
    name: str
    path: str


def _discover_shells(search_path: str | None) -> tuple[_ShellExecutable, ...]:
    available: list[_ShellExecutable] = []
    for name in _SHELL_PREFERENCE:
        executable = shutil.which(name, path=search_path)
        if executable is not None:
            available.append(_ShellExecutable(name=name, path=os.path.abspath(executable)))
    return tuple(available)


def _shell_choices_text(available: tuple[_ShellExecutable, ...]) -> str:
    if not available:
        return "No supported shells were found on PATH."
    names = ", ".join(item.name for item in available)
    return f"Available shells: {names}. Omit shell to use {available[0].name}."


def _select_shell(requested: str | None, available: tuple[_ShellExecutable, ...]) -> _ShellExecutable:
    if not available:
        tried = ", ".join(_SHELL_PREFERENCE)
        raise HandlerError(f"No supported shell found on PATH (tried: {tried})")
    if requested is None:
        return available[0]
    for item in available:
        if requested == item.name:
            return item
    choices = ", ".join(item.name for item in available)
    raise HandlerError(f"Unknown shell {requested!r}; available shells: {choices}")


_AVAILABLE_SHELLS = _discover_shells(os.environ.get("PATH"))
_ShellName = Annotated[
    str | None,
    Field(description=f"Shell name to execute. {_shell_choices_text(_AVAILABLE_SHELLS)}"),
]


def _kill_process(proc: asyncio.subprocess.Process) -> None:
    """Kill the process and its entire process group."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()


def _format_timestamp(ts: float) -> str:
    mins, secs = divmod(ts, 60)
    return f"{int(mins):02d}:{secs:06.3f}"


def _open_output_log(cwd: str, command: str, max_output_chars: int) -> tuple[TextIO, str]:
    output_dir = Path(cwd).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir = output_dir.resolve()

    try:
        fd, path = tempfile.mkstemp(
            prefix=".axio-shell-output-",
            suffix=".log",
            dir=output_dir,
        )
    except OSError:
        fd, path = tempfile.mkstemp(
            prefix="axio-shell-output-",
            suffix=".log",
        )

    f = os.fdopen(fd, "w", encoding="utf-8")
    f.write(f"command: {command}\n")
    f.write(f"cwd: {output_dir}\n")
    f.write(f"inline output cap: {max_output_chars} chars\n\n")
    f.write(f"inline output summary: first {_INLINE_HEAD_LINES} and last {_INLINE_TAIL_LINES} lines\n\n")
    return f, path


def _write_output_record(f: TextIO, ts: float, key: str, text: str) -> None:
    f.write(f"[{_format_timestamp(ts)} {key}] {text}")
    if not text.endswith("\n"):
        f.write("\n")


def _text_len(records: list[OutputRecord]) -> int:
    return sum(len(text) for _, _, text in records)


def _clip_prefix(record: OutputRecord, limit: int) -> OutputRecord | None:
    if limit <= 0:
        return None
    ts, key, text = record
    if len(text) <= limit:
        return record
    if limit <= len(_PREFIX_TRUNCATION_MARKER):
        return (ts, key, text[:limit])
    return (ts, key, text[: limit - len(_PREFIX_TRUNCATION_MARKER)].rstrip("\n") + _PREFIX_TRUNCATION_MARKER)


def _clip_suffix(record: OutputRecord, limit: int) -> OutputRecord | None:
    if limit <= 0:
        return None
    ts, key, text = record
    if len(text) <= limit:
        return record
    if limit <= len(_SUFFIX_TRUNCATION_MARKER):
        return (ts, key, text[-limit:])
    return (ts, key, _SUFFIX_TRUNCATION_MARKER + text[-(limit - len(_SUFFIX_TRUNCATION_MARKER)) :].lstrip("\n"))


def _take_prefix(records: list[OutputRecord], limit: int) -> list[OutputRecord]:
    result: list[OutputRecord] = []
    remaining = limit
    for record in records:
        clipped = _clip_prefix(record, remaining)
        if clipped is None:
            break
        result.append(clipped)
        remaining -= len(clipped[2])
    return result


def _take_suffix(records: list[OutputRecord], limit: int) -> list[OutputRecord]:
    result: list[OutputRecord] = []
    remaining = limit
    for record in reversed(records):
        clipped = _clip_suffix(record, remaining)
        if clipped is None:
            break
        result.append(clipped)
        remaining -= len(clipped[2])
    result.reverse()
    return result


def _fit_head_tail_to_budget(
    head_records: list[OutputRecord],
    tail_records: list[OutputRecord],
    limit: int,
) -> tuple[list[OutputRecord], list[OutputRecord]]:
    if limit <= 0:
        return [], []
    if _text_len(head_records) + _text_len(tail_records) <= limit:
        return head_records, tail_records

    head_budget = limit // 2
    tail_budget = limit - head_budget
    head_need = _text_len(head_records)
    tail_need = _text_len(tail_records)

    if head_need < head_budget:
        tail_budget += head_budget - head_need
        head_budget = head_need
    if tail_need < tail_budget:
        head_budget += tail_budget - tail_need
        tail_budget = tail_need

    return _take_prefix(head_records, head_budget), _take_suffix(tail_records, tail_budget)


def _format_records_plain(records: list[OutputRecord]) -> str:
    """Merge consecutive same-stream records within 0.5s into log entries.

    Produces structured output so the model sees stdout vs stderr with
    timing: ``[00:01.234 stderr] something went wrong``.
    """
    if not records:
        return "(no output)"

    # (first_ts, last_ts, key, accumulated_text)
    merged: list[tuple[float, float, str, str]] = []
    for ts, key, text in records:
        if merged and merged[-1][2] == key and (ts - merged[-1][1]) <= 0.5:
            prev = merged[-1]
            merged[-1] = (prev[0], ts, key, prev[3] + text)
        else:
            merged.append((ts, ts, key, text))

    lines: list[str] = []
    for first_ts, _, key, text in merged:
        header = f"[{_format_timestamp(first_ts)} {key}]"
        lines.append(f"{header} {text.rstrip(chr(10))}")
    return "\n".join(lines)


def _is_large_output_notice(record: OutputRecord) -> bool:
    return isinstance(record[2], _LargeOutputNotice)


def _is_control_record(record: OutputRecord) -> bool:
    return isinstance(record[2], _ShellControl)


def _notice_limit(record: OutputRecord) -> int:
    match = _LIMIT_RE.search(record[2])
    return int(match.group("limit")) if match is not None else _MAX_INLINE_OUTPUT_CHARS


def _format_large_records(records: list[OutputRecord], notice_index: int) -> str:
    notice = records[notice_index]
    limit = _notice_limit(notice)
    output_records = [
        record for i, record in enumerate(records) if i != notice_index and not _is_control_record(record)
    ]
    control_records = [record for i, record in enumerate(records) if i != notice_index and _is_control_record(record)]

    head_records = output_records[:_INLINE_HEAD_LINES]
    tail_start = max(len(output_records) - _INLINE_TAIL_LINES, len(head_records))
    tail_records = output_records[tail_start:]

    raw_budget = limit
    while True:
        visible_head, visible_tail = _fit_head_tail_to_budget(head_records, tail_records, raw_budget)
        result = _format_records_plain([*visible_head, notice, *visible_tail, *control_records])
        overflow = len(result) - limit
        if overflow <= 0:
            return result
        if raw_budget <= 0:
            return result[:limit]
        raw_budget = max(0, raw_budget - overflow)


def _format_records(records: list[OutputRecord]) -> str:
    for i, record in enumerate(records):
        if _is_large_output_notice(record):
            return _format_large_records(records, i)
    return _format_records_plain(records)


def _split_decoded_output(text: str) -> list[str]:
    """Split decoded reads at newlines without withholding a partial tail."""
    chunks: list[str] = []
    start = 0
    while (newline := text.find("\n", start)) >= 0:
        chunks.append(text[start : newline + 1])
        start = newline + 1
    if start < len(text):
        chunks.append(text[start:])
    return chunks


async def _shell_stream(
    command: str,
    timeout: int = 5,
    cwd: str = ".",
    stdin: str | None = None,
    max_output_chars: int = _MAX_INLINE_OUTPUT_CHARS,
    shell: _ShellName = None,
) -> AsyncGenerator[tuple[str, str], None]:
    """Yield tagged chunks in the order accepted from the two pipe readers.

    This is the executor's observed order, not a global ordering of writes to
    the process's independent stdout and stderr file descriptors.
    """
    if max_output_chars < 0:
        raise HandlerError("max_output_chars must be >= 0")

    executable = _select_shell(shell, _AVAILABLE_SHELLS)
    try:
        proc = await asyncio.create_subprocess_exec(
            executable.path,
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise HandlerError(f"Failed to start shell command: {exc}") from exc

    assert proc.stdout is not None
    assert proc.stderr is not None

    queue: asyncio.Queue[tuple[str, str] | _ReaderDone] = asyncio.Queue(maxsize=_PIPE_QUEUE_ITEMS)

    async def _read_pipe(pipe: asyncio.StreamReader, key: str) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while data := await pipe.read(_PIPE_READ_BYTES):
                text = decoder.decode(data)
                if text:
                    await queue.put((key, text))
            tail = decoder.decode(b"", final=True)
            if tail:
                await queue.put((key, tail))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(_ReaderDone(exc))
        else:
            await queue.put(_ReaderDone())

    stdout_task = asyncio.create_task(_read_pipe(proc.stdout, "stdout"))
    stderr_task = asyncio.create_task(_read_pipe(proc.stderr, "stderr"))
    reader_tasks = (stdout_task, stderr_task)

    timed_out = False
    output_chars = 0
    output_record_count = 0
    yielded_output_count = 0
    streamed_output_chars = 0
    large_output = False
    notice_emitted = False
    buffered_records: list[OutputRecord] = []
    tail_records: list[OutputRecord] = []
    output_log: TextIO | None = None
    output_log_path: str | None = None
    cleanup_complete = False

    async def _stop_process_and_readers() -> None:
        nonlocal cleanup_complete
        if cleanup_complete:
            return
        if proc.returncode is None or any(not task.done() for task in reader_tasks):
            _kill_process(proc)
        for task in reader_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*reader_tasks, return_exceptions=True)
        await proc.wait()
        cleanup_complete = True

    if stdin is not None:
        try:
            assert proc.stdin is not None
            proc.stdin.write(stdin.encode())
            await proc.stdin.drain()
            proc.stdin.close()
        except asyncio.CancelledError:
            await _stop_process_and_readers()
            raise
        except Exception:
            await _stop_process_and_readers()
            raise

    deadline = time.monotonic() + timeout
    t0 = time.monotonic()

    def _elapsed() -> float:
        return time.monotonic() - t0

    def _ensure_output_log() -> str:
        nonlocal output_log, output_log_path
        if output_log is None:
            output_log, output_log_path = _open_output_log(cwd, command, max_output_chars)
            for ts, record_key, record_text in buffered_records:
                _write_output_record(output_log, ts, record_key, record_text)
            buffered_records.clear()
        assert output_log_path is not None
        return output_log_path

    def _append_tail(record: OutputRecord) -> None:
        tail_records.append(record)
        if len(tail_records) > _INLINE_TAIL_LINES:
            del tail_records[0]

    def _large_output_notice() -> OutputRecord:
        path = _ensure_output_log()
        text = _LargeOutputNotice(
            f"[output is too large, saved to {path}; showing first {_INLINE_HEAD_LINES} "
            f"and last {_INLINE_TAIL_LINES} lines within {max_output_chars} chars because output "
            f"exceeded {max_output_chars} chars]\n"
        )
        return (_elapsed(), "stderr", text)

    def _emit_large_output_notice() -> list[tuple[str, str]]:
        nonlocal notice_emitted
        if notice_emitted:
            return []
        notice_emitted = True
        notice = _large_output_notice()
        if output_log is not None:
            _write_output_record(output_log, notice[0], notice[1], notice[2])
        return [(notice[1], notice[2])]

    def _record_output(key: str, text: str) -> list[tuple[str, str]]:
        nonlocal large_output, output_chars, output_record_count, streamed_output_chars, yielded_output_count
        ts = _elapsed()
        record = (ts, key, text)
        output_chars += len(text)
        output_record_count += 1
        _append_tail(record)

        if not large_output:
            buffered_records.append((ts, key, text))
            if output_chars > max_output_chars:
                large_output = True
                _ensure_output_log()
                buffered_records.clear()
                chunks: list[tuple[str, str]] = []
                if output_record_count <= _INLINE_HEAD_LINES:
                    remaining = max_output_chars - streamed_output_chars
                    clipped = _clip_prefix(record, remaining)
                    if clipped is not None:
                        yielded_output_count += 1
                        streamed_output_chars += len(clipped[2])
                        chunks.append((clipped[1], clipped[2]))
                chunks.extend(_emit_large_output_notice())
                return chunks

            yielded_output_count += 1
            streamed_output_chars += len(text)
            return [(key, text)]

        assert output_log is not None
        _write_output_record(output_log, ts, key, text)
        return []

    def _record_control(key: str, text: str) -> OutputRecord:
        record = (_elapsed(), key, _ShellControl(text))
        if output_log is not None:
            _write_output_record(output_log, record[0], key, text)
        return record

    def _pending_records() -> list[OutputRecord]:
        if not large_output:
            return buffered_records[yielded_output_count:]

        overlap = max(0, _INLINE_HEAD_LINES + len(tail_records) - output_record_count)
        visible_tail = tail_records[overlap:] if overlap else tail_records
        return _take_suffix(visible_tail, max_output_chars)

    try:
        done_count = 0
        while done_count < 2:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=remaining)
            except TimeoutError:
                timed_out = True
                break
            if isinstance(item, _ReaderDone):
                done_count += 1
                if item.error is not None:
                    raise item.error
            else:
                key, text = item
                for record_text in _split_decoded_output(text):
                    for chunk in _record_output(key, record_text):
                        yield chunk

        if timed_out:
            await _stop_process_and_readers()
            while True:
                try:
                    accepted = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(accepted, _ReaderDone):
                    continue
                key, text = accepted
                for record_text in _split_decoded_output(text):
                    for chunk in _record_output(key, record_text):
                        yield chunk

            timeout_control_records = [_record_control("stderr", f"[timeout: command exceeded {timeout}s]")]
            for _, key, text in [*_pending_records(), *timeout_control_records]:
                yield (key, text)
            return

        await asyncio.gather(*reader_tasks)
        returncode = await proc.wait()
        cleanup_complete = True
        control_records: list[OutputRecord] = []
        if returncode != 0:
            control_records.append(_record_control("stderr", f"[exit code: {returncode}]"))
        for _, key, text in [*_pending_records(), *control_records]:
            yield (key, text)
    finally:
        await _stop_process_and_readers()
        if output_log is not None:
            output_log.close()


async def shell(
    command: StrictStr,
    timeout: int = 5,
    cwd: StrictStr = ".",
    stdin: str | None = None,
    max_output_chars: int = _MAX_INLINE_OUTPUT_CHARS,
    shell: _ShellName = None,
) -> str:
    """Run a shell command and return combined stdout/stderr. Use for git,
    build tools, grep, tests, or any CLI operation. Non-zero exit codes
    are reported. Optionally pass stdin data for commands that read from
    standard input. The default timeout is 5s — raise it for long-running
    commands (installs, builds, and test suites often need 60-300s). Output
    larger than max_output_chars is saved to a local log file; inline output is
    summarized to the first 5 and last 5 lines.
    Live and final output preserve executor-observed stdout/stderr order; this
    does not imply a global syscall order across the two descriptors. Avoid
    interactive commands. Commands default to the first shell discovered on
    PATH (bash is preferred); pass shell to select another advertised shell.
    Commands use non-login ``-c`` mode and do not load login profiles."""
    records: list[OutputRecord] = []
    t0 = time.monotonic()
    async for key, text in _shell_stream(command, timeout, cwd, stdin, max_output_chars, shell):
        records.append((time.monotonic() - t0, key, text))
    return _format_records(records)


# Streaming hooks consumed by axio.tool.Tool.call_streaming / format_stream_result.
shell.stream = _shell_stream  # type: ignore[attr-defined]
shell.format_stream_result = staticmethod(_format_records)  # type: ignore[attr-defined]
