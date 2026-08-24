"""Materialize resumable REPL state from one durable journal prefix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from axio.blocks import (
    AudioBlock,
    AudioMediaType,
    ContentBlock,
    ImageBlock,
    ImageMediaType,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
    VideoMediaType,
)
from axio.messages import InputProvenance, Message
from axio.tool_codec import (
    ToolArgumentCodecError,
    decode_framed_values,
    sanitize_presentation_value,
)

from axio_repl._journal import LEGACY_SCHEMA_VERSION, SCHEMA_VERSION, read_journal


class RecoveryError(ValueError):
    """A valid JSON journal cannot be materialized into canonical state."""


@dataclass(frozen=True, slots=True)
class RecoveredPendingInput:
    source_id: str
    text: str
    target_agent_id: str
    submitted_at: datetime | None = None
    author: str = "human"

    def __post_init__(self) -> None:
        if not self.source_id or not self.text or not self.target_agent_id:
            raise RecoveryError("recovered pending input fields must not be empty")
        if self.submitted_at is not None and (
            self.submitted_at.tzinfo is None or self.submitted_at.utcoffset() is None
        ):
            raise RecoveryError("recovered pending input submitted_at must be timezone-aware")
        if not self.author:
            raise RecoveryError("recovered pending input author must not be empty")


@dataclass(frozen=True, slots=True)
class RecoveryMaterialization:
    source_session_id: str
    messages: tuple[Message, ...]
    pending_inputs: tuple[RecoveredPendingInput, ...]
    editor_text: str
    recovery_ids: tuple[str, ...]
    discarded_tail_bytes: int


@dataclass(slots=True)
class _TurnFragments:
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_names: dict[str, str] = field(default_factory=dict)
    tool_argument_codecs: dict[str, str] = field(default_factory=dict)
    tool_arguments: dict[str, list[str]] = field(default_factory=dict)
    tool_fields: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    tool_output: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _DeferredRecovery:
    tool_use_id: str
    agent_id: str
    turn_id: str | None
    phase: str


def materialize_recovery(events_path: Path) -> RecoveryMaterialization:
    result = read_journal(events_path)
    records = result.records
    if not records:
        raise RecoveryError("session journal is empty")
    source_session_id = _string(records[0].get("session_id"), "session_id")
    source_schema = records[0].get("schema_version")
    if type(source_schema) is not int or source_schema not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        raise RecoveryError("unsupported journal schema at storage sequence 1")
    messages: list[Message] = []
    pending: dict[str, RecoveredPendingInput] = {}
    input_status: dict[str, str] = {}
    editor_text = ""
    turns: dict[str, _TurnFragments] = {}
    turn_reasons: dict[str, str] = {}
    committed_assistant_text: dict[str, list[str]] = {}
    committed_interrupt_notices: set[str] = set()
    child_turns: dict[tuple[str, str], _TurnFragments] = {}
    child_turn_reasons: dict[tuple[str, str], str] = {}
    child_committed_assistant_text: dict[tuple[str, str], list[str]] = {}
    deferred_tools: list[_DeferredRecovery] = []
    applied: set[str] = set()

    for expected_seq, record in enumerate(records, start=1):
        if record.get("schema_version") != source_schema:
            raise RecoveryError(f"unsupported journal schema at storage sequence {expected_seq}")
        record_seq = record.get("seq")
        if type(record_seq) is not int or record_seq != expected_seq:
            raise RecoveryError(f"non-contiguous journal storage sequence at record {expected_seq}")
        if record.get("session_id") != source_session_id:
            raise RecoveryError(f"session_id changed at storage sequence {expected_seq}")
        kind = _string(record.get("kind"), "kind")
        turn_id_value = record.get("turn_id")
        turn_id = turn_id_value if isinstance(turn_id_value, str) and turn_id_value else None

        if kind == "message_committed":
            record_agent_id = _string(record.get("agent_id"), "message_committed.agent_id")
            payload = _mapping(record.get("payload"), "message_committed.payload")
            message = _decode_message(_mapping(payload.get("message"), "message"), events_path.parent)
            source_input_id = payload.get("source_input_id")
            if source_input_id is not None:
                if message.role != "user":
                    raise RecoveryError("message commit with source_input_id is not a user message")
                _confirm_claimed_input_commit(
                    pending,
                    input_status,
                    target_agent_id=record_agent_id,
                    source_input_id=_string(source_input_id, "message_committed.source_input_id"),
                    message_text=_message_text(message),
                )
            if record_agent_id == "main":
                messages.append(message)
                if turn_id is not None and message.role == "assistant":
                    committed_assistant_text.setdefault(turn_id, []).append(_message_text(message))
                if message.role == "user":
                    committed_interrupt_notices.update(_notice_turn_ids(message))
            elif turn_id is not None and message.role == "assistant":
                child_committed_assistant_text.setdefault((record_agent_id, turn_id), []).append(
                    _message_text(message)
                )
            continue
        if kind == "turn_checkpoint":
            record_agent_id = _string(record.get("agent_id"), "turn_checkpoint.agent_id")
            if turn_id is None:
                raise RecoveryError("turn_checkpoint has no turn_id")
            payload = _mapping(record.get("payload"), "turn_checkpoint.payload")
            if record_agent_id == "main":
                fragments = turns.get(turn_id)
            else:
                fragments = child_turns.get((record_agent_id, turn_id))
            if fragments is not None:
                _apply_turn_checkpoint(fragments, payload)
            continue
        if kind not in {
            "input_buffered",
            "input_recalled",
            "input_claimed",
            "input_delivered",
            "editor_snapshot",
            "recovery_applied",
            "interruption_committed",
            "shutdown_recorded",
            "turn_started",
            "turn_finished",
            "stream_event",
        }:
            continue
        record_agent_id = _string(record.get("agent_id"), f"{kind}.agent_id")
        payload = _mapping(record.get("payload"), f"{kind}.payload")
        event = _mapping(payload.get("event"), f"{kind}.event")

        if kind in {"turn_started", "turn_finished", "stream_event"} and record_agent_id != "main":
            if turn_id is None:
                continue
            child_key = (record_agent_id, turn_id)
            if kind == "turn_started":
                if child_key in child_turns:
                    raise RecoveryError(f"duplicate turn_started for {record_agent_id}:{turn_id}")
                child_turns[child_key] = _TurnFragments()
            elif kind == "turn_finished":
                if child_key not in child_turns:
                    raise RecoveryError(f"turn_finished without turn_started for {record_agent_id}:{turn_id}")
                if event.get("status") != "cancelled":
                    child_turns.pop(child_key, None)
            elif child_key in child_turns:
                _apply_stream_event(child_turns[child_key], event)
            continue

        if kind == "input_buffered":
            source_id = _string(event.get("input_id"), "input_buffered.input_id")
            if source_id in input_status:
                raise RecoveryError(f"duplicate input id: {source_id}")
            raw_submitted_at = event.get("submitted_at")
            submitted_at = (
                _timestamp(raw_submitted_at, "input_buffered.submitted_at") if raw_submitted_at is not None else None
            )
            raw_author = event.get("author")
            pending[source_id] = RecoveredPendingInput(
                source_id=source_id,
                text=_string(event.get("text"), "input_buffered.text"),
                target_agent_id=_string(
                    event.get("intended_target_agent_id"),
                    "input_buffered.intended_target_agent_id",
                ),
                submitted_at=submitted_at,
                author=_string(raw_author, "input_buffered.author") if raw_author is not None else "human",
            )
            input_status[source_id] = "pending"
            editor_text = ""
        elif kind == "input_recalled":
            source_ids = _string_tuple(event.get("input_ids"), "input_recalled.input_ids")
            _require_input_status(input_status, source_ids, "pending", "input_recalled")
            for source_id in source_ids:
                pending.pop(source_id, None)
                input_status[source_id] = "recalled"
            editor_text = _string(event.get("editor_text"), "input_recalled.editor_text")
        elif kind == "input_claimed":
            source_ids = _string_tuple(event.get("input_ids"), "input_claimed.input_ids")
            _require_input_status(input_status, source_ids, "pending", "input_claimed")
            claimed_target = _string(event.get("target_agent_id"), "input_claimed.target_agent_id")
            for source_id in source_ids:
                input_status[source_id] = "claimed"
                pending[source_id] = replace(pending[source_id], target_agent_id=claimed_target)
        elif kind == "input_delivered":
            source_ids = _string_tuple(event.get("input_ids"), "input_delivered.input_ids")
            for source_id in source_ids:
                actual = input_status.get(source_id)
                if actual not in {"claimed", "delivered"}:
                    raise RecoveryError(
                        f"input_delivered requires input {source_id!r} in claimed state, got {actual or 'unknown'}"
                    )
                pending.pop(source_id, None)
                input_status[source_id] = "delivered"
        elif kind == "editor_snapshot":
            editor_text = _string(event.get("text"), "editor_snapshot.text")
        elif kind == "recovery_applied":
            applied.update(_string_tuple(event.get("recovery_ids"), "recovery_applied.recovery_ids"))
        elif kind == "interruption_committed":
            interrupted_turn_id = event.get("captured_turn_id")
            if isinstance(interrupted_turn_id, str) and interrupted_turn_id:
                target_agent_id = _string(
                    event.get("target_agent_id"),
                    "interruption_committed.target_agent_id",
                )
                reason = _string(event.get("reason"), "interruption_committed.reason")
                if target_agent_id == "main":
                    turn_reasons[interrupted_turn_id] = reason
                else:
                    child_turn_reasons[(target_agent_id, interrupted_turn_id)] = reason
        elif kind == "shutdown_recorded":
            interrupted_turn_id = event.get("interrupted_turn_id")
            if isinstance(interrupted_turn_id, str) and interrupted_turn_id:
                turn_reasons[interrupted_turn_id] = _string(event.get("reason"), "shutdown_recorded.reason")
            deferred_ids = _string_tuple(
                event.get("deferred_tool_use_ids"),
                "shutdown_recorded.deferred_tool_use_ids",
            )
            raw_agent_ids = event.get("deferred_tool_agent_ids", [])
            raw_turn_ids = event.get("deferred_tool_turn_ids", [])
            raw_phases = event.get("deferred_tool_phases", [])
            agent_ids = _string_tuple(
                raw_agent_ids,
                "shutdown_recorded.deferred_tool_agent_ids",
                unique=False,
            )
            turn_ids = _optional_string_tuple(raw_turn_ids, "shutdown_recorded.deferred_tool_turn_ids")
            phases = _string_tuple(
                raw_phases,
                "shutdown_recorded.deferred_tool_phases",
                unique=False,
            )
            if not agent_ids and deferred_ids:
                agent_ids = ("main",) * len(deferred_ids)
            if not turn_ids and deferred_ids:
                turn_ids = (None,) * len(deferred_ids)
            if not phases and deferred_ids:
                phases = ("unknown",) * len(deferred_ids)
            if (
                len(deferred_ids) != len(agent_ids)
                or len(deferred_ids) != len(turn_ids)
                or len(deferred_ids) != len(phases)
            ):
                raise RecoveryError("shutdown deferred-tool snapshot fields have different lengths")
            deferred_tools.extend(
                _DeferredRecovery(tool_use_id, agent_id, deferred_turn_id, phase)
                for tool_use_id, agent_id, deferred_turn_id, phase in zip(
                    deferred_ids,
                    agent_ids,
                    turn_ids,
                    phases,
                    strict=True,
                )
            )
        elif kind == "turn_started" and turn_id is not None:
            if turn_id in turns:
                raise RecoveryError(f"duplicate turn_started for {turn_id}")
            turns[turn_id] = _TurnFragments()
        elif kind == "turn_finished" and turn_id is not None:
            if turn_id not in turns:
                raise RecoveryError(f"turn_finished without turn_started for {turn_id}")
            if event.get("status") != "cancelled":
                turns.pop(turn_id, None)
        elif kind == "stream_event" and turn_id is not None and turn_id in turns:
            _apply_stream_event(turns[turn_id], event)

    recovery_ids: list[str] = []
    for turn_id, fragments in turns.items():
        recovery_id = f"{source_session_id}:{turn_id}"
        if recovery_id in applied or turn_id in committed_interrupt_notices:
            continue
        if partial_text := "".join(fragments.text):
            if partial_text not in committed_assistant_text.get(turn_id, ()):
                messages.append(Message(role="assistant", content=[TextBlock(text=partial_text)]))
        messages.append(
            Message(
                role="user",
                content=[
                    TextBlock(
                        text=_interruption_notice(
                            turn_id,
                            fragments,
                            reason=turn_reasons.get(turn_id),
                        )
                    )
                ],
                provenance=InputProvenance(
                    human_authored=False,
                    source="recovery",
                    author="axio-repl",
                ),
            )
        )
        recovery_ids.append(recovery_id)

    for (agent_id, turn_id), fragments in child_turns.items():
        recovery_id = f"{source_session_id}:agent:{agent_id}:{turn_id}"
        if recovery_id in applied:
            continue
        if not fragments.text:
            fragments.text.extend(child_committed_assistant_text.get((agent_id, turn_id), ()))
        messages.append(
            Message(
                role="user",
                content=[
                    TextBlock(
                        text=_interruption_notice(
                            turn_id,
                            fragments,
                            reason=child_turn_reasons.get((agent_id, turn_id)),
                            agent_id=agent_id,
                        )
                    )
                ],
                provenance=InputProvenance(
                    human_authored=False,
                    source="recovery",
                    author=agent_id,
                ),
            )
        )
        recovery_ids.append(recovery_id)

    for deferred in deferred_tools:
        turn_label = deferred.turn_id or "unknown"
        recovery_id = f"{source_session_id}:deferred:{deferred.agent_id}:{turn_label}:{deferred.tool_use_id}"
        if recovery_id in applied:
            continue
        messages.append(
            Message(
                role="user",
                content=[
                    TextBlock(
                        text=(
                            "[Recovered deferred tool was cancelled when the previous process stopped: "
                            f"call_id={deferred.tool_use_id}, original_agent={deferred.agent_id}, "
                            f"original_turn={turn_label}, "
                            f"last_phase={deferred.phase}.]"
                        )
                    )
                ],
                provenance=InputProvenance(
                    human_authored=False,
                    source="recovery",
                    author="axio-repl",
                ),
            )
        )
        recovery_ids.append(recovery_id)

    return RecoveryMaterialization(
        source_session_id=source_session_id,
        messages=tuple(messages),
        pending_inputs=tuple(pending.values()),
        editor_text=editor_text,
        recovery_ids=tuple(recovery_ids),
        discarded_tail_bytes=result.discarded_tail_bytes,
    )


def _apply_stream_event(fragments: _TurnFragments, event: dict[str, Any]) -> None:
    record_type = event.get("record_type")
    if record_type == "TextDelta":
        fragments.text.append(_string(event.get("delta"), "TextDelta.delta"))
    elif record_type == "ReasoningDelta":
        fragments.reasoning.append(_string(event.get("delta"), "ReasoningDelta.delta"))
    elif record_type == "ToolUseStart":
        tool_use_id = _string(event.get("tool_use_id"), "ToolUseStart.tool_use_id")
        fragments.tool_names[tool_use_id] = _string(event.get("name"), "ToolUseStart.name")
        argument_codec = event.get("argument_codec")
        if argument_codec is not None:
            fragments.tool_argument_codecs[tool_use_id] = _string(
                argument_codec,
                "ToolUseStart.argument_codec",
            )
        fragments.tool_arguments.setdefault(tool_use_id, [])
    elif record_type == "ToolInputDelta":
        tool_use_id = _string(event.get("tool_use_id"), "ToolInputDelta.tool_use_id")
        fragments.tool_arguments.setdefault(tool_use_id, []).append(
            _string(event.get("partial_json"), "ToolInputDelta.partial_json")
        )
    elif record_type == "ToolOutputDelta":
        tool_use_id = _string(event.get("tool_use_id"), "ToolOutputDelta.tool_use_id")
        fragments.tool_output.setdefault(tool_use_id, []).append(_string(event.get("delta"), "ToolOutputDelta.delta"))


def _apply_turn_checkpoint(fragments: _TurnFragments, payload: dict[str, Any]) -> None:
    text = _string(payload.get("text"), "turn_checkpoint.text")
    if text:
        fragments.text.append(text)
    for tool_use_id, name in _string_mapping(payload.get("tool_names"), "turn_checkpoint.tool_names").items():
        fragments.tool_names[tool_use_id] = name
        fragments.tool_arguments.setdefault(tool_use_id, [])
    for tool_use_id, codec in _string_mapping(
        payload.get("tool_argument_codecs", {}),
        "turn_checkpoint.tool_argument_codecs",
    ).items():
        fragments.tool_argument_codecs[tool_use_id] = codec
    for tool_use_id, arguments in _string_mapping(
        payload.get("tool_arguments"), "turn_checkpoint.tool_arguments"
    ).items():
        if arguments:
            fragments.tool_arguments.setdefault(tool_use_id, []).append(arguments)
    raw_fields = _mapping(payload.get("tool_fields"), "turn_checkpoint.tool_fields")
    for raw_tool_use_id, raw_values in raw_fields.items():
        tool_use_id = _string(raw_tool_use_id, "turn_checkpoint.tool_fields key")
        values = _string_mapping(raw_values, f"turn_checkpoint.tool_fields.{tool_use_id}")
        target = fragments.tool_fields.setdefault(tool_use_id, {})
        for key, value in values.items():
            if value:
                target.setdefault(key, []).append(value)
            else:
                target.setdefault(key, [])
    for tool_use_id, output in _string_mapping(payload.get("tool_output"), "turn_checkpoint.tool_output").items():
        if output:
            fragments.tool_output.setdefault(tool_use_id, []).append(output)


def _interruption_notice(
    turn_id: str,
    fragments: _TurnFragments,
    *,
    reason: str | None,
    agent_id: str | None = None,
) -> str:
    reason_detail = f" Recorded reason: {reason}." if reason is not None else ""
    if agent_id is None:
        header = f"[Recovered interrupted turn {turn_id}. The previous process ended before context commit."
    else:
        header = (
            f"[Recovered interrupted background-agent turn {turn_id}; original_agent={agent_id}. "
            "The child context is unavailable after process restart."
        )
    sections = [f"{header}{reason_detail}]"]
    if agent_id is not None and fragments.text:
        sections.append("Available assistant text fragment:\n" + "".join(fragments.text))
    if fragments.reasoning:
        sections.append("Available reasoning fragment:\n" + "".join(fragments.reasoning))
    for tool_use_id, name in fragments.tool_names.items():
        tool_detail = [f"Available partial tool call: name={name}, call_id={tool_use_id}"]
        if arguments := "".join(fragments.tool_arguments.get(tool_use_id, ())):
            codec = fragments.tool_argument_codecs.get(tool_use_id)
            if codec is None:
                display_arguments = arguments
            else:
                try:
                    display_arguments = json.dumps(
                        sanitize_presentation_value(decode_framed_values(json.loads(arguments), codec)),
                        ensure_ascii=False,
                    )
                except (json.JSONDecodeError, ToolArgumentCodecError):
                    display_arguments = "[invalid or incomplete encoded arguments]"
            tool_detail.append("Arguments fragment:\n" + display_arguments)
        if fields := fragments.tool_fields.get(tool_use_id):
            field_lines = [f"{key}: {''.join(parts)}" for key, parts in fields.items()]
            tool_detail.append("Argument fields:\n" + "\n".join(field_lines))
        if output := "".join(fragments.tool_output.get(tool_use_id, ())):
            tool_detail.append("Output fragment:\n" + output)
        sections.append("\n".join(tool_detail))
    return "\n\n".join(sections)


def _message_text(message: Message) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


def _notice_turn_ids(message: Message) -> set[str]:
    result: set[str] = set()
    for block in message.content:
        if not isinstance(block, TextBlock) or not block.text.startswith("[Turn "):
            continue
        turn_id, separator, suffix = block.text.removeprefix("[Turn ").partition(" ")
        if separator and suffix.startswith("was interrupted by Escape."):
            result.add(turn_id)
    return result


def _decode_message(data: dict[str, Any], session_dir: Path) -> Message:
    role = data.get("role")
    if role not in {"user", "assistant", "system"}:
        raise RecoveryError(f"invalid message role: {role!r}")
    raw_content = data.get("content")
    if not isinstance(raw_content, list):
        raise RecoveryError("message content is not a list")
    provenance: InputProvenance | None = None
    raw_provenance = data.get("provenance")
    if raw_provenance is not None:
        provenance_data = _mapping(raw_provenance, "message.provenance")
        record_type = provenance_data.get("record_type")
        if record_type not in {None, "InputProvenance"}:
            raise RecoveryError(f"invalid message provenance record type: {record_type!r}")
        try:
            provenance = InputProvenance.from_dict(provenance_data)
        except ValueError as error:
            raise RecoveryError(f"invalid message provenance: {error}") from error
    return Message(
        role=cast(Literal["user", "assistant", "system"], role),
        content=[_decode_block(_mapping(block, "message block"), session_dir) for block in raw_content],
        provenance=provenance,
    )


def _decode_block(data: dict[str, Any], session_dir: Path) -> ContentBlock:
    record_type = data.get("record_type")
    if record_type == "TextBlock":
        return TextBlock(text=_string(data.get("text"), "TextBlock.text"))
    if record_type in {"ImageBlock", "AudioBlock", "VideoBlock"}:
        media_type = _string(data.get("media_type"), f"{record_type}.media_type")
        binary = _read_attachment(_mapping(data.get("data"), f"{record_type}.data"), session_dir)
        if record_type == "ImageBlock":
            return ImageBlock(media_type=_image_media_type(media_type), data=binary)
        if record_type == "AudioBlock":
            return AudioBlock(media_type=_audio_media_type(media_type), data=binary)
        return VideoBlock(media_type=_video_media_type(media_type), data=binary)
    if record_type == "ToolUseBlock":
        raw_input = data.get("input")
        if not isinstance(raw_input, dict):
            raise RecoveryError("ToolUseBlock.input is not an object")
        return ToolUseBlock(
            id=_string(data.get("id"), "ToolUseBlock.id"),
            name=_string(data.get("name"), "ToolUseBlock.name"),
            input=cast(dict[str, Any], raw_input),
        )
    if record_type == "ToolResultBlock":
        raw_content = data.get("content")
        content: str | list[TextBlock | ImageBlock | AudioBlock | VideoBlock]
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            nested: list[TextBlock | ImageBlock | AudioBlock | VideoBlock] = []
            for raw_block in raw_content:
                block = _decode_block(_mapping(raw_block, "ToolResultBlock block"), session_dir)
                if not isinstance(block, (TextBlock, ImageBlock, AudioBlock, VideoBlock)):
                    raise RecoveryError("ToolResultBlock contains a protocol block")
                nested.append(block)
            content = nested
        else:
            raise RecoveryError("ToolResultBlock.content has an invalid type")
        return ToolResultBlock(
            tool_use_id=_string(data.get("tool_use_id"), "ToolResultBlock.tool_use_id"),
            content=content,
            is_error=bool(data.get("is_error")),
        )
    raise RecoveryError(f"unsupported message block type: {record_type!r}")


def _read_attachment(data: dict[str, Any], session_dir: Path) -> bytes:
    if data.get("type") != "attachment":
        raise RecoveryError("binary block does not reference an attachment")
    relative = Path(_string(data.get("path"), "attachment.path"))
    root = session_dir.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise RecoveryError("attachment path escapes the session directory")
    binary = target.read_bytes()
    expected_size = data.get("size")
    if not isinstance(expected_size, int) or len(binary) != expected_size:
        raise RecoveryError("attachment size does not match its journal record")
    expected_digest = _string(data.get("sha256"), "attachment.sha256")
    if hashlib.sha256(binary).hexdigest() != expected_digest:
        raise RecoveryError("attachment digest does not match its journal record")
    return binary


def _image_media_type(value: str) -> ImageMediaType:
    if value not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        raise RecoveryError(f"unsupported image media type: {value!r}")
    return cast(ImageMediaType, value)


def _audio_media_type(value: str) -> AudioMediaType:
    if value not in {
        "audio/x-aac",
        "audio/flac",
        "audio/mp3",
        "audio/m4a",
        "audio/mpeg",
        "audio/mpga",
        "audio/mp4",
        "audio/ogg",
        "audio/pcm",
        "audio/wav",
        "audio/webm",
    }:
        raise RecoveryError(f"unsupported audio media type: {value!r}")
    return cast(AudioMediaType, value)


def _video_media_type(value: str) -> VideoMediaType:
    if value not in {
        "video/mp4",
        "video/mpeg",
        "video/mov",
        "video/avi",
        "video/x-flv",
        "video/mpg",
        "video/webm",
        "video/wmv",
        "video/3gpp",
    }:
        raise RecoveryError(f"unsupported video media type: {value!r}")
    return cast(VideoMediaType, value)


def _mapping(value: object, label: str, *, required: bool = True) -> dict[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} is not an object")
    return cast(dict[str, Any], value)


def _string_mapping(value: object, label: str) -> dict[str, str]:
    mapping = _mapping(value, label)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise RecoveryError(f"{label} is not a string mapping")
        result[key] = item
    return result


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RecoveryError(f"{label} is not a string")
    return value


def _timestamp(value: object, label: str) -> datetime:
    raw = _string(value, label)
    try:
        result = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise RecoveryError(f"{label} is not an ISO 8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise RecoveryError(f"{label} must be timezone-aware")
    return result


def _string_tuple(value: object, label: str, *, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RecoveryError(f"{label} is not a string list")
    result = tuple(value)
    if unique and len(result) != len(set(result)):
        raise RecoveryError(f"{label} contains duplicate ids")
    return result


def _optional_string_tuple(value: object, label: str) -> tuple[str | None, ...]:
    if not isinstance(value, list) or not all(item is None or isinstance(item, str) for item in value):
        raise RecoveryError(f"{label} is not a nullable string list")
    return tuple(value)


def _require_input_status(
    statuses: dict[str, str],
    source_ids: tuple[str, ...],
    expected: str,
    transition: str,
) -> None:
    for source_id in source_ids:
        actual = statuses.get(source_id)
        if actual != expected:
            raise RecoveryError(
                f"{transition} requires input {source_id!r} in {expected} state, got {actual or 'unknown'}"
            )


def _confirm_claimed_input_commit(
    pending: dict[str, RecoveredPendingInput],
    statuses: dict[str, str],
    *,
    target_agent_id: str,
    source_input_id: str,
    message_text: str,
) -> None:
    entry = pending.get(source_input_id)
    status = statuses.get(source_input_id)
    if status == "delivered":
        return
    if entry is None or status != "claimed":
        raise RecoveryError(f"message commit references input {source_input_id!r} in {status or 'unknown'} state")
    if entry.target_agent_id != target_agent_id:
        raise RecoveryError(
            f"message commit target {target_agent_id!r} does not match input {source_input_id!r} target"
        )
    if entry.text != message_text:
        raise RecoveryError(f"message commit content does not match input {source_input_id!r}")
    pending.pop(source_input_id, None)
    statuses[source_input_id] = "delivered"
