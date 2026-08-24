"""Provider-wire codecs for tool arguments.

Some OpenAI-compatible serving stacks parse native tool markup before emitting
``function.arguments`` and apply ``strip()`` to each string parameter. Axio
cannot reconstruct data removed there. The structural frame below moves an
exact string inside a one-property JSON object, making its boundary whitespace
interior JSON data while keeping the source text readable.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote

TOOL_ARGUMENT_CODEC = "axio.verbatim.v1"
TOOL_ARGUMENT_FRAME_KEY = "__axio_verbatim_v1__"
_SCHEMA_CODEC_KEY = "x-axio-tool-argument-codec"
_FRAME_DESCRIPTION = (
    f"Wire format for one exact string: pass an object whose only property is "
    f"{TOOL_ARGUMENT_FRAME_KEY!r}; its value is the original string. Preserve the inner value exactly, "
    "including leading/trailing spaces, tabs, newlines, and an empty value."
)

_SCHEMA_MAP_KEYWORDS = frozenset(
    {
        "$defs",
        "definitions",
        "properties",
        "patternProperties",
        "dependentSchemas",
    }
)
_SCHEMA_SINGLE_KEYWORDS = frozenset({"additionalProperties", "items"})
_SCHEMA_LIST_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_IDENTIFIERS = ("$anchor", "$dynamicAnchor", "$id")
_UNSUPPORTED_VALUE_KEYWORDS = ("contains", "else", "if", "then", "unevaluatedItems", "unevaluatedProperties")


class ToolArgumentCodecError(ValueError):
    """Provider tool arguments do not satisfy the advertised wire framing."""


def _require_codec(codec: str) -> None:
    if codec != TOOL_ARGUMENT_CODEC:
        raise ToolArgumentCodecError(f"Unsupported tool argument codec: {codec!r}")


def _is_direct_string_schema(schema: Mapping[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type == "string":
        return True
    if isinstance(schema.get("const"), str):
        return True
    enum = schema.get("enum")
    return isinstance(enum, list) and bool(enum) and all(isinstance(item, str) for item in enum)


def _has_direct_string_constraint(schema: Mapping[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type == "string" or isinstance(schema_type, list) and "string" in schema_type:
        return True
    if isinstance(schema.get("const"), str):
        return True
    enum = schema.get("enum")
    return isinstance(enum, list) and any(isinstance(item, str) for item in enum)


def _wrap_string_schema(schema: dict[str, Any], *, include_marker: bool) -> dict[str, Any]:
    inner = copy.deepcopy(schema)
    original_description = inner.get("description")
    default = inner.pop("default", None) if isinstance(inner.get("default"), str) else None
    identifiers = {key: inner.pop(key) for key in _SCHEMA_IDENTIFIERS if key in inner}
    description = _FRAME_DESCRIPTION
    if isinstance(original_description, str) and original_description:
        description = f"{description} Decoded value: {original_description}"
    wrapped: dict[str, Any] = {
        "type": "object",
        "properties": {TOOL_ARGUMENT_FRAME_KEY: inner},
        "required": [TOOL_ARGUMENT_FRAME_KEY],
        "additionalProperties": False,
        "description": description,
    }
    if include_marker:
        wrapped[_SCHEMA_CODEC_KEY] = TOOL_ARGUMENT_CODEC
    wrapped.update(identifiers)
    if default is not None:
        wrapped["default"] = {TOOL_ARGUMENT_FRAME_KEY: default}
    return wrapped


def _split_string_type(schema: dict[str, Any], *, include_marker: bool) -> dict[str, Any] | None:
    schema_type = schema.get("type")
    if not isinstance(schema_type, list) or "string" not in schema_type:
        return None
    remaining = [item for item in schema_type if item != "string"]
    identifiers = {key: copy.deepcopy(schema[key]) for key in _SCHEMA_IDENTIFIERS if key in schema}
    branch_schema = copy.deepcopy(schema)
    for key in identifiers:
        branch_schema.pop(key, None)
    string_branch = copy.deepcopy(branch_schema)
    string_branch["type"] = "string"
    if not remaining:
        wrapped = _wrap_string_schema(string_branch, include_marker=include_marker)
        wrapped.update(identifiers)
        return wrapped
    other_branch = copy.deepcopy(branch_schema)
    other_branch["type"] = remaining[0] if len(remaining) == 1 else remaining
    split = {
        "anyOf": [
            _wrap_string_schema(string_branch, include_marker=include_marker),
            _encode_schema_node(other_branch, include_marker=include_marker),
        ]
    }
    split.update(identifiers)
    return split


def _split_mixed_enum(schema: dict[str, Any], *, include_marker: bool) -> dict[str, Any] | None:
    if "type" in schema:
        return None
    enum = schema.get("enum")
    if not isinstance(enum, list):
        return None
    strings = [item for item in enum if isinstance(item, str)]
    others = [item for item in enum if not isinstance(item, str)]
    if not strings or not others:
        return None
    identifiers = {key: copy.deepcopy(schema[key]) for key in _SCHEMA_IDENTIFIERS if key in schema}
    branch_schema = copy.deepcopy(schema)
    for key in identifiers:
        branch_schema.pop(key, None)
    string_branch = copy.deepcopy(branch_schema)
    string_branch["type"] = "string"
    string_branch["enum"] = strings
    other_branch = copy.deepcopy(branch_schema)
    other_branch["enum"] = others
    split = {
        "anyOf": [
            _wrap_string_schema(string_branch, include_marker=include_marker),
            _encode_schema_node(other_branch, include_marker=include_marker),
        ]
    }
    split.update(identifiers)
    return split


def _encode_schema_node(schema: dict[str, Any], *, include_marker: bool) -> dict[str, Any]:
    if "$ref" in schema and _has_direct_string_constraint(schema):
        raise ToolArgumentCodecError(
            f"String schema {schema['$ref']!r} has validation siblings and cannot be framed safely"
        )
    if split := _split_string_type(schema, include_marker=include_marker):
        return split
    if split := _split_mixed_enum(schema, include_marker=include_marker):
        return split
    if _is_direct_string_schema(schema):
        return _wrap_string_schema(schema, include_marker=include_marker)

    result = copy.deepcopy(schema)
    for keyword in _SCHEMA_MAP_KEYWORDS:
        value = result.get(keyword)
        if isinstance(value, dict):
            result[keyword] = {
                key: _encode_schema_node(item, include_marker=include_marker) if isinstance(item, dict) else item
                for key, item in value.items()
            }
    for keyword in _SCHEMA_SINGLE_KEYWORDS:
        value = result.get(keyword)
        if isinstance(value, dict):
            result[keyword] = _encode_schema_node(value, include_marker=include_marker)
    for keyword in _SCHEMA_LIST_KEYWORDS:
        value = result.get(keyword)
        if isinstance(value, list):
            result[keyword] = [
                _encode_schema_node(item, include_marker=include_marker) if isinstance(item, dict) else item
                for item in value
            ]
    return result


def encode_tool_schema(schema: Mapping[str, Any], codec: str) -> dict[str, Any]:
    """Return the provider-facing schema for *codec* without mutating *schema*."""

    _require_codec(codec)
    _build_internal_schema(schema)
    return _encode_schema_node(dict(schema), include_marker=False)


def _find_anchor(value: Any, anchor: str, seen: set[int]) -> Mapping[str, Any] | None:
    identity = id(value)
    if identity in seen:
        return None
    seen.add(identity)
    if isinstance(value, Mapping):
        if value.get("$anchor") == anchor or value.get("$dynamicAnchor") == anchor:
            return value
        for child in value.values():
            found = _find_anchor(child, anchor, seen)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_anchor(child, anchor, seen)
            if found is not None:
                return found
    return None


def _resolve_local_ref(root: Mapping[str, Any], ref: str, path: str) -> Mapping[str, Any]:
    if ref == "#":
        return root
    if not ref.startswith("#"):
        raise ToolArgumentCodecError(f"{path}: external schema reference {ref!r} cannot be decoded")
    if not ref.startswith("#/"):
        anchor = unquote(ref[1:])
        resolved = _find_anchor(root, anchor, set())
        if resolved is None:
            raise ToolArgumentCodecError(f"{path}: unresolved schema anchor {ref!r}")
        return resolved
    current: Any = root
    for raw_part in unquote(ref[2:]).split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise ToolArgumentCodecError(f"{path}: unresolved schema reference {ref!r}")
        current = current[part]
    if not isinstance(current, Mapping):
        raise ToolArgumentCodecError(f"{path}: schema reference {ref!r} does not resolve to an object")
    return current


def _value_matches_type(value: Any, schema_type: object) -> bool:
    if isinstance(schema_type, list):
        return any(_value_matches_type(value, item) for item in schema_type)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return True


def _wire_shape_matches(
    value: Any,
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    encode: bool,
    active: set[tuple[int, int, bool]] | None = None,
) -> bool:
    active = set() if active is None else active
    token = (id(value), id(schema), encode)
    if token in active:
        return True
    active.add(token)
    try:
        ref = schema.get("$ref")
        if isinstance(ref, str) and not _wire_shape_matches(
            value,
            _resolve_local_ref(root, ref, "$"),
            root,
            encode,
            active,
        ):
            return False
        if schema.get(_SCHEMA_CODEC_KEY) == TOOL_ARGUMENT_CODEC:
            if encode:
                return isinstance(value, str)
            return (
                isinstance(value, dict)
                and set(value) == {TOOL_ARGUMENT_FRAME_KEY}
                and isinstance(value[TOOL_ARGUMENT_FRAME_KEY], str)
            )

        for keyword in ("anyOf", "oneOf"):
            branches = schema.get(keyword)
            if isinstance(branches, list):
                matches = sum(
                    isinstance(branch, Mapping) and _wire_shape_matches(value, branch, root, encode, active)
                    for branch in branches
                )
                if matches == 0 or (keyword == "oneOf" and matches != 1):
                    return False
        all_of = schema.get("allOf")
        if isinstance(all_of, list) and not all(
            not isinstance(branch, Mapping) or _wire_shape_matches(value, branch, root, encode, active)
            for branch in all_of
        ):
            return False

        schema_type = schema.get("type")
        if schema_type is not None and not _value_matches_type(value, schema_type):
            return False
        enum = schema.get("enum")
        if isinstance(enum, list) and value not in enum:
            return False
        if "const" in schema and value != schema["const"]:
            return False

        if isinstance(value, str):
            if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
                return False
            if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
                return False
            pattern = schema.get("pattern")
            if isinstance(pattern, str):
                try:
                    if re.search(pattern, value) is None:
                        return False
                except re.error:
                    return False

        if isinstance(value, dict):
            required = schema.get("required")
            if isinstance(required, list) and any(not isinstance(key, str) or key not in value for key in required):
                return False
            properties = schema.get("properties")
            property_schemas = properties if isinstance(properties, Mapping) else {}
            patterns = schema.get("patternProperties")
            pattern_schemas = patterns if isinstance(patterns, Mapping) else {}
            additional = schema.get("additionalProperties", True)
            for key, item in value.items():
                candidates: list[Mapping[str, Any]] = []
                explicit = property_schemas.get(key)
                if isinstance(explicit, Mapping):
                    candidates.append(explicit)
                for pattern, pattern_schema in pattern_schemas.items():
                    try:
                        matches_pattern = re.search(str(pattern), key) is not None
                    except re.error:
                        return False
                    if matches_pattern and isinstance(pattern_schema, Mapping):
                        candidates.append(pattern_schema)
                if not candidates:
                    if additional is False:
                        return False
                    if isinstance(additional, Mapping):
                        candidates.append(additional)
                if any(not _wire_shape_matches(item, candidate, root, encode, active) for candidate in candidates):
                    return False
            if isinstance(schema.get("minProperties"), int) and len(value) < schema["minProperties"]:
                return False
            if isinstance(schema.get("maxProperties"), int) and len(value) > schema["maxProperties"]:
                return False
            dependent_schemas = schema.get("dependentSchemas")
            if isinstance(dependent_schemas, Mapping):
                for dependency, dependent_schema in dependent_schemas.items():
                    if (
                        dependency in value
                        and isinstance(dependent_schema, Mapping)
                        and not _wire_shape_matches(value, dependent_schema, root, encode, active)
                    ):
                        return False

        if isinstance(value, list):
            prefix = schema.get("prefixItems")
            prefix_schemas = prefix if isinstance(prefix, list) else []
            items = schema.get("items")
            for index, item in enumerate(value):
                item_schema: Mapping[str, Any] | None = None
                if index < len(prefix_schemas) and isinstance(prefix_schemas[index], Mapping):
                    item_schema = prefix_schemas[index]
                elif isinstance(items, Mapping):
                    item_schema = items
                elif items is False:
                    return False
                if item_schema is not None and not _wire_shape_matches(item, item_schema, root, encode, active):
                    return False
            if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
                return False
            if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
                return False
        return True
    finally:
        active.discard(token)


def _child_path(path: str, key: str) -> str:
    return f"{path}.{key}" if key.isidentifier() else f"{path}[{key!r}]"


def _frame_value(value: Any, path: str, encode: bool) -> Any:
    if encode:
        if not isinstance(value, str):
            raise ToolArgumentCodecError(f"{path}: framed tool argument must be a string, got {type(value).__name__}")
        return {TOOL_ARGUMENT_FRAME_KEY: value}
    if not isinstance(value, dict):
        raise ToolArgumentCodecError(f"{path}: missing {TOOL_ARGUMENT_FRAME_KEY!r} string frame")
    if set(value) != {TOOL_ARGUMENT_FRAME_KEY}:
        raise ToolArgumentCodecError(f"{path}: string frame must contain only {TOOL_ARGUMENT_FRAME_KEY!r}")
    decoded = value[TOOL_ARGUMENT_FRAME_KEY]
    if not isinstance(decoded, str):
        raise ToolArgumentCodecError(
            f"{path}: {TOOL_ARGUMENT_FRAME_KEY!r} must contain a string, got {type(decoded).__name__}"
        )
    return decoded


def _transform_union(
    value: Any,
    branches: list[Any],
    root: Mapping[str, Any],
    path: str,
    encode: bool,
    active: set[tuple[int, int, bool]],
    completed_frames: set[str],
) -> Any:
    errors: list[ToolArgumentCodecError] = []
    transformed: list[tuple[Any, set[str]]] = []
    candidates = [branch for branch in branches if isinstance(branch, Mapping)]
    if (
        not encode
        and isinstance(value, dict)
        and set(value) == {TOOL_ARGUMENT_FRAME_KEY}
        and isinstance(value[TOOL_ARGUMENT_FRAME_KEY], str)
    ):
        candidates.sort(key=lambda branch: not _accepts_frame_at_root(branch, root, set()))
    for branch in candidates:
        if not isinstance(branch, Mapping) or not _wire_shape_matches(value, branch, root, encode):
            continue
        candidate_frames = set(completed_frames)
        try:
            candidate_value = _transform_value(value, branch, root, path, encode, active, candidate_frames)
        except ToolArgumentCodecError as exc:
            errors.append(exc)
        else:
            transformed.append((candidate_value, candidate_frames))
    if transformed:
        selected, selected_frames = transformed[0]
        if any(candidate != selected for candidate, _frames in transformed[1:]):
            raise ToolArgumentCodecError(f"{path}: union branches require ambiguous wire transformations")
        completed_frames.update(selected_frames)
        return selected
    if errors:
        raise errors[0]
    if not encode and any(isinstance(branch, Mapping) and _contains_codec(branch, root, set()) for branch in branches):
        raise ToolArgumentCodecError(f"{path}: value does not match the required encoded string frame")
    return value


def _accepts_frame_at_root(schema: Mapping[str, Any], root: Mapping[str, Any], seen: set[int]) -> bool:
    identity = id(schema)
    if identity in seen:
        return False
    seen.add(identity)
    if schema.get(_SCHEMA_CODEC_KEY) == TOOL_ARGUMENT_CODEC:
        return True
    ref = schema.get("$ref")
    if isinstance(ref, str) and _accepts_frame_at_root(_resolve_local_ref(root, ref, "$"), root, seen):
        return True
    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list) and any(
            isinstance(branch, Mapping) and _accepts_frame_at_root(branch, root, seen) for branch in branches
        ):
            return True
    return False


def _schema_may_accept_string(schema: Mapping[str, Any], root: Mapping[str, Any], seen: set[int]) -> bool:
    identity = id(schema)
    if identity in seen:
        return True
    seen.add(identity)
    if schema.get(_SCHEMA_CODEC_KEY) == TOOL_ARGUMENT_CODEC:
        return False
    ref = schema.get("$ref")
    if isinstance(ref, str) and not _schema_may_accept_string(_resolve_local_ref(root, ref, "$"), root, seen):
        return False
    schema_type = schema.get("type")
    if schema_type is not None and not _value_matches_type("value", schema_type):
        return False
    if "const" in schema:
        return isinstance(schema["const"], str)
    enum = schema.get("enum")
    if isinstance(enum, list):
        return any(isinstance(item, str) for item in enum)
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            return any(
                isinstance(branch, Mapping) and _schema_may_accept_string(branch, root, set(seen))
                for branch in branches
            )
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        return all(
            not isinstance(branch, Mapping) or _schema_may_accept_string(branch, root, set(seen)) for branch in all_of
        )
    return True


def _can_accept_reserved_object(schema: Mapping[str, Any], root: Mapping[str, Any], seen: set[int]) -> bool:
    identity = id(schema)
    if identity in seen:
        return True
    seen.add(identity)
    if schema.get(_SCHEMA_CODEC_KEY) == TOOL_ARGUMENT_CODEC:
        return False
    ref = schema.get("$ref")
    if isinstance(ref, str) and not _can_accept_reserved_object(_resolve_local_ref(root, ref, "$"), root, seen):
        return False
    schema_type = schema.get("type")
    if schema_type is not None and not _value_matches_type({}, schema_type):
        return False
    if "const" in schema:
        constant = schema["const"]
        return (
            isinstance(constant, dict)
            and set(constant) == {TOOL_ARGUMENT_FRAME_KEY}
            and isinstance(constant[TOOL_ARGUMENT_FRAME_KEY], str)
        )
    enum = schema.get("enum")
    if isinstance(enum, list):
        return any(
            isinstance(item, dict)
            and set(item) == {TOOL_ARGUMENT_FRAME_KEY}
            and isinstance(item[TOOL_ARGUMENT_FRAME_KEY], str)
            for item in enum
        )
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            return any(
                isinstance(branch, Mapping) and _can_accept_reserved_object(branch, root, set(seen))
                for branch in branches
            )
    all_of = schema.get("allOf")
    if isinstance(all_of, list) and not all(
        not isinstance(branch, Mapping) or _can_accept_reserved_object(branch, root, set(seen)) for branch in all_of
    ):
        return False
    required = schema.get("required")
    if isinstance(required, list) and any(key != TOOL_ARGUMENT_FRAME_KEY for key in required):
        return False
    properties = schema.get("properties")
    property_schemas = properties if isinstance(properties, Mapping) else {}
    patterns = schema.get("patternProperties")
    pattern_schemas = patterns if isinstance(patterns, Mapping) else {}
    candidates: list[Mapping[str, Any]] = []
    explicit = property_schemas.get(TOOL_ARGUMENT_FRAME_KEY)
    if isinstance(explicit, Mapping):
        candidates.append(explicit)
    for pattern, pattern_schema in pattern_schemas.items():
        try:
            matches = re.search(str(pattern), TOOL_ARGUMENT_FRAME_KEY) is not None
        except re.error:
            return False
        if matches and isinstance(pattern_schema, Mapping):
            candidates.append(pattern_schema)
    if not candidates:
        additional = schema.get("additionalProperties", True)
        if additional is False:
            return False
        if isinstance(additional, Mapping):
            candidates.append(additional)
    return all(_schema_may_accept_string(candidate, root, set()) for candidate in candidates)


def _validate_unambiguous_frames(schema: Mapping[str, Any], root: Mapping[str, Any], seen: set[int]) -> None:
    identity = id(schema)
    if identity in seen:
        return
    seen.add(identity)
    ref = schema.get("$ref")
    if isinstance(ref, str):
        _validate_unambiguous_frames(_resolve_local_ref(root, ref, "$"), root, seen)
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            mappings = [branch for branch in branches if isinstance(branch, Mapping)]
            frame_branches = [branch for branch in mappings if _accepts_frame_at_root(branch, root, set())]
            if frame_branches and any(
                branch not in frame_branches and _can_accept_reserved_object(branch, root, set())
                for branch in mappings
            ):
                raise ToolArgumentCodecError(
                    "String framing collides with an object branch in the same union; "
                    "use a structurally distinct schema"
                )
    for keyword in _SCHEMA_SINGLE_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, Mapping):
            _validate_unambiguous_frames(child, root, seen)
    for keyword in _SCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, Mapping):
            for child in children.values():
                if isinstance(child, Mapping):
                    _validate_unambiguous_frames(child, root, seen)
    for keyword in _SCHEMA_LIST_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                if isinstance(child, Mapping):
                    _validate_unambiguous_frames(child, root, seen)


def _validate_reserved_schema_keys(schema: Mapping[str, Any], seen: set[int]) -> None:
    identity = id(schema)
    if identity in seen:
        return
    seen.add(identity)
    properties = schema.get("properties")
    if isinstance(properties, Mapping) and TOOL_ARGUMENT_FRAME_KEY in properties:
        raise ToolArgumentCodecError(f"Tool schemas cannot use reserved property name {TOOL_ARGUMENT_FRAME_KEY!r}")
    for keyword in _SCHEMA_SINGLE_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, Mapping):
            _validate_reserved_schema_keys(child, seen)
    for keyword in _SCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, Mapping):
            for child in children.values():
                if isinstance(child, Mapping):
                    _validate_reserved_schema_keys(child, seen)
    for keyword in _SCHEMA_LIST_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                if isinstance(child, Mapping):
                    _validate_reserved_schema_keys(child, seen)


def _schema_contains_string_declaration(
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    seen: set[int],
) -> bool:
    identity = id(schema)
    if identity in seen:
        return False
    seen.add(identity)
    if _has_direct_string_constraint(schema) or "$dynamicRef" in schema:
        return True
    ref = schema.get("$ref")
    if isinstance(ref, str):
        return _schema_contains_string_declaration(_resolve_local_ref(root, ref, "$"), root, seen)
    for keyword in (*_SCHEMA_SINGLE_KEYWORDS, *_UNSUPPORTED_VALUE_KEYWORDS):
        child = schema.get(keyword)
        if isinstance(child, Mapping) and _schema_contains_string_declaration(child, root, seen):
            return True
    for keyword in _SCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, Mapping) and any(
            isinstance(child, Mapping) and _schema_contains_string_declaration(child, root, seen)
            for child in children.values()
        ):
            return True
    for keyword in _SCHEMA_LIST_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list) and any(
            isinstance(child, Mapping) and _schema_contains_string_declaration(child, root, seen) for child in children
        ):
            return True
    return False


def _validate_supported_schema_keywords(
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    seen: set[int],
) -> None:
    identity = id(schema)
    if identity in seen:
        return
    seen.add(identity)
    if "$dynamicRef" in schema:
        raise ToolArgumentCodecError("$dynamicRef is not supported by the tool argument codec")
    for keyword in _UNSUPPORTED_VALUE_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, Mapping) and _schema_contains_string_declaration(child, root, set()):
            raise ToolArgumentCodecError(
                f"String declarations under JSON Schema keyword {keyword!r} "
                "are not supported by the tool argument codec"
            )
    for keyword in (*_SCHEMA_SINGLE_KEYWORDS, *_UNSUPPORTED_VALUE_KEYWORDS):
        child = schema.get(keyword)
        if isinstance(child, Mapping):
            _validate_supported_schema_keywords(child, root, seen)
    for keyword in _SCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, Mapping):
            for child in children.values():
                if isinstance(child, Mapping):
                    _validate_supported_schema_keywords(child, root, seen)
    for keyword in _SCHEMA_LIST_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                if isinstance(child, Mapping):
                    _validate_supported_schema_keywords(child, root, seen)


def _reject_reserved_data_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_reserved_data_keys(item, f"{path}[{index}]")
        return
    if not isinstance(value, dict):
        return
    if TOOL_ARGUMENT_FRAME_KEY in value:
        raise ToolArgumentCodecError(
            f"{path}: canonical tool data cannot use reserved property {TOOL_ARGUMENT_FRAME_KEY!r}"
        )
    for key, item in value.items():
        _reject_reserved_data_keys(item, _child_path(path, key))


def _build_internal_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    _validate_supported_schema_keywords(schema, schema, set())
    _validate_reserved_schema_keys(schema, set())
    provider_schema = _encode_schema_node(dict(schema), include_marker=True)
    _validate_unambiguous_frames(provider_schema, provider_schema, set())
    return provider_schema


def _contains_codec(schema: Mapping[str, Any], root: Mapping[str, Any], seen: set[int]) -> bool:
    identity = id(schema)
    if identity in seen:
        return False
    seen.add(identity)
    if schema.get(_SCHEMA_CODEC_KEY) == TOOL_ARGUMENT_CODEC:
        return True
    ref = schema.get("$ref")
    if isinstance(ref, str) and _contains_codec(_resolve_local_ref(root, ref, "$"), root, seen):
        return True
    for keyword in (*_SCHEMA_SINGLE_KEYWORDS,):
        child = schema.get(keyword)
        if isinstance(child, Mapping) and _contains_codec(child, root, seen):
            return True
    for keyword in (*_SCHEMA_MAP_KEYWORDS,):
        children = schema.get(keyword)
        if isinstance(children, Mapping) and any(
            isinstance(child, Mapping) and _contains_codec(child, root, seen) for child in children.values()
        ):
            return True
    for keyword in (*_SCHEMA_LIST_KEYWORDS,):
        children = schema.get(keyword)
        if isinstance(children, list) and any(
            isinstance(child, Mapping) and _contains_codec(child, root, seen) for child in children
        ):
            return True
    return False


def _transform_object(
    value: dict[str, Any],
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    encode: bool,
    active: set[tuple[int, int, bool]],
    completed_frames: set[str],
) -> dict[str, Any]:
    result = dict(value)
    properties = schema.get("properties")
    property_schemas = properties if isinstance(properties, Mapping) else {}
    pattern_properties = schema.get("patternProperties")
    pattern_schemas = pattern_properties if isinstance(pattern_properties, Mapping) else {}
    additional = schema.get("additionalProperties")
    for key, item in value.items():
        candidates: list[Mapping[str, Any]] = []
        explicit = property_schemas.get(key)
        if isinstance(explicit, Mapping):
            candidates.append(explicit)
        for pattern, pattern_schema in pattern_schemas.items():
            if isinstance(pattern_schema, Mapping) and re.search(str(pattern), key):
                candidates.append(pattern_schema)
        if not candidates and isinstance(additional, Mapping):
            candidates.append(additional)
        transformed = item
        for candidate in candidates:
            transformed = _transform_value(
                transformed,
                candidate,
                root,
                _child_path(path, key),
                encode,
                active,
                completed_frames,
            )
        result[key] = transformed
    dependent_schemas = schema.get("dependentSchemas")
    if isinstance(dependent_schemas, Mapping):
        for dependency, dependent_schema in dependent_schemas.items():
            if dependency in value and isinstance(dependent_schema, Mapping):
                result = _transform_value(
                    result,
                    dependent_schema,
                    root,
                    path,
                    encode,
                    active,
                    completed_frames,
                )
    return result


def _transform_array(
    value: list[Any],
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    encode: bool,
    active: set[tuple[int, int, bool]],
    completed_frames: set[str],
) -> list[Any]:
    result = list(value)
    prefix = schema.get("prefixItems")
    prefix_schemas = prefix if isinstance(prefix, list) else []
    items = schema.get("items")
    for index, item in enumerate(value):
        item_schema: Mapping[str, Any] | None = None
        if index < len(prefix_schemas) and isinstance(prefix_schemas[index], Mapping):
            item_schema = prefix_schemas[index]
        elif isinstance(items, Mapping):
            item_schema = items
        if item_schema is not None:
            result[index] = _transform_value(
                item,
                item_schema,
                root,
                f"{path}[{index}]",
                encode,
                active,
                completed_frames,
            )
    return result


def _transform_value(
    value: Any,
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    encode: bool,
    active: set[tuple[int, int, bool]],
    completed_frames: set[str],
) -> Any:
    token = (id(value), id(schema), encode)
    if token in active:
        return value
    active.add(token)
    try:
        ref = schema.get("$ref")
        if isinstance(ref, str):
            value = _transform_value(
                value,
                _resolve_local_ref(root, ref, path),
                root,
                path,
                encode,
                active,
                completed_frames,
            )

        if schema.get(_SCHEMA_CODEC_KEY) == TOOL_ARGUMENT_CODEC:
            if path in completed_frames:
                return value
            framed = _frame_value(value, path, encode)
            completed_frames.add(path)
            return framed

        for keyword in ("anyOf", "oneOf"):
            branches = schema.get(keyword)
            if isinstance(branches, list):
                value = _transform_union(value, branches, root, path, encode, active, completed_frames)
                break

        if isinstance(value, dict):
            value = _transform_object(value, schema, root, path, encode, active, completed_frames)
        elif isinstance(value, list):
            value = _transform_array(value, schema, root, path, encode, active, completed_frames)

        branches = schema.get("allOf")
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, Mapping) and _wire_shape_matches(value, branch, root, encode):
                    value = _transform_value(value, branch, root, path, encode, active, completed_frames)
        return value
    finally:
        active.discard(token)


def encode_tool_arguments(arguments: Any, schema: Mapping[str, Any], codec: str) -> Any:
    """Encode canonical arguments according to the provider-facing schema."""

    _require_codec(codec)
    _reject_reserved_data_keys(arguments)
    provider_schema = _build_internal_schema(schema)
    return _transform_value(arguments, provider_schema, provider_schema, "$", True, set(), set())


def decode_tool_arguments(arguments: Any, schema: Mapping[str, Any], codec: str) -> Any:
    """Strictly decode provider arguments; required frames fail closed."""

    _require_codec(codec)
    provider_schema = _build_internal_schema(schema)
    decoded = _transform_value(arguments, provider_schema, provider_schema, "$", False, set(), set())
    _reject_reserved_data_keys(decoded)
    return decoded


def decode_framed_values(value: Any, codec: str) -> Any:
    """Decode exact structural frames without a schema, for presentation."""

    _require_codec(codec)
    if isinstance(value, list):
        return [decode_framed_values(item, codec) for item in value]
    if not isinstance(value, dict):
        return value
    if TOOL_ARGUMENT_FRAME_KEY in value:
        if set(value) == {TOOL_ARGUMENT_FRAME_KEY} and isinstance(value[TOOL_ARGUMENT_FRAME_KEY], str):
            return value[TOOL_ARGUMENT_FRAME_KEY]
        raise ToolArgumentCodecError("Malformed reserved string frame in presentation data")
    return {key: decode_framed_values(item, codec) for key, item in value.items()}


def sanitize_presentation_value(value: Any) -> Any:
    """Replace Unicode surrogate code points recursively before terminal output."""

    if isinstance(value, str):
        return "".join("\ufffd" if 0xD800 <= ord(char) <= 0xDFFF else char for char in value)
    if isinstance(value, list):
        return [sanitize_presentation_value(item) for item in value]
    if isinstance(value, dict):
        return {sanitize_presentation_value(key): sanitize_presentation_value(item) for key, item in value.items()}
    return value
