from __future__ import annotations

import json

import pytest

from axio.tool_codec import (
    TOOL_ARGUMENT_CODEC,
    TOOL_ARGUMENT_FRAME_KEY,
    ToolArgumentCodecError,
    decode_tool_arguments,
    encode_tool_arguments,
    encode_tool_schema,
)


def test_schema_and_arguments_round_trip_nested_string_leaves() -> None:
    schema = {
        "type": "object",
        "properties": {
            "top": {"type": "string", "enum": ["          sentinel"]},
            "nested": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                    }
                },
            },
            "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "dynamic": {"type": "object", "additionalProperties": {"type": "string"}},
        },
    }
    arguments = {
        "top": "          sentinel",
        "nested": {"items": ["\tsentinel", 7, "sentinel          "]},
        "nullable": None,
        "dynamic": {"empty": "", "multiline": "first\n          second\n\tthird"},
    }

    wire_schema = encode_tool_schema(schema, TOOL_ARGUMENT_CODEC)
    top = wire_schema["properties"]["top"]
    assert top["type"] == "object"
    assert top["required"] == [TOOL_ARGUMENT_FRAME_KEY]
    assert top["additionalProperties"] is False
    assert "x-axio-tool-argument-codec" not in top
    assert top["properties"][TOOL_ARGUMENT_FRAME_KEY]["enum"] == ["          sentinel"]

    wire_arguments = encode_tool_arguments(arguments, schema, TOOL_ARGUMENT_CODEC)
    assert wire_arguments["top"] == {TOOL_ARGUMENT_FRAME_KEY: "          sentinel"}
    assert wire_arguments["nested"]["items"][0] == {TOOL_ARGUMENT_FRAME_KEY: "\tsentinel"}
    assert wire_arguments["nested"]["items"][1] == 7
    assert wire_arguments["dynamic"]["empty"] == {TOOL_ARGUMENT_FRAME_KEY: ""}
    assert decode_tool_arguments(wire_arguments, schema, TOOL_ARGUMENT_CODEC) == arguments


def test_literal_frame_key_content_has_no_collision() -> None:
    schema = {"type": "object", "properties": {"content": {"type": "string"}}}
    content = json.dumps({TOOL_ARGUMENT_FRAME_KEY: "literal"}) + "          "
    wire = encode_tool_arguments({"content": content}, schema, TOOL_ARGUMENT_CODEC)

    assert wire == {"content": {TOOL_ARGUMENT_FRAME_KEY: content}}
    assert decode_tool_arguments(wire, schema, TOOL_ARGUMENT_CODEC) == {"content": content}


def test_reserved_frame_key_is_rejected_as_schema_or_canonical_data() -> None:
    schema = {
        "type": "object",
        "properties": {
            "metadata": {
                "type": "object",
                "properties": {TOOL_ARGUMENT_FRAME_KEY: {"type": "string"}},
                "required": [TOOL_ARGUMENT_FRAME_KEY],
            }
        },
        "required": ["metadata"],
    }
    arguments = {"metadata": {TOOL_ARGUMENT_FRAME_KEY: "  literal  "}}

    with pytest.raises(ToolArgumentCodecError, match="reserved property name"):
        encode_tool_schema(schema, TOOL_ARGUMENT_CODEC)
    with pytest.raises(ToolArgumentCodecError, match="canonical tool data"):
        encode_tool_arguments(arguments, {"type": "object"}, TOOL_ARGUMENT_CODEC)


@pytest.mark.parametrize(
    "wire",
    [
        {"content": "unframed"},
        {"content": {TOOL_ARGUMENT_FRAME_KEY: 7}},
        {"content": {TOOL_ARGUMENT_FRAME_KEY: "ok", "extra": True}},
        {"content": {"wrong": "value"}},
    ],
)
def test_declared_string_leaf_requires_one_exact_frame(wire: dict[str, object]) -> None:
    schema = {"type": "object", "properties": {"content": {"type": "string"}}}

    with pytest.raises(ToolArgumentCodecError, match=r"\$\.content"):
        decode_tool_arguments(wire, schema, TOOL_ARGUMENT_CODEC)


def test_local_ref_and_tuple_items_round_trip() -> None:
    schema = {
        "$defs": {"verbatim": {"type": "string", "format": "uri-reference"}},
        "type": "object",
        "properties": {
            "pair": {
                "type": "array",
                "prefixItems": [{"$ref": "#/$defs/verbatim"}, {"type": "integer"}],
            }
        },
    }
    arguments = {"pair": ["  relative/path  ", 3]}

    wire_schema = encode_tool_schema(schema, TOOL_ARGUMENT_CODEC)
    inner = wire_schema["$defs"]["verbatim"]["properties"][TOOL_ARGUMENT_FRAME_KEY]
    assert inner["format"] == "uri-reference"

    wire = encode_tool_arguments(arguments, schema, TOOL_ARGUMENT_CODEC)
    assert decode_tool_arguments(wire, schema, TOOL_ARGUMENT_CODEC) == arguments


def test_anchor_and_recursive_root_refs_round_trip() -> None:
    anchored_schema = {
        "$defs": {
            "entry": {
                "$anchor": "entry",
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            }
        },
        "type": "object",
        "properties": {"entry": {"$ref": "#entry"}},
    }
    anchored_arguments = {"entry": {"label": "  anchored  "}}
    anchored_wire = encode_tool_arguments(anchored_arguments, anchored_schema, TOOL_ARGUMENT_CODEC)
    assert decode_tool_arguments(anchored_wire, anchored_schema, TOOL_ARGUMENT_CODEC) == anchored_arguments

    recursive_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "child": {"anyOf": [{"$ref": "#"}, {"type": "null"}]},
        },
        "required": ["label", "child"],
    }
    recursive_arguments = {"label": " root ", "child": {"label": " child ", "child": None}}
    recursive_wire = encode_tool_arguments(recursive_arguments, recursive_schema, TOOL_ARGUMENT_CODEC)
    assert decode_tool_arguments(recursive_wire, recursive_schema, TOOL_ARGUMENT_CODEC) == recursive_arguments

    pointer_schema = {
        "$defs": {"space name": {"type": "string"}},
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/space%20name"}},
    }
    pointer_arguments = {"value": "  pointer  "}
    pointer_wire = encode_tool_arguments(pointer_arguments, pointer_schema, TOOL_ARGUMENT_CODEC)
    assert decode_tool_arguments(pointer_wire, pointer_schema, TOOL_ARGUMENT_CODEC) == pointer_arguments


def test_anchored_nullable_and_mixed_string_unions_keep_one_identifier() -> None:
    nullable_schema = {
        "$defs": {"maybe": {"$anchor": "maybe", "type": ["string", "null"]}},
        "type": "object",
        "properties": {"value": {"$ref": "#maybe"}},
    }
    nullable_provider_schema = encode_tool_schema(nullable_schema, TOOL_ARGUMENT_CODEC)
    assert json.dumps(nullable_provider_schema).count('"$anchor"') == 1
    assert encode_tool_arguments({"value": None}, nullable_schema, TOOL_ARGUMENT_CODEC) == {"value": None}
    nullable_wire = encode_tool_arguments({"value": "  text  "}, nullable_schema, TOOL_ARGUMENT_CODEC)
    assert decode_tool_arguments(nullable_wire, nullable_schema, TOOL_ARGUMENT_CODEC) == {"value": "  text  "}

    enum_schema = {
        "$defs": {"choice": {"$anchor": "choice", "enum": [" exact ", None]}},
        "type": "object",
        "properties": {"value": {"$ref": "#choice"}},
    }
    enum_provider_schema = encode_tool_schema(enum_schema, TOOL_ARGUMENT_CODEC)
    assert json.dumps(enum_provider_schema).count('"$anchor"') == 1
    assert encode_tool_arguments({"value": None}, enum_schema, TOOL_ARGUMENT_CODEC) == {"value": None}
    enum_wire = encode_tool_arguments({"value": " exact "}, enum_schema, TOOL_ARGUMENT_CODEC)
    assert decode_tool_arguments(enum_wire, enum_schema, TOOL_ARGUMENT_CODEC) == {"value": " exact "}


def test_string_ref_with_validation_siblings_fails_before_provider_schema() -> None:
    schema = {
        "$defs": {"text": {"type": "string"}},
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/text", "type": "string"}},
    }

    with pytest.raises(ToolArgumentCodecError, match="validation siblings"):
        encode_tool_schema(schema, TOOL_ARGUMENT_CODEC)

    nullable_schema = {
        "$defs": {"text": {"type": "string"}},
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/text", "type": ["string", "null"]}},
    }
    with pytest.raises(ToolArgumentCodecError, match="validation siblings"):
        encode_tool_schema(nullable_schema, TOOL_ARGUMENT_CODEC)


def test_unknown_schema_leaves_remain_unchanged() -> None:
    schema = {"type": "object", "propertyNames": {"type": "string", "pattern": "^[a-z]+$"}}
    arguments = {"provider_owned": "  unknown  "}

    assert encode_tool_schema(schema, TOOL_ARGUMENT_CODEC)["propertyNames"] == schema["propertyNames"]
    assert encode_tool_arguments(arguments, schema, TOOL_ARGUMENT_CODEC) == arguments
    assert decode_tool_arguments(arguments, schema, TOOL_ARGUMENT_CODEC) == arguments


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "contains": {"type": "string"}},
        {"type": "object", "unevaluatedProperties": {"type": "string"}},
        {"type": "array", "unevaluatedItems": {"type": "string"}},
        {"type": "object", "if": {"properties": {"value": {"type": "string"}}}},
        {"$dynamicRef": "#value"},
    ],
)
def test_unsupported_string_value_keywords_fail_at_schema_boundary(schema: dict[str, object]) -> None:
    with pytest.raises(ToolArgumentCodecError, match="not supported"):
        encode_tool_schema(schema, TOOL_ARGUMENT_CODEC)


def test_unsupported_keyword_classification_resolves_local_refs() -> None:
    numeric_schema = {
        "$defs": {"number": {"type": "number"}},
        "type": "array",
        "contains": {"$ref": "#/$defs/number"},
    }
    assert encode_tool_schema(numeric_schema, TOOL_ARGUMENT_CODEC) == numeric_schema

    string_schema = {
        "$defs": {"text": {"type": "string"}},
        "type": "array",
        "contains": {"$ref": "#/$defs/text"},
    }
    with pytest.raises(ToolArgumentCodecError, match="not supported"):
        encode_tool_schema(string_schema, TOOL_ARGUMENT_CODEC)


def test_dependent_schema_string_leaf_round_trips() -> None:
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "integer"}},
        "dependentSchemas": {
            "mode": {
                "properties": {"content": {"type": "string"}},
            }
        },
    }
    arguments = {"mode": 1, "content": "  dependent  "}

    wire = encode_tool_arguments(arguments, schema, TOOL_ARGUMENT_CODEC)

    assert wire == {"mode": 1, "content": {TOOL_ARGUMENT_FRAME_KEY: "  dependent  "}}
    assert decode_tool_arguments(wire, schema, TOOL_ARGUMENT_CODEC) == arguments


def test_union_branch_matching_applies_dependent_schemas() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"mode": {"type": "integer"}},
                        "dependentSchemas": {
                            "mode": {
                                "properties": {"x": {"type": "integer"}},
                                "required": ["x"],
                            }
                        },
                    },
                    {
                        "type": "object",
                        "properties": {"mode": {"type": "integer"}},
                        "dependentSchemas": {
                            "mode": {
                                "properties": {"x": {"type": "string"}},
                                "required": ["x"],
                            }
                        },
                    },
                ]
            }
        },
    }
    arguments = {"value": {"mode": 1, "x": "  exact  "}}

    wire = encode_tool_arguments(arguments, schema, TOOL_ARGUMENT_CODEC)

    assert wire == {"value": {"mode": 1, "x": {TOOL_ARGUMENT_FRAME_KEY: "  exact  "}}}
    assert decode_tool_arguments(wire, schema, TOOL_ARGUMENT_CODEC) == arguments


def test_overlapping_all_of_string_paths_are_transformed_once() -> None:
    schema = {
        "type": "object",
        "allOf": [
            {"properties": {"content": {"type": "string"}}},
            {"properties": {"content": {"type": "string", "minLength": 1}}},
        ],
    }
    arguments = {"content": "  value  "}

    wire = encode_tool_arguments(arguments, schema, TOOL_ARGUMENT_CODEC)
    assert wire == {"content": {TOOL_ARGUMENT_FRAME_KEY: "  value  "}}
    assert decode_tool_arguments(wire, schema, TOOL_ARGUMENT_CODEC) == arguments


def test_type_array_and_mixed_enum_only_frame_string_variants() -> None:
    schema = {
        "type": "object",
        "properties": {
            "optional": {"type": ["string", "null"]},
            "choice": {"enum": [" literal ", 7]},
        },
    }

    string_wire = encode_tool_arguments(
        {"optional": "  value  ", "choice": " literal "},
        schema,
        TOOL_ARGUMENT_CODEC,
    )
    assert string_wire == {
        "optional": {TOOL_ARGUMENT_FRAME_KEY: "  value  "},
        "choice": {TOOL_ARGUMENT_FRAME_KEY: " literal "},
    }
    assert decode_tool_arguments(string_wire, schema, TOOL_ARGUMENT_CODEC) == {
        "optional": "  value  ",
        "choice": " literal ",
    }

    scalar_wire = encode_tool_arguments({"optional": None, "choice": 7}, schema, TOOL_ARGUMENT_CODEC)
    assert scalar_wire == {"optional": None, "choice": 7}
    assert decode_tool_arguments(scalar_wire, schema, TOOL_ARGUMENT_CODEC) == scalar_wire


def test_string_union_decodes_frame_before_broad_object_branch() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"kind": {"type": "string"}},
                        "required": ["kind"],
                        "additionalProperties": False,
                    },
                    {"type": "string"},
                ]
            }
        },
    }
    arguments = {"value": "  exact  "}

    wire = encode_tool_arguments(arguments, schema, TOOL_ARGUMENT_CODEC)

    assert wire == {"value": {TOOL_ARGUMENT_FRAME_KEY: "  exact  "}}
    assert decode_tool_arguments(wire, schema, TOOL_ARGUMENT_CODEC) == arguments


def test_union_selects_object_branch_by_nested_schema_before_transforming() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": ["x"],
                    },
                    {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                    },
                ]
            }
        },
    }
    arguments = {"value": {"x": "  exact  "}}

    wire = encode_tool_arguments(arguments, schema, TOOL_ARGUMENT_CODEC)

    assert wire == {"value": {"x": {TOOL_ARGUMENT_FRAME_KEY: "  exact  "}}}
    assert decode_tool_arguments(wire, schema, TOOL_ARGUMENT_CODEC) == arguments


def test_string_union_with_colliding_object_branch_fails_at_schema_boundary() -> None:
    schema = {
        "anyOf": [
            {"type": "object", "additionalProperties": True},
            {"type": "string"},
        ]
    }

    with pytest.raises(ToolArgumentCodecError, match="collides with an object branch"):
        encode_tool_schema(schema, TOOL_ARGUMENT_CODEC)


def test_unknown_codec_is_rejected() -> None:
    with pytest.raises(ToolArgumentCodecError, match="Unsupported"):
        encode_tool_schema({"type": "string"}, "unknown")
