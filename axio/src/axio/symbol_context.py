"""Conservative enclosing-symbol heuristics for unified diff hunk headers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePath

_PYTHON_EXTENSIONS = frozenset({".py", ".pyi"})
_BRACE_EXTENSIONS = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".cxx",
        ".go",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".java",
        ".js",
        ".jsx",
        ".mjs",
        ".rs",
        ".ts",
        ".tsx",
    }
)
_JAVASCRIPT_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".ts", ".tsx"})
_C_FAMILY_EXTENSIONS = frozenset({".c", ".cc", ".cpp", ".cs", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".java"})

_PYTHON_SYMBOL = re.compile(
    r"^(?P<indent>[ \t]*)(?P<kind>async[ \t]+def|def|class)[ \t]+"
    r"(?P<name>[A-Za-z_]\w*)(?=[ \t]*[(:])"
)
_CLASS_SYMBOL = re.compile(r"\b(?:class|struct|interface|trait|enum)[ \t]+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)")
_GO_TYPE = re.compile(r"\btype[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]+(?:struct|interface)\b")
_RUST_IMPL_FOR = re.compile(r"\bimpl(?:[ \t]*<[^>{}]*>)?[ \t]+[^{}]*?\bfor[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_:<>]*)")
_RUST_IMPL = re.compile(r"\bimpl(?:[ \t]*<[^>{}]*>)?[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_:<>]*)")
_GO_FUNCTION = re.compile(
    r"\bfunc[ \t]+(?:\([ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]+\*?(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\[[^\]]*])?[^)]*\)[ \t]+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]*\("
)
_RUST_FUNCTION = re.compile(r"\b(?:async[ \t]+)?fn[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]*(?:<[^>{}]*>)?[ \t]*\(")
_JS_FUNCTION = re.compile(r"\b(?:async[ \t]+)?function\*?[ \t]+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)[ \t]*\(")
_JS_ARROW = re.compile(
    r"\b(?:const|let|var)[ \t]+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)"
    r"(?:[ \t]*:[^=;{}]+)?[ \t]*=[ \t]*(?:async[ \t]+)?(?:\([^(){}]*\)|[A-Za-z_$][A-Za-z0-9_$]*)"
    r"(?:[ \t]*:[^=;{}]+)?[ \t]*=>[ \t]*\{"
)
_CALLABLE_NAME = re.compile(r"(?P<name>[A-Za-z_$~][A-Za-z0-9_$~]*)[ \t]*(?:<[^;{}()]*>)?[ \t]*\(")
_CONTROL_NAMES = frozenset({"catch", "do", "else", "for", "if", "lock", "match", "switch", "while", "with"})
_MAX_SIGNATURE_LINES = 8
_MAX_SIGNATURE_CHARS = 1200
_MAX_SYMBOL_BYTES = 120


@dataclass(slots=True)
class _IndentSymbol:
    indent: int
    label: str
    signature_depth: int = 0


@dataclass(slots=True)
class _BraceSymbol:
    open_depth: int
    label: str


def line_contexts(path: str, lines: Sequence[str]) -> tuple[str | None, ...]:
    """Return the nearest enclosing symbol for each 1-based source line.

    The first tuple element is an unused sentinel. Unsupported extensions and
    constructs that cannot be identified conservatively produce ``None``.
    """
    extension = PurePath(path).suffix.lower()
    if extension in _PYTHON_EXTENSIONS:
        contexts = _python_contexts(lines)
    elif extension in _BRACE_EXTENSIONS:
        contexts = _brace_contexts(lines, extension)
    else:
        contexts = [None] * len(lines)
    return (None, *contexts)


def sanitize_symbol(value: str) -> str | None:
    """Return a bounded, terminal-safe dotted identifier or ``None``."""
    valid_parts = all(part and re.fullmatch(r"[A-Za-z_$~][A-Za-z0-9_$~:<>]*", part) for part in value.split("."))
    if not value or not valid_parts:
        return None
    encoded = value.encode("utf-8")
    if len(encoded) <= _MAX_SYMBOL_BYTES:
        return value
    clipped = encoded[:_MAX_SYMBOL_BYTES]
    while clipped:
        try:
            return clipped.decode("utf-8").rstrip(".:<>") or None
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    return None


def _python_contexts(lines: Sequence[str]) -> list[str | None]:
    contexts: list[str | None] = [None] * len(lines)
    active: list[_IndentSymbol] = []
    decorator_lines: list[int] = []
    decorator_indent: int | None = None

    for index, raw_line in enumerate(lines):
        expanded = raw_line.expandtabs(8)
        stripped = expanded.lstrip(" ")
        indent = len(expanded) - len(stripped)
        if active and active[-1].signature_depth > 0:
            contexts[index] = active[-1].label
            active[-1].signature_depth = max(
                0,
                active[-1].signature_depth + _python_delimiter_delta(expanded),
            )
            continue
        if not stripped:
            decorator_lines.clear()
            decorator_indent = None
            continue
        if stripped.startswith("#"):
            if active and indent > active[-1].indent:
                contexts[index] = active[-1].label
            continue
        while active and indent <= active[-1].indent:
            active.pop()
        if stripped.startswith("@"):
            if decorator_indent != indent:
                decorator_lines.clear()
                decorator_indent = indent
            decorator_lines.append(index)
            continue

        match = _PYTHON_SYMBOL.match(expanded)
        if match is None:
            decorator_lines.clear()
            decorator_indent = None
            if active:
                contexts[index] = active[-1].label
            continue

        name = match.group("name")
        parent = active[-1].label if active else ""
        label = sanitize_symbol(f"{parent}.{name}" if parent else name)
        decorator_start = decorator_lines if decorator_indent == indent else []
        decorator_lines = []
        decorator_indent = None
        if label is None:
            continue
        active.append(
            _IndentSymbol(
                indent=indent,
                label=label,
                signature_depth=max(0, _python_delimiter_delta(expanded)),
            )
        )
        contexts[index] = label
        for decorator_index in decorator_start:
            contexts[decorator_index] = label

    return contexts


def _brace_contexts(lines: Sequence[str], extension: str) -> list[str | None]:
    masked_lines = _mask_brace_source(lines, extension)
    contexts: list[str | None] = [None] * len(lines)
    active: list[_BraceSymbol] = []
    depth = 0

    for index, line in enumerate(masked_lines):
        while active and depth < active[-1].open_depth:
            active.pop()
        opening = line.find("{")
        candidate: str | None = None
        if opening >= 0 and line.count("{") == 1:
            signature = _signature_ending_at(masked_lines, index, opening)
            candidate = _brace_candidate(signature, extension, active[-1].label if active else None)
        if candidate is not None:
            active.append(_BraceSymbol(open_depth=depth + 1, label=candidate))
            contexts[index] = candidate
        elif active:
            contexts[index] = active[-1].label
        depth += line.count("{") - line.count("}")
        if depth < 0:
            depth = 0

    return contexts


def _python_delimiter_delta(line: str) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for character in line:
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character == "#":
            break
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
    return depth


def _signature_ending_at(lines: Sequence[str], index: int, opening: int) -> str:
    fragments = list(lines[max(0, index - _MAX_SIGNATURE_LINES + 1) : index])
    fragments.append(lines[index][: opening + 1])
    signature = " ".join(fragment.strip() for fragment in fragments)
    boundary = max(signature.rfind(";"), signature.rfind("}"), signature.rfind("{", 0, -1))
    return signature[boundary + 1 :][-_MAX_SIGNATURE_CHARS:].strip()


def _brace_candidate(signature: str, extension: str, parent: str | None) -> str | None:
    if not signature or signature.startswith("#"):
        return None
    class_match = _CLASS_SYMBOL.search(signature)
    if class_match is not None:
        return _nested_label(parent, class_match.group("name"))
    if extension == ".go":
        type_match = _GO_TYPE.search(signature)
        if type_match is not None:
            return _nested_label(parent, type_match.group("name"))
        function_match = _GO_FUNCTION.search(signature)
        if function_match is not None:
            receiver = function_match.group("receiver")
            name = function_match.group("name")
            return sanitize_symbol(f"{receiver}.{name}" if receiver else name)
        return None
    if extension == ".rs":
        impl_match = _RUST_IMPL_FOR.search(signature) or _RUST_IMPL.search(signature)
        if impl_match is not None:
            name = impl_match.group("name").split("::")[-1]
            return _nested_label(parent, name)
        function_match = _RUST_FUNCTION.search(signature)
        if function_match is not None:
            return _nested_label(parent, function_match.group("name"))
        return None
    if extension in _JAVASCRIPT_EXTENSIONS:
        for pattern in (_JS_FUNCTION, _JS_ARROW):
            function_match = pattern.search(signature)
            if function_match is not None:
                return _nested_label(parent, function_match.group("name"))
        return _c_like_callable(signature, parent, allow_bare_method=True)
    if extension in _C_FAMILY_EXTENSIONS:
        return _c_like_callable(signature, parent, allow_bare_method=parent is not None)
    return None


def _c_like_callable(signature: str, parent: str | None, *, allow_bare_method: bool) -> str | None:
    if "=>" in signature or "=" in signature.split("(", maxsplit=1)[0]:
        return None
    matches = list(_CALLABLE_NAME.finditer(signature))
    if not matches:
        return None
    match = matches[-1]
    name = match.group("name")
    if name in _CONTROL_NAMES:
        return None
    prefix = signature[: match.start()].strip()
    if not prefix and not allow_bare_method:
        return None
    if prefix.split()[-1:] and prefix.split()[-1] in _CONTROL_NAMES:
        return None
    return _nested_label(parent, name)


def _nested_label(parent: str | None, name: str) -> str | None:
    return sanitize_symbol(f"{parent}.{name}" if parent else name)


def _mask_brace_source(lines: Sequence[str], extension: str) -> list[str]:
    masked: list[str] = []
    block_depth = 0
    quote: str | None = None
    escaped = False
    preprocessor_continuation = False

    for raw_line in lines:
        if extension in _C_FAMILY_EXTENSIONS and (preprocessor_continuation or raw_line.lstrip().startswith("#")):
            preprocessor_continuation = raw_line.rstrip().endswith("\\")
            masked.append(" " * len(raw_line))
            continue
        output: list[str] = []
        index = 0
        while index < len(raw_line):
            character = raw_line[index]
            following = raw_line[index + 1] if index + 1 < len(raw_line) else ""
            if block_depth:
                if character == "/" and following == "*" and extension == ".rs":
                    block_depth += 1
                    output.extend((" ", " "))
                    index += 2
                elif character == "*" and following == "/":
                    block_depth -= 1
                    output.extend((" ", " "))
                    index += 2
                else:
                    output.append(" " if character != "\n" else "\n")
                    index += 1
                continue
            if quote is not None:
                output.append(" " if character != "\n" else "\n")
                if escaped:
                    escaped = False
                elif character == "\\" and quote != "`":
                    escaped = True
                elif character == quote:
                    quote = None
                index += 1
                continue
            if character == "/" and following == "/":
                output.extend(" " * (len(raw_line) - index))
                break
            if character == "/" and following == "*":
                block_depth = 1
                output.extend((" ", " "))
                index += 2
                continue
            if character in {'"', "'", "`"} and _starts_quote(raw_line, index, character, extension):
                quote = character
                output.append(" ")
                index += 1
                continue
            output.append(character if character >= " " or character == "\t" else " ")
            index += 1
        if quote in {'"', "'"}:
            quote = None
            escaped = False
        masked.append("".join(output))
    return masked


def _starts_quote(line: str, index: int, quote: str, extension: str) -> bool:
    if quote != "'" or extension != ".rs":
        return True
    closing = line.find("'", index + 1, min(len(line), index + 6))
    return closing > index + 1
