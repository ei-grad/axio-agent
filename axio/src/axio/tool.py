"""Tool: frozen dataclass binding a handler callable to a name, guards, and concurrency."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from types import MappingProxyType
from typing import Any, ParamSpec, TypeVar, get_args, get_type_hints

from .background import BACKGROUND_PROPERTY
from .exceptions import (
    GuardCrash,
    GuardError,
    HandlerCrash,
    HandlerError,
    ToolInputPreparationError,
    ToolProtocolError,
)
from .field import MISSING, FieldInfo, bare_type, get_field_info
from .models import Capability
from .permission import PermissionGuard
from .schema import build_tool_schema
from .types import ToolCallID, ToolName

type JSONSchema = dict[str, Any]

logger = logging.getLogger(__name__)

# Universal opt-in argument understood by the agent loop and never passed to a
# handler; named here so the schema and the dispatcher cannot drift apart.
BACKGROUND_PARAM = "background"

# Maps JSON Schema primitive type names to Python types used for validation.
SCHEMA_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _coerce_scalar(value: str, target: type) -> Any:
    if target is bool:
        low = value.strip().lower()
        return {"true": True, "false": False}.get(low, value)
    if target in (int, float):
        try:
            return target(value)
        except ValueError:
            return value
    return value


def _coerce_model_argument(value: Any, hint: Any) -> Any:
    """Forgive the argument types models most often get wrong.

    A list arrives as '["a","b"]', a number as "10", a flag as "true". The
    schema advertises the real type, so this does not invite the mistake - it
    just avoids spending a whole turn telling the model about one. Anything that
    does not convert cleanly is left alone to fail validation as before.
    """
    if not isinstance(value, bool) and isinstance(value, str):
        targets = {bare_type(hint), *(bare_type(a) for a in get_args(hint))}
        if list in targets:
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return value
            return parsed if isinstance(parsed, list) else value
        for target in (bool, int, float):
            if target in targets:
                return _coerce_scalar(value, target)
    return value


def hint_from_json_schema(prop_schema: dict[str, Any]) -> Any:
    """Return the Python type hint for a single JSON Schema property definition."""
    t = prop_schema.get("type")
    if t is not None:
        return SCHEMA_JSON_TYPE_MAP.get(t, object)
    any_of = prop_schema.get("anyOf")
    if any_of is not None:
        non_null = [s for s in any_of if s.get("type") != "null"]
        has_null = len(non_null) < len(any_of)
        if len(non_null) == 1:
            inner = hint_from_json_schema(non_null[0])
            return (inner | None) if has_null else inner
    return object


# Set to the tool's ``context`` value before each handler invocation.
# Handlers that cannot receive context as a parameter retrieve it via ``CONTEXT.get()``.
CONTEXT: ContextVar[Any] = ContextVar("CONTEXT")


@dataclass(frozen=True, slots=True)
class ToolCallContext:
    """Identity of the agent tool call currently being executed."""

    tool_use_id: ToolCallID
    tool_name: ToolName
    iteration: int


# Set by Agent dispatch before invoking a handler. Direct Tool calls have no
# protocol-level tool-use ID and therefore leave this variable unset.
CURRENT_TOOL_CALL: ContextVar[ToolCallContext] = ContextVar("CURRENT_TOOL_CALL")


@dataclass(frozen=True, slots=True)
class ToolInputContext:
    """Immutable provider-boundary facts available to tool protocol hooks."""

    model_id: str | None = None
    model_capabilities: frozenset[Capability] = frozenset()
    argument_codec: str | None = None
    policy: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", MappingProxyType(dict(self.policy)))


@dataclass(frozen=True, slots=True)
class PreparedToolInput:
    """Canonical handler kwargs plus an opaque persisted preparation mode."""

    input: dict[str, Any]
    mode: str = "canonical"


@dataclass(frozen=True, slots=True)
class ToolProtocolContext:
    """Request facts and visible historical preparation modes for one tool."""

    request: ToolInputContext
    prior_input_preparations: Mapping[str | None, int]
    latest_state: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prior_input_preparations",
            MappingProxyType(dict(self.prior_input_preparations)),
        )


@dataclass(frozen=True, slots=True)
class ToolProtocolTransition:
    """A stable protocol state and optional chronological model-facing notice."""

    state_id: str
    text: str | None


type ToolInputPreparer = Callable[[dict[str, Any], ToolInputContext], PreparedToolInput]
type ToolProtocolTransitionProvider = Callable[[ToolProtocolContext], ToolProtocolTransition]

P = ParamSpec("P")
R = TypeVar("R")


def with_tool_hooks(
    *,
    input_preparer: ToolInputPreparer | None = None,
    protocol_transition: ToolProtocolTransitionProvider | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Attach hooks to a handler so entry-point-created Tools inherit them."""

    def decorate(handler: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        setattr(handler, "_tool_input_preparer", input_preparer)
        setattr(handler, "_tool_protocol_transition", protocol_transition)
        return handler

    return decorate


def _default_format_stream_result(chunks: list[tuple[float, str, str]]) -> str:
    """Default streaming-aggregator: join all text, discard keys/timestamps."""
    return "".join(text for _, _, text in chunks)


@dataclass(frozen=True, slots=True)
class Tool[T]:
    name: ToolName
    handler: Callable[..., Awaitable[Any]]
    description: str = ""
    guards: tuple[PermissionGuard, ...] = ()
    concurrency: int | None = None
    input_preparer: ToolInputPreparer | None = field(default=None, repr=False, compare=False, kw_only=True)
    protocol_transition: ToolProtocolTransitionProvider | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )

    context: T = field(default=MappingProxyType({}), compare=False)  # type: ignore[assignment]
    schema: MappingProxyType[str, Any] = field(default=MappingProxyType({}), repr=False, compare=False)
    detachable: bool = True
    _semaphore: asyncio.Semaphore | None = field(init=False, default=None, repr=False, compare=False)
    _fields: Mapping[str, tuple[Any, FieldInfo]] = field(
        init=False, repr=False, compare=False, default_factory=lambda: MappingProxyType({})
    )
    _accepts_var_kwargs: bool = field(init=False, default=False, repr=False, compare=False)
    _schema_explicit: bool = field(init=False, default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not inspect.iscoroutinefunction(self.handler):
            raise TypeError(
                f"Tool {self.name!r} handler {self.handler!r} must be an async function (coroutinefunction)."
            )
        if not self.description:
            object.__setattr__(self, "description", self.handler.__doc__ or "")
        if self.input_preparer is None:
            inherited_preparer = getattr(self.handler, "_tool_input_preparer", None)
            if inherited_preparer is not None:
                if not callable(inherited_preparer):
                    raise TypeError(f"Tool {self.name!r} handler input preparer must be callable")
                object.__setattr__(self, "input_preparer", inherited_preparer)
        if self.protocol_transition is None:
            inherited_transition = getattr(self.handler, "_tool_protocol_transition", None)
            if inherited_transition is not None:
                if not callable(inherited_transition):
                    raise TypeError(f"Tool {self.name!r} handler protocol transition must be callable")
                object.__setattr__(self, "protocol_transition", inherited_transition)
        hints = get_type_hints(self.handler, include_extras=True)
        param_hints = {k: v for k, v in hints.items() if k != "return"}
        try:
            sig = inspect.signature(self.handler)
        except (ValueError, TypeError):
            sig = None
        fields: dict[str, tuple[Any, FieldInfo]] = {}
        for name, hint in param_hints.items():
            fi = get_field_info(hint) or FieldInfo()
            if sig is not None and name in sig.parameters:
                param = sig.parameters[name]
                if param.default is not inspect.Parameter.empty and fi.default is MISSING:
                    # Merge sig default into FieldInfo (covers StrictStr and plain defaults).
                    fi = dc_replace(fi, default=param.default)
            fields[name] = (hint, fi)
        param_fields = MappingProxyType(fields)
        accepts_var_kwargs = sig is not None and any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        schema_explicit = bool(self.schema)
        if not self.schema:
            object.__setattr__(self, "schema", MappingProxyType(build_tool_schema(self.handler, hints=param_hints)))
        # For handlers with an explicit schema: synthesise _fields from schema properties
        # so type validation, default injection, and kwarg filtering all use the schema.
        if schema_explicit:
            schema_props: dict[str, Any] = dict(self.schema).get("properties") or {}
            schema_fields: dict[str, tuple[Any, FieldInfo]] = {}
            for prop_name, prop_schema in schema_props.items():
                hint = hint_from_json_schema(prop_schema)
                default = prop_schema.get("default", MISSING)
                schema_fields[prop_name] = (hint, FieldInfo(default=default))
            if schema_fields:
                param_fields = MappingProxyType(schema_fields)
        object.__setattr__(self, "_fields", param_fields)
        object.__setattr__(self, "_accepts_var_kwargs", accepts_var_kwargs)
        object.__setattr__(self, "_schema_explicit", schema_explicit)
        if self.concurrency is not None:
            object.__setattr__(self, "_semaphore", asyncio.Semaphore(self.concurrency))

    @asynccontextmanager
    async def _acquire(self) -> AsyncGenerator[None, None]:
        if self._semaphore is None:
            yield
            return
        async with self._semaphore:
            yield

    @property
    def input_schema(self) -> JSONSchema:
        schema = copy.deepcopy(dict(self.schema))
        properties = schema.setdefault("properties", {})
        if isinstance(properties, dict):
            if self.detachable and BACKGROUND_PARAM not in properties:
                properties[BACKGROUND_PARAM] = copy.deepcopy(BACKGROUND_PROPERTY)
            elif not self.detachable:
                properties.pop(BACKGROUND_PARAM, None)
                required = schema.get("required")
                if isinstance(required, list):
                    schema["required"] = [name for name in required if name != BACKGROUND_PARAM]
        return schema

    @property
    def supports_streaming(self) -> bool:
        """Handler supports streaming if it exposes a ``.stream`` async-generator attribute."""
        return callable(getattr(self.handler, "stream", None))

    def format_stream_result(self, chunks: list[tuple[float, str, str]]) -> str:
        """Aggregate streamed chunks into the final tool result string.

        Handlers may attach a ``format_stream_result`` callable for structured
        output (e.g. shell log records). Defaults to text concatenation.
        """
        fn = getattr(self.handler, "format_stream_result", None)
        if callable(fn):
            result = fn(chunks)
            return result if isinstance(result, str) else str(result)
        return _default_format_stream_result(chunks)

    def prepare_input(
        self,
        input: dict[str, Any],
        context: ToolInputContext | None = None,
    ) -> PreparedToolInput:
        """Prepare raw semantic input exactly once before validation and persistence."""

        if self.input_preparer is None:
            return PreparedToolInput(input=dict(input))
        try:
            prepared = self.input_preparer(dict(input), context or ToolInputContext())
        except ToolInputPreparationError:
            raise
        except Exception as exc:
            # Third-party hook callables may raise arbitrary implementation exceptions.
            raise ToolInputPreparationError(f"{type(exc).__name__}: {exc}") from exc
        try:
            if not isinstance(prepared, PreparedToolInput):
                raise ToolInputPreparationError("input preparer must return PreparedToolInput")
            if not isinstance(prepared.input, Mapping):
                raise ToolInputPreparationError("prepared input must be a mapping")
            if any(not isinstance(key, str) for key in prepared.input):
                raise ToolInputPreparationError("prepared input keys must be strings")
            if not isinstance(prepared.mode, str) or not prepared.mode:
                raise ToolInputPreparationError("prepared input mode must be a non-empty string")
            normalized = dict(prepared.input)
        except ToolInputPreparationError:
            raise
        except Exception as exc:
            # User-defined Mapping implementations may fail while being copied.
            raise ToolInputPreparationError(f"Invalid prepared input: {type(exc).__name__}: {exc}") from exc
        return PreparedToolInput(input=normalized, mode=prepared.mode)

    def resolve_protocol_transition(self, context: ToolProtocolContext) -> ToolProtocolTransition | None:
        """Resolve one bounded protocol state without mutating the Tool or history."""

        if self.protocol_transition is None:
            return None
        try:
            transition = self.protocol_transition(context)
        except ToolProtocolError:
            raise
        except Exception as exc:
            # Third-party hook callables may raise arbitrary implementation exceptions.
            raise ToolProtocolError(f"Tool {self.name!r} protocol hook failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(transition, ToolProtocolTransition):
            raise ToolProtocolError(f"Tool {self.name!r} protocol hook must return ToolProtocolTransition")
        if not isinstance(transition.state_id, str) or not transition.state_id or len(transition.state_id) > 256:
            raise ToolProtocolError(f"Tool {self.name!r} protocol state must contain 1..256 characters")
        if transition.text is not None and (
            not isinstance(transition.text, str) or not transition.text or len(transition.text) > 4096
        ):
            raise ToolProtocolError(f"Tool {self.name!r} protocol text must contain 1..4096 characters")
        return transition

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Inject defaults, validate types, and strip extras per schema."""
        required_set: set[str] = set(self.schema.get("required", []))
        for name, (hint, fi) in self._fields.items():
            if name not in kwargs:
                if fi.default is not MISSING and name not in required_set:
                    kwargs[name] = fi.default
            else:
                kwargs[name] = _coerce_model_argument(kwargs[name], hint)
                fi.validate(kwargs[name], name, hint)
        missing = [name for name in required_set if name not in kwargs]
        if missing:
            raise HandlerError(f"Missing required field(s): {', '.join(missing)}")
        if self._schema_explicit:
            schema_props = self.schema.get("properties")
            if schema_props is not None:
                kwargs = {k: v for k, v in kwargs.items() if k in schema_props}
        elif not self._accepts_var_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in self._fields}
        return kwargs

    async def __call__(self, **kwargs: Any) -> Any:
        prepared = self.prepare_input(kwargs)
        return await self.invoke_prepared(**prepared.input)

    async def invoke_prepared(self, **kwargs: Any) -> Any:
        """Validate, guard, and invoke kwargs already prepared for this Tool."""

        async with self._acquire():
            try:
                kwargs = self._prepare_kwargs(kwargs)
            except HandlerError:
                raise
            except Exception as exc:
                raise HandlerError(str(exc)) from exc

            for guard in self.guards:
                try:
                    kwargs = await guard(self, **kwargs)
                except GuardError:
                    raise
                except Exception as exc:
                    raise GuardCrash(f"{type(exc).__name__}: {exc}") from exc

            try:
                if self._schema_explicit:
                    schema_props = self.schema.get("properties")
                    if schema_props is not None:
                        kwargs = {k: v for k, v in kwargs.items() if k in schema_props}
                elif not self._accepts_var_kwargs:
                    kwargs = {k: v for k, v in kwargs.items() if k in self._fields}
                token = CONTEXT.set(self.context)
                try:
                    return await self.handler(**kwargs)
                finally:
                    CONTEXT.reset(token)
            except HandlerError:
                raise
            except Exception as exc:
                raise HandlerCrash(f"{type(exc).__name__}: {exc}") from exc

    async def call_streaming(self, **kwargs: Any) -> AsyncGenerator[tuple[str, str], None]:
        """Execute handler, yielding ``(key, text)`` chunks for streaming output.

        Uses ``handler.stream(**kwargs)`` if the handler exposes one. Otherwise
        falls back to ``__call__()`` and yields the full result as a single
        ``("output", ...)`` chunk. Semaphore is held for the entire iteration.
        """
        prepared = self.prepare_input(kwargs)
        async for chunk in self.call_streaming_prepared(**prepared.input):
            yield chunk

    async def call_streaming_prepared(self, **kwargs: Any) -> AsyncGenerator[tuple[str, str], None]:
        """Stream kwargs already prepared for this Tool."""

        async with self._acquire():
            try:
                kwargs = self._prepare_kwargs(kwargs)
            except HandlerError:
                raise
            except Exception as exc:
                raise HandlerError(str(exc)) from exc

            for guard in self.guards:
                try:
                    kwargs = await guard(self, **kwargs)
                except GuardError:
                    raise
                except Exception as exc:
                    raise GuardCrash(f"{type(exc).__name__}: {exc}") from exc

            if self._schema_explicit:
                schema_props = self.schema.get("properties")
                if schema_props is not None:
                    kwargs = {k: v for k, v in kwargs.items() if k in schema_props}
            elif not self._accepts_var_kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in self._fields}

            stream_fn = getattr(self.handler, "stream", None)
            token = CONTEXT.set(self.context)
            try:
                if callable(stream_fn):
                    async for chunk in stream_fn(**kwargs):
                        yield chunk
                else:
                    result = await self.handler(**kwargs)
                    if isinstance(result, str):
                        yield ("output", result)
                    else:
                        yield ("output", str(result))
            except HandlerError:
                raise
            except Exception as exc:
                raise HandlerCrash(f"{type(exc).__name__}: {exc}") from exc
            finally:
                CONTEXT.reset(token)
