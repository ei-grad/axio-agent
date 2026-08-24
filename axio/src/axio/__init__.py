"""axio - public API."""

from .agent import Agent, PatchLineFraming
from .blocks import TextBlock, ToolResultBlock, ToolUseBlock
from .context import ContextStore, MemoryContextStore
from .effort import EFFORT_LEVELS, EffortControl, EffortLevel, EffortMechanism, EffortRuntime, EffortState
from .events import (
    AudioOutputDelta,
    IterationEnd,
    SpeechStarted,
    SpeechStopped,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
    TranscriptDelta,
    TurnComplete,
)
from .exceptions import (
    GuardCrash,
    GuardError,
    HandlerCrash,
    HandlerError,
    ToolInputPreparationError,
    ToolProtocolError,
)
from .field import Field, FieldInfo, StrictStr
from .messages import InputAuthority, InputProvenance, Message
from .permission import ConcurrentGuard, PermissionGuard
from .realtime import RealtimeAgent
from .selector import ToolSelector
from .stream import AgentStream
from .tool import (
    CONTEXT,
    CURRENT_TOOL_CALL,
    PreparedToolInput,
    Tool,
    ToolCallContext,
    ToolInputContext,
    ToolProtocolContext,
    ToolProtocolTransition,
    with_tool_hooks,
)
from .transport import CompletionTransport, RealtimeSession, RealtimeTransport
from .types import CostSource, StopReason, Usage

__all__ = [
    # core
    "Agent",
    "PatchLineFraming",
    "Tool",
    "CONTEXT",
    "CURRENT_TOOL_CALL",
    "ToolCallContext",
    "ToolInputContext",
    "PreparedToolInput",
    "ToolProtocolContext",
    "ToolProtocolTransition",
    "with_tool_hooks",
    "ContextStore",
    "MemoryContextStore",
    "CompletionTransport",
    "EffortControl",
    "EffortLevel",
    "EffortMechanism",
    "EffortRuntime",
    "EffortState",
    "EFFORT_LEVELS",
    # events
    "StreamEvent",
    "TextDelta",
    "IterationEnd",
    "ToolUseStart",
    "ToolInputDelta",
    "ToolResult",
    # realtime
    "RealtimeAgent",
    "RealtimeTransport",
    "RealtimeSession",
    "AudioOutputDelta",
    "TranscriptDelta",
    "SpeechStarted",
    "SpeechStopped",
    "TurnComplete",
    # messages & blocks
    "Message",
    "InputProvenance",
    "InputAuthority",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    # types & errors
    "CostSource",
    "StopReason",
    "Usage",
    "GuardError",
    "GuardCrash",
    "HandlerError",
    "HandlerCrash",
    "ToolInputPreparationError",
    "ToolProtocolError",
    # permissions
    "PermissionGuard",
    "ConcurrentGuard",
    # field annotations
    "Field",
    "FieldInfo",
    "StrictStr",
    # advanced
    "ToolSelector",
    "AgentStream",
]
