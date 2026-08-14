"""Exception hierarchy for axio."""


class AxioError(Exception):
    """Base exception for all axio errors."""


class ToolError(AxioError):
    """Base for tool-related errors."""


class GuardError(ToolError):
    """Guard denied the tool call."""


class GuardCrash(GuardError):
    """A guard implementation crashed, as opposed to deliberately denying."""


class HandlerError(ToolError):
    """Expected tool failure, reported to the model."""


class HandlerCrash(HandlerError):
    """An unexpected exception escaped a tool handler."""


class StreamError(AxioError):
    """Error during stream collection."""
