"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections.abc import Mapping
from typing import Any, ClassVar


class EvaluationError(ValueError):
    """An evaluation point failed; the run may continue with other points.

    It subclasses ``ValueError`` because GEMSEO's parallel execution re-raises
    only ``ValueError`` subclasses.

    Attributes:
        code: Stable machine-readable code of the error class.
        retryable: Whether evaluating the same point again may succeed.
        message: What went wrong, without the tool prefix.
        tool: Name of the tool that failed, if any.
    """

    code: ClassVar[str] = "EVALUATION_FAILED"
    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, tool: str | None = None) -> None:
        """Build the error.

        Args:
            message: What went wrong.
            tool: Name of the tool that failed, prefixed to the message.
        """
        self.message = message
        self.tool = tool
        super().__init__(f"Tool '{tool}': {message}" if tool else message)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the error for a service response.

        Returns:
            The ``code``, ``message``, ``tool`` and ``retryable`` of the error;
            ``evaluation_error_from_payload`` rebuilds the error from it.
        """
        return {
            "code": self.code,
            "message": self.message,
            "tool": self.tool,
            "retryable": self.retryable,
        }


class ToolError(EvaluationError):
    """A tool could not produce valid outputs for a point."""

    code: ClassVar[str] = "TOOL_FAILED"


class ToolExecutionError(ToolError):
    """The tool function raised an exception."""

    code: ClassVar[str] = "TOOL_FAILED"


class ToolOutputError(ToolError):
    """The tool returned outputs that break the output contract."""

    code: ClassVar[str] = "OUTPUT_INVALID"


class InfeasiblePointError(ToolError):
    """A tool declares that it cannot compute this point, e.g. a failed mesh.

    Tools raise it themselves to report a point outside their domain of
    validity, which is a point failure rather than a tool crash.
    """

    code: ClassVar[str] = "POINT_INFEASIBLE"


class MDANotConvergedError(EvaluationError):
    """The coupled tools did not reach a fixed point within the MDA limits."""

    code: ClassVar[str] = "MDA_NOT_CONVERGED"


# ToolError shares TOOL_FAILED with ToolExecutionError, the class actually raised.
_ERRORS_BY_CODE: dict[str, type[EvaluationError]] = {
    error_class.code: error_class
    for error_class in (
        EvaluationError,
        ToolExecutionError,
        ToolOutputError,
        InfeasiblePointError,
        MDANotConvergedError,
    )
}


def evaluation_error_from_payload(payload: Any) -> EvaluationError | None:
    """Rebuild the error a service serialized with ``EvaluationError.to_payload``.

    Args:
        payload: The decoded ``detail`` of a service error response.

    Returns:
        The error with the same class, message and tool, or ``None`` if the
        payload is not a serialized evaluation error: unknown ``code``, wrong
        field types, or a ``retryable`` flag that does not match the class.
    """
    if not isinstance(payload, Mapping):
        return None
    code = payload.get("code")
    message = payload.get("message")
    tool = payload.get("tool")
    retryable = payload.get("retryable")
    if not isinstance(code, str) or not isinstance(message, str):
        return None
    error_class = _ERRORS_BY_CODE.get(code)
    if error_class is None or retryable is not error_class.retryable:
        return None
    if tool is not None and not isinstance(tool, str):
        return None
    return error_class(message, tool=tool)
