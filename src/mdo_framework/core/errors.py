"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import ClassVar


class EvaluationError(ValueError):
    """An evaluation point failed; the run may continue with other points.

    It subclasses ``ValueError`` because GEMSEO's parallel execution re-raises
    only ``ValueError`` subclasses.

    Attributes:
        code: Stable machine-readable code of the error class.
        retryable: Whether evaluating the same point again may succeed.
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
        self.tool = tool
        super().__init__(f"Tool '{tool}': {message}" if tool else message)


class ToolError(EvaluationError):
    """A tool could not produce valid outputs for a point."""

    code: ClassVar[str] = "TOOL_FAILED"


class ToolExecutionError(ToolError):
    """The tool function raised an exception."""

    code: ClassVar[str] = "TOOL_FAILED"


class ToolOutputError(ToolError):
    """The tool returned outputs that break the output contract."""

    code: ClassVar[str] = "OUTPUT_INVALID"
