"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import pytest

from mdo_framework.core.errors import (
    EvaluationError,
    ToolError,
    ToolExecutionError,
    ToolOutputError,
)


@pytest.mark.parametrize(
    ("error_class", "code"),
    [
        (EvaluationError, "EVALUATION_FAILED"),
        (ToolError, "TOOL_FAILED"),
        (ToolExecutionError, "TOOL_FAILED"),
        (ToolOutputError, "OUTPUT_INVALID"),
    ],
)
def test_error_codes_are_stable(error_class, code):
    assert error_class.code == code
    assert error_class.retryable is False


def test_taxonomy_hierarchy():
    assert issubclass(ToolError, EvaluationError)
    assert issubclass(ToolExecutionError, ToolError)
    assert issubclass(ToolOutputError, ToolError)
    assert not issubclass(ToolOutputError, ToolExecutionError)


def test_errors_are_value_errors_so_gemseo_parallel_execution_reraises_them():
    assert issubclass(EvaluationError, ValueError)


def test_message_is_prefixed_with_the_tool_name():
    error = ToolExecutionError("RuntimeError: solver diverged", tool="T")

    assert error.tool == "T"
    assert str(error) == "Tool 'T': RuntimeError: solver diverged"


def test_message_is_unchanged_without_a_tool():
    error = EvaluationError("point failed")

    assert error.tool is None
    assert str(error) == "point failed"
