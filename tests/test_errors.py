"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import pytest

from mdo_framework.core.errors import (
    EvaluationError,
    MDANotConvergedError,
    ToolError,
    ToolExecutionError,
    ToolOutputError,
    evaluation_error_from_payload,
)


@pytest.mark.parametrize(
    ("error_class", "code"),
    [
        (EvaluationError, "EVALUATION_FAILED"),
        (ToolError, "TOOL_FAILED"),
        (ToolExecutionError, "TOOL_FAILED"),
        (ToolOutputError, "OUTPUT_INVALID"),
        (MDANotConvergedError, "MDA_NOT_CONVERGED"),
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
    assert issubclass(MDANotConvergedError, EvaluationError)
    assert not issubclass(MDANotConvergedError, ToolError)


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


@pytest.mark.parametrize(
    "error",
    [
        EvaluationError("point rejected"),
        ToolExecutionError("RuntimeError: solver diverged", tool="T"),
        ToolOutputError("non-finite values for ['f']", tool="T"),
        MDANotConvergedError("residual 1e+00 > 1e-06 (couplings: y1, y2)"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_payload_round_trip_rebuilds_the_same_error(error):
    payload = error.to_payload()

    rebuilt = evaluation_error_from_payload(payload)

    assert payload == {
        "code": error.code,
        "message": error.message,
        "tool": error.tool,
        "retryable": error.retryable,
    }
    assert type(rebuilt) is type(error)
    assert str(rebuilt) == str(error)
    assert rebuilt.tool == error.tool


def test_payload_message_has_no_tool_prefix():
    error = ToolExecutionError("KeyError: 'a'", tool="T")

    assert str(error) == "Tool 'T': KeyError: 'a'"
    assert error.to_payload()["message"] == "KeyError: 'a'"


@pytest.mark.parametrize(
    "payload",
    [
        {"code": "UNKNOWN", "message": "m", "tool": None},
        {"code": "TOOL_FAILED"},
        {"code": "TOOL_FAILED", "message": 3, "tool": None},
        {"code": "TOOL_FAILED", "message": "m", "tool": 3},
        {"message": "m"},
        "TOOL_FAILED",
        None,
    ],
)
def test_unknown_or_malformed_payload_is_not_an_evaluation_error(payload):
    assert evaluation_error_from_payload(payload) is None
