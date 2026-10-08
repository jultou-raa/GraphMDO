"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Failing, NaN, plateaued and infeasible evaluations during a real run.
"""

import math

import pytest

from mdo_framework.optimization.optimizer import OptimizationExecutionError
from mdo_framework.schema import RangeVar, StateVar, StudySchema, ToolSpec

pytestmark = pytest.mark.e2e

MINIMIZE_F = [{"name": "f", "minimize": True}]
LINE_SCHEMA = StudySchema(  # T(x) -> f with x on [0, 10]; x0 is the centre, 5
    variables=[RangeVar(name="x", lower=0.0, upper=10.0), StateVar(name="f")],
    tools=[ToolSpec(name="T", inputs=["x"], outputs=["f"])],
)


def diverging(x: float) -> float:
    if x > 7:
        raise RuntimeError("solver diverged")
    return (x - 6) ** 2


def nan_above_7(x: float) -> float:
    return math.nan if x > 7 else (x - 1) ** 2


def plateau_above_3(x: float) -> float:
    return 1e6 if x > 3 else (x - 1) ** 2


def fails_near_centre(x: float) -> float:
    if abs(x - 5) < 0.5:
        raise RuntimeError("mesh generation failed")
    return (x - 8) ** 2


def always_down(x: float) -> float:
    raise RuntimeError("license server down")


def test_tool_exceptions_become_failed_trials(build_optimizer, recorded, ax_recorder):
    tool = recorded(diverging)
    optimizer, _ = build_optimizer(LINE_SCHEMA, {"T": tool}, MINIMIZE_F)

    result = optimizer.optimize(n_steps=4, n_init=4)

    raising = sum(1 for call in tool.calls if call["x"] > 7)
    assert len(tool.calls) == 8
    assert raising >= 2
    assert len({call["x"] for call in tool.calls}) == len(tool.calls)
    assert result["stop_reason"] == "budget"
    assert ax_recorder.count("FAILED") == raising
    assert result["evaluations"]["failed"] == raising
    failed = [entry for entry in result["history"] if entry["status"] == "failed"]
    assert len(failed) == raising
    assert {entry["phase"] for entry in failed} == {"init", "bo"}
    assert all("solver diverged" in entry["reason"] for entry in failed)
    assert result["best_parameters"]["x"] <= 7


@pytest.mark.parametrize(
    "function", [nan_above_7, plateau_above_3], ids=["nan", "plateau"]
)
def test_nan_or_penalty_plateau_uses_full_budget(build_optimizer, recorded, function):
    tool = recorded(function)
    optimizer, _ = build_optimizer(LINE_SCHEMA, {"T": tool}, MINIMIZE_F)

    result = optimizer.optimize(n_steps=4, n_init=3)

    assert len(tool.calls) == 7
    assert result["stop_reason"] == "budget"


def test_failure_at_initial_point(build_optimizer, recorded):
    tool = recorded(fails_near_centre)
    optimizer, _ = build_optimizer(LINE_SCHEMA, {"T": tool}, MINIMIZE_F)

    result = optimizer.optimize(n_steps=3, n_init=3, evaluate_x0=True)

    start = result["history"][0]
    assert (start["phase"], start["status"]) == ("x0", "failed")
    assert "mesh generation failed" in start["reason"]
    assert result["stop_reason"] == "budget"
    assert abs(result["best_parameters"]["x"] - 5) >= 0.5


def test_failure_limit_stops_a_run_where_nothing_completes(build_optimizer, recorded):
    tool = recorded(always_down)
    optimizer, _ = build_optimizer(LINE_SCHEMA, {"T": tool}, MINIMIZE_F)

    with pytest.raises(OptimizationExecutionError, match="No trial") as caught:
        optimizer.optimize(n_steps=5, n_init=5, max_consecutive_failures=3)

    partial = caught.value.partial_result
    assert partial["stop_reason"] == "consecutive_failures"
    assert len(tool.calls) == 3
    assert [entry["status"] for entry in partial["history"]] == ["failed"] * 3
    assert partial["best_parameters"] is None


def test_infeasible_problem_is_flagged(build_optimizer, recorded):
    schema = StudySchema(
        variables=[
            RangeVar(name="x", lower=-1.0, upper=1.0),
            StateVar(name="f"),
            StateVar(name="g"),
        ],
        tools=[ToolSpec(name="T", inputs=["x"], outputs=["f", "g"])],
    )
    tool = recorded(lambda x: {"f": x**2, "g": 1 + x**2})
    optimizer, _ = build_optimizer(
        schema, {"T": tool}, MINIMIZE_F, [{"name": "g", "op": "<=", "bound": 0.0}]
    )

    result = optimizer.optimize(n_steps=2, n_init=2)

    assert result["feasible"] is False
    assert result["constraints"]["g"]["satisfied"] is False
    assert result["constraints"]["g"]["margin"] < 0
    assert result["best_parameters"] is not None
    assert not any(entry["feasible"] for entry in result["history"])
