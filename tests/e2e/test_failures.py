"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Failing, NaN, plateaued and infeasible evaluations during a real run.
"""

import math

import pytest

pytestmark = pytest.mark.e2e

FLOAT = {"param_type": "continuous", "value_type": "float"}
MINIMIZE_F = [{"name": "f", "minimize": True}]
LINE_SCHEMA = {  # T(x) -> f with x on [0, 10]; x0 is the centre, 5
    "tools": [{"name": "T", "fidelity": "high", "inputs": ["x"], "outputs": ["f"]}],
    "variables": [
        {"name": "x", "lower": 0.0, "upper": 10.0, **FLOAT},
        {"name": "f", **FLOAT},
    ],
}


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


@pytest.mark.xfail(
    strict=True,
    reason="#46, #47: tool exceptions are abandoned instead of failed trials",
)
def test_tool_exceptions_become_failed_trials(build_optimizer, recorded, ax_recorder):
    tool = recorded(diverging)
    optimizer, _ = build_optimizer(LINE_SCHEMA, {"T": tool}, MINIMIZE_F)

    optimizer.optimize(n_steps=4, n_init=4)

    raising = sum(1 for call in tool.calls if call["x"] > 7)
    assert len(tool.calls) == 9
    assert raising >= 1
    assert ax_recorder.count("FAILED") == raising


@pytest.mark.xfail(
    strict=True, reason="#43: NaN outputs or plateaus stop the run early"
)
@pytest.mark.parametrize(
    "function", [nan_above_7, plateau_above_3], ids=["nan", "plateau"]
)
def test_nan_or_penalty_plateau_uses_full_budget(build_optimizer, recorded, function):
    tool = recorded(function)
    optimizer, _ = build_optimizer(LINE_SCHEMA, {"T": tool}, MINIMIZE_F)

    optimizer.optimize(n_steps=4, n_init=3)

    assert len(tool.calls) == 8


@pytest.mark.xfail(
    strict=True, reason="#48: a failure at the initial point x0 aborts the run"
)
def test_failure_at_initial_point(build_optimizer, recorded):
    tool = recorded(fails_near_centre)
    optimizer, _ = build_optimizer(LINE_SCHEMA, {"T": tool}, MINIMIZE_F)

    result = optimizer.optimize(n_steps=3, n_init=3)

    assert abs(result["best_parameters"]["x"] - 5) >= 0.5


@pytest.mark.xfail(
    strict=True, reason="#45: no feasible point crashes, no feasibility flag"
)
def test_infeasible_problem_is_flagged(build_optimizer, recorded):
    schema = {
        "tools": [
            {"name": "T", "fidelity": "high", "inputs": ["x"], "outputs": ["f", "g"]}
        ],
        "variables": [
            {"name": "x", "lower": -1.0, "upper": 1.0, **FLOAT},
            {"name": "f", **FLOAT},
            {"name": "g", **FLOAT},
        ],
    }
    tool = recorded(lambda x: {"f": x**2, "g": 1 + x**2})
    optimizer, _ = build_optimizer(
        schema, {"T": tool}, MINIMIZE_F, [{"name": "g", "op": "<=", "bound": 0.0}]
    )

    result = optimizer.optimize(n_steps=2, n_init=2)

    assert result["feasible"] is False
