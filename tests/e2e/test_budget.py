"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Real Ax/GEMSEO tests for the optimization budget contract:
total evaluations = n_init (Sobol) + n_steps (BoTorch), plus the start point x0
when it is evaluated.
"""

import pytest

from mdo_framework.optimization.optimizer import OptimizationConfigurationError
from mdo_framework.schema import ChoiceVar, RangeVar, StateVar, StudySchema, ToolSpec

pytestmark = pytest.mark.e2e

PARABOLOID_SCHEMA = StudySchema(  # main.py demo
    variables=[
        RangeVar(name="x", lower=-10.0, upper=10.0),
        RangeVar(name="y", lower=-10.0, upper=10.0),
        StateVar(name="f_xy"),
        StateVar(name="c_xy"),
    ],
    tools=[ToolSpec(name="Paraboloid", inputs=["x", "y"], outputs=["f_xy", "c_xy"])],
)
DISCRETE_SCHEMA = StudySchema(
    variables=[ChoiceVar(name="m", choices=["a", "b", "c"]), StateVar(name="f")],
    tools=[ToolSpec(name="T", inputs=["m"], outputs=["f"])],
)
MINIMIZE_F_XY = [{"name": "f_xy", "minimize": True}]


@pytest.mark.parametrize(
    ("n_init", "n_steps", "evaluate_x0"), [(5, 5, True), (2, 3, False), (3, 4, None)]
)
def test_n_steps_counts_bo_iterations(
    build_optimizer, recorded, ax_recorder, n_init, n_steps, evaluate_x0
):
    tool = recorded(
        lambda x, y: {"f_xy": (x - 3) ** 2 + x * y + (y + 4) ** 2 - 3, "c_xy": x - y}
    )
    optimizer, _ = build_optimizer(
        PARABOLOID_SCHEMA,
        {"Paraboloid": tool},
        MINIMIZE_F_XY,
        [{"name": "c_xy", "op": "<=", "bound": 0.0}],
    )
    result = optimizer.optimize(n_steps=n_steps, n_init=n_init, evaluate_x0=evaluate_x0)

    x0 = 1 if evaluate_x0 else 0
    assert len(tool.calls) == x0 + n_init + n_steps
    assert result["evaluations"] == {
        "x0": x0,
        "init": n_init,
        "bo": n_steps,
        "failed": 0,
    }
    expected_kinds = {
        ("GenerationStep_0_Sobol", "COMPLETED"): n_init,
        ("GenerationStep_1_BoTorch", "COMPLETED"): n_steps,
    }
    if x0:
        expected_kinds[("x0", "COMPLETED")] = 1
    assert ax_recorder.trial_kinds() == expected_kinds
    assert result["stop_reason"] == "budget"


def test_declared_initial_values_enable_the_start_point(build_optimizer, recorded):
    schema = StudySchema(
        variables=[
            RangeVar(name="x", lower=-10.0, upper=10.0, initial=2.0),
            RangeVar(name="y", lower=-10.0, upper=10.0, initial=-3.0),
            StateVar(name="f_xy"),
            StateVar(name="c_xy"),
        ],
        tools=[
            ToolSpec(name="Paraboloid", inputs=["x", "y"], outputs=["f_xy", "c_xy"])
        ],
    )
    tool = recorded(lambda x, y: {"f_xy": x**2 + y**2, "c_xy": x - y})
    optimizer, _ = build_optimizer(schema, {"Paraboloid": tool}, MINIMIZE_F_XY)

    result = optimizer.optimize(n_steps=2, n_init=2)

    assert tool.calls[0] == {"x": 2.0, "y": -3.0}
    assert len(tool.calls) == 1 + 2 + 2
    assert result["history"][0]["phase"] == "x0"


def test_exhausted_discrete_space_terminates_and_reports_why(
    build_optimizer, ax_recorder
):
    calls = []

    def tool(m):
        calls.append(m)
        return {"a": 1.0, "b": 0.5, "c": 2.0}[m]

    optimizer, _ = build_optimizer(
        DISCRETE_SCHEMA, {"T": tool}, [{"name": "f", "minimize": True}]
    )
    result = optimizer.optimize(n_steps=50, n_init=2)

    assert set(calls) <= {"a", "b", "c"}
    assert len(calls) == len(set(calls))
    assert result["best_parameters"] == {"m": "b"}
    assert result["stop_reason"] == "search_space_exhausted"
    statuses = {status for _, status in ax_recorder.trial_kinds()}
    assert statuses == {"COMPLETED"}


@pytest.mark.parametrize(("n_init", "n_steps"), [(0, 5), (5, 0), (-1, 5), (5, -2)])
def test_invalid_budget_is_rejected_before_any_tool_call(
    build_optimizer, n_init, n_steps
):
    calls = []

    def paraboloid(x, y):
        calls.append((x, y))
        return {"f_xy": x + y, "c_xy": x - y}

    optimizer, _ = build_optimizer(
        PARABOLOID_SCHEMA, {"Paraboloid": paraboloid}, MINIMIZE_F_XY
    )
    with pytest.raises(OptimizationConfigurationError, match="must be >= 1"):
        optimizer.optimize(n_steps=n_steps, n_init=n_init)
    assert calls == []
