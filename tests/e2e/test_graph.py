"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Graph shapes the optimizer must handle: coupling, unrelated tools, fixed inputs.
"""

import math

import pytest

from mdo_framework.schema import FixedParam, RangeVar, StateVar, StudySchema, ToolSpec

pytestmark = pytest.mark.e2e

# Scalar Sellar; the couplings y1, y2 deliberately have no `initial_guess`.
SELLAR_SCHEMA = StudySchema(
    variables=[
        RangeVar(name="x", lower=0.0, upper=10.0),
        RangeVar(name="z1", lower=-10.0, upper=10.0),
        RangeVar(name="z2", lower=0.0, upper=10.0),
        StateVar(name="y1"),
        StateVar(name="y2"),
        StateVar(name="obj"),
        StateVar(name="c1"),
        StateVar(name="c2"),
    ],
    tools=[
        ToolSpec(name="Sellar1", inputs=["x", "z1", "z2", "y2"], outputs=["y1"]),
        ToolSpec(name="Sellar2", inputs=["z1", "z2", "y1"], outputs=["y2"]),
        ToolSpec(
            name="System",
            inputs=["x", "z1", "z2", "y1", "y2"],
            outputs=["obj", "c1", "c2"],
        ),
    ],
)
SELLAR_REGISTRY = {
    "Sellar1": lambda x, z1, z2, y2: z1**2 + z2 + x - 0.2 * y2,
    "Sellar2": lambda z1, z2, y1: math.sqrt(abs(y1)) + z1 + z2,
    "System": lambda x, z1, z2, y1, y2: {
        "obj": x**2 + z2 + y1 + math.exp(-y2),
        "c1": 3.16 - y1,
        "c2": y2 - 24.0,
    },
}
SELLAR_X0 = {"x": 5.0, "z1": 0.0, "z2": 5.0}  # box centre


@pytest.mark.xfail(
    strict=True,
    reason="#40, #42: coupled tools fail without a 'value' on y1/y2",
)
def test_sellar_without_coupling_values(build_optimizer):
    optimizer, evaluator = build_optimizer(
        SELLAR_SCHEMA,
        SELLAR_REGISTRY,
        [{"name": "obj", "minimize": True}],
        [
            {"name": "c1", "op": "<=", "bound": 0.0},
            {"name": "c2", "op": "<=", "bound": 0.0},
        ],
    )
    coupling = evaluator.evaluate({"x": 1.0, "z1": 5.0, "z2": 2.0}, ["y1", "y2"])
    assert coupling["y1"] == pytest.approx(25.5883, rel=1e-4)
    assert coupling["y2"] == pytest.approx(12.0585, rel=1e-4)
    obj_x0 = evaluator.evaluate(SELLAR_X0, ["obj"])["obj"]

    result = optimizer.optimize(n_steps=10, n_init=5)

    best = evaluator.evaluate(result["best_parameters"], ["obj", "c1", "c2"])
    assert best["c1"] <= 1e-6 and best["c2"] <= 1e-6
    assert best["obj"] <= 0.75 * obj_x0


@pytest.mark.xfail(
    strict=True, reason="#61: every graph tool must be registered, even unused ones"
)
def test_unrelated_unregistered_tool(build_optimizer, recorded):
    schema = StudySchema(
        variables=[
            RangeVar(name="x", lower=-10.0, upper=4.0),
            RangeVar(name="y", lower=-4.0, upper=10.0),
            StateVar(name="f_xy"),
            RangeVar(name="z", lower=0.0, upper=1.0),
            StateVar(name="u"),
        ],
        tools=[
            ToolSpec(name="Paraboloid", inputs=["x", "y"], outputs=["f_xy"]),
            ToolSpec(name="Unrelated", inputs=["z"], outputs=["u"]),
        ],
    )
    tool = recorded(lambda x, y: (x - 1) ** 2 + (y + 2) ** 2)
    optimizer, _ = build_optimizer(
        schema, {"Paraboloid": tool}, [{"name": "f_xy", "minimize": True}]
    )

    result = optimizer.optimize(n_steps=2, n_init=2)

    best = result["best_parameters"]
    assert result["best_objectives"]["f_xy"] == pytest.approx(tool.function(**best))


def test_fixed_variable_not_optimized(build_optimizer, recorded):
    schema = StudySchema(
        variables=[
            RangeVar(name="v", lower=0.0, upper=10.0),
            FixedParam(name="rho", value=1.225),
            StateVar(name="f"),
        ],
        tools=[ToolSpec(name="T", inputs=["v", "rho"], outputs=["f"])],
    )
    tool = recorded(lambda v, rho: rho * (v - 7) ** 2)
    optimizer, _ = build_optimizer(
        schema, {"T": tool}, [{"name": "f", "minimize": True}]
    )

    result = optimizer.optimize(n_steps=3, n_init=3)

    assert {call["rho"] for call in tool.calls} == {1.225}
    assert set(result["best_parameters"]) == {"v"}
