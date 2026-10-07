"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Objective and constraint semantics: convergence, maximization, Pareto, `>=`.
"""

import pytest

from mdo_framework.schema import RangeVar, StateVar, StudySchema, ToolSpec

pytestmark = pytest.mark.e2e

CONSTRAINED_OPTIMUM = 65 / 3  # shifted paraboloid at (-1/3, -1/3); f(x0) = 73


def line_schema(lower: float, upper: float, outputs: list[str]) -> StudySchema:
    """One tool T(x) -> outputs, with x on [lower, upper]."""
    return StudySchema(
        variables=[
            RangeVar(name="x", lower=lower, upper=upper),
            *(StateVar(name=name) for name in outputs),
        ],
        tools=[ToolSpec(name="T", inputs=["x"], outputs=outputs)],
    )


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """True if `a` is no worse than `b` everywhere and better somewhere (minimize)."""
    return all(a[k] <= b[k] for k in b) and any(a[k] < b[k] for k in b)


def test_paraboloid_constrained(shifted_paraboloid, ax_recorder):
    optimizer, tool = shifted_paraboloid()

    result = optimizer.optimize(n_steps=10, n_init=5)

    best = result["best_parameters"]
    f_best = result["best_objectives"]["f_xy"]
    assert best["x"] - best["y"] <= 1e-6
    assert f_best - CONSTRAINED_OPTIMUM <= 2.0
    assert ax_recorder.trial_kinds()[("GenerationStep_1_BoTorch", "COMPLETED")] >= 1
    assert f_best == pytest.approx(tool.function(**best)["f_xy"])


@pytest.mark.xfail(
    strict=True, reason="#34: GEMSEO negates a maximized objective to '-f'"
)
def test_maximize_sign(build_optimizer, recorded):
    tool = recorded(lambda x: -((x - 2) ** 2))
    optimizer, _ = build_optimizer(
        line_schema(-5.0, 5.0, ["f"]), {"T": tool}, [{"name": "f", "minimize": False}]
    )

    result = optimizer.optimize(n_steps=6, n_init=3)

    best = result["best_parameters"]
    assert abs(best["x"] - 2) < 0.5  # x0 = 0 is 2 away
    assert result["best_objectives"]["f"] == pytest.approx(tool.function(**best))


@pytest.mark.xfail(
    strict=True, reason="#35: multi-objective becomes vector 'f1_f2' unknown to Ax"
)
def test_moo_pareto_consistent(build_optimizer, recorded):
    tool = recorded(lambda x: {"f1": (x - 1) ** 2, "f2": (x + 1) ** 2})
    optimizer, _ = build_optimizer(
        line_schema(-2.0, 6.0, ["f1", "f2"]),
        {"T": tool},
        [{"name": "f1", "minimize": True}, {"name": "f2", "minimize": True}],
    )

    result = optimizer.optimize(n_steps=6, n_init=4)

    best = result["best_parameters"]
    expected = tool.function(**best)
    assert result["best_objectives"] == pytest.approx(expected)
    for trial in result["history"]:
        assert not dominates(trial["objectives"], expected)
    assert -1.1 <= best["x"] <= 1.1  # x0 = 2 is dominated


@pytest.mark.xfail(
    strict=True, reason="#50: '>=' constraints leak '-[c_xy-2.0]' into history"
)
def test_ge_constraint_user_names(shifted_paraboloid):
    optimizer, _ = shifted_paraboloid(">=", 2.0)  # x0 has c_xy = -6

    result = optimizer.optimize(n_steps=3, n_init=3)

    for trial in result["history"]:
        x, y = trial["parameters"]["x"], trial["parameters"]["y"]
        assert trial["objectives"]["c_xy"] == pytest.approx(x - y)
        assert not any(k.startswith("-") or "[" in k for k in trial["objectives"])
    best = result["best_parameters"]
    assert best["x"] - best["y"] >= 2.0 - 1e-6
