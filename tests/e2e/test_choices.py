"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Choice and integer values reach tools exactly as declared in the graph.
"""

import pytest

from mdo_framework.optimization.optimizer import BayesianOptimizer
from mdo_framework.schema import (
    ChoiceVar,
    ObjectiveSpec,
    RangeVar,
    StateVar,
    StudySchema,
    ToolSpec,
)

pytestmark = pytest.mark.e2e

MINIMIZE_F = [{"name": "f", "minimize": True}]
DENSITY = {"aluminum": 0.2, "composite": 0.1}
GEARBOX_SCHEMA = StudySchema(
    variables=[
        RangeVar(name="x", lower=0.0, upper=1.0),
        ChoiceVar(name="gear", choices=[10, 20, 30]),
        RangeVar(name="n", lower=1, upper=5, value_type="int"),
        ChoiceVar(name="material", choices=["aluminum", "composite"]),
        StateVar(name="f"),
    ],
    tools=[ToolSpec(name="T", inputs=["x", "gear", "n", "material"], outputs=["f"])],
)


def choice_schema(choices: list) -> StudySchema:
    """One tool T(x, c) -> f, with x a float on [0, 1] and c a choice."""
    return StudySchema(
        variables=[
            RangeVar(name="x", lower=0.0, upper=1.0),
            ChoiceVar(name="c", choices=choices),
            StateVar(name="f"),
        ],
        tools=[ToolSpec(name="T", inputs=["x", "c"], outputs=["f"])],
    )


def gearbox(x, gear, n, material):
    steps = sum(1 for _ in range(n))  # requires a real int
    return (x - 0.3) ** 2 + 0.01 * gear + 0.1 * steps + DENSITY[material]


def assert_declared(call):
    assert call["gear"] in {10, 20, 30} and type(call["gear"]) is int
    assert type(call["n"]) is int and 1 <= call["n"] <= 5
    assert call["material"] in DENSITY


class RecordingExecutionService:
    """Evaluator without a `.problem`: forces the RemoteDiscipline path."""

    def __init__(self):
        self.received: list[dict] = []

    def evaluate(self, parameters, objectives):
        self.received.append(dict(parameters))
        return {"f": (parameters["c"] - 3) ** 2 + parameters["z"]}


@pytest.mark.parametrize(
    "choices",
    [["a", "b", "c"], [0.5, 2.0, 8.0], [1, 2, 3], [True, False]],
    ids=["str", "float", "int", "bool"],
)
def test_choice_values_round_trip(build_optimizer, recorded, choices):
    # A continuous x keeps the space from being exhausted (#74).
    offsets = dict(zip(choices, [1.0, 0.0, 2.0]))  # choices[1] is best
    tool = recorded(lambda x, c: (x - 0.8) ** 2 + offsets[c])
    optimizer, _ = build_optimizer(choice_schema(choices), {"T": tool}, MINIMIZE_F)

    result = optimizer.optimize(n_steps=3, n_init=3)

    for call in tool.calls:
        assert call["c"] in choices
        assert type(call["c"]) is type(choices[0])
    for trial in result["history"]:
        assert trial["parameters"] in tool.calls
    best = result["best_parameters"]
    assert best["c"] == choices[1]
    assert result["best_objectives"]["f"] == pytest.approx(tool.function(**best))


def test_remote_discipline_delivers_declared_numeric_choices():
    service = RecordingExecutionService()
    design_variables = [
        ChoiceVar(name="c", choices=[1, 2, 3]),
        RangeVar(name="z", lower=0.0, upper=1.0),
    ]
    result = BayesianOptimizer(
        service, design_variables, [ObjectiveSpec(name="f")]
    ).optimize(n_steps=4, n_init=4)

    received_c = [p["c"] for p in service.received]
    assert set(received_c) <= {1, 2, 3}
    # Ax may re-propose evaluated designs (GEMSEO cache hits, no service call),
    # so every history entry must match some design the service received.
    for trial in result["history"]:
        assert any(
            p["c"] == trial["parameters"]["c"]
            and p["z"] == pytest.approx(trial["parameters"]["z"])
            for p in service.received
        )
    assert result["best_parameters"]["c"] in {1, 2, 3}
    expected_f = (result["best_parameters"]["c"] - 3) ** 2 + result["best_parameters"][
        "z"
    ]
    assert result["best_objectives"]["f"] == pytest.approx(expected_f)


def test_local_optimize_and_explore_deliver_declared_values(build_optimizer, recorded):
    tool = recorded(gearbox)
    optimizer, evaluator = build_optimizer(GEARBOX_SCHEMA, {"T": tool}, MINIMIZE_F)
    result = optimizer.optimize(n_steps=3, n_init=3)

    assert tool.calls
    for call in tool.calls:
        assert_declared(call)
    for trial in result["history"]:
        assert trial["parameters"] in tool.calls

    best = result["best_parameters"]
    assert_declared(best)
    assert evaluator.evaluate(best, ["f"])["f"] == pytest.approx(
        result["best_objectives"]["f"]
    )

    tool.calls.clear()
    optimizer.explore(n_samples=4)
    assert len(tool.calls) == 4
    for call in tool.calls:
        assert_declared(call)
