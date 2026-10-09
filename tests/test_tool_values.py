"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Tests for the values tools receive (codec, component, Execution Service).
"""

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import services.execution.main as execution
from mdo_framework.core.components import ToolComponent, to_tool_value
from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.optimization.parameter_codec import ParameterValueError
from mdo_framework.schema import (
    ChoiceVar,
    RangeVar,
    StateVar,
    StudySchema,
    ToolSpec,
)

GEARBOX_SCHEMA = StudySchema(
    variables=[
        RangeVar(name="x", lower=0.0, upper=1.0),
        ChoiceVar(name="gear", choices=[10, 20, 30]),
        RangeVar(name="n", lower=1, upper=5, value_type="int"),
        ChoiceVar(name="material", choices=["aluminum", "composite"]),
        StateVar(name="f"),
    ],
    tools=[
        ToolSpec(name="T", inputs=["x", "gear", "n", "material"], outputs=["f"]),
    ],
)
DENSITY = {"aluminum": 0.2, "composite": 0.1}


class RecordingTool:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, x, gear, n, material):
        self.calls.append({"x": x, "gear": gear, "n": n, "material": material})
        steps = sum(1 for _ in range(n))  # requires a real int
        return (x - 0.3) ** 2 + 0.01 * gear + 0.1 * steps + DENSITY[material]


def build_local(tool):
    builder = GraphProblemBuilder(GEARBOX_SCHEMA)
    mda = builder.build_problem({"T": tool})
    return GEARBOX_SCHEMA, LocalEvaluator(mda, builder.variable_specs)


def test_to_tool_value_decodes_specs():
    choice = {"name": "g", "type": "choice", "values": [10, 20, 30]}
    strings = {"name": "m", "type": "choice", "values": ["a", "b"]}
    integer = {"name": "n", "type": "range", "value_type": "int"}
    assert to_tool_value(choice, np.array([2.0])) == 30
    assert to_tool_value(strings, np.array([1])) == "b"
    value = to_tool_value(integer, np.array([2.6]))
    assert value == 3 and type(value) is int
    assert to_tool_value(None, np.array([1.5])) == 1.5
    vector = np.array([1.0, 2.0])
    assert to_tool_value(integer, vector) is vector
    with pytest.raises(ParameterValueError):
        to_tool_value(choice, np.array([3.0]))


def test_tool_component_receives_declared_values():
    received = {}

    def tool(gear, material, n):
        received.update(gear=gear, material=material, n=n)
        return 1.0

    specs = {
        "gear": {"name": "gear", "type": "choice", "values": [10, 20, 30]},
        "material": {"name": "material", "type": "choice", "values": ["al", "cf"]},
        "n": {"name": "n", "type": "range", "value_type": "int"},
    }
    component = ToolComponent("T", tool, ["gear", "material", "n"], ["f"], specs=specs)
    component.execute(
        {"gear": np.array([1.0]), "material": np.array([1.0]), "n": np.array([4.0])}
    )
    assert received == {"gear": 20, "material": "cf", "n": 4}
    assert type(received["n"]) is int


def test_choice_initial_must_be_declared_and_is_not_a_default():
    with pytest.raises(ValidationError, match="initial"):
        ChoiceVar(name="gear", choices=[10, 20, 30], initial=15)

    schema = StudySchema(
        variables=[
            ChoiceVar(name="gear", choices=[10, 20, 30], initial=20),
            StateVar(name="f"),
        ],
        tools=[ToolSpec(name="T", inputs=["gear"], outputs=["f"])],
    )
    mda = GraphProblemBuilder(schema).build_problem({"T": lambda gear: gear})
    assert "gear" not in mda.default_input_data


def test_local_and_execution_service_pass_identical_tool_inputs(monkeypatch):
    local_tool = RecordingTool()
    schema, evaluator = build_local(local_tool)
    remote_tool = RecordingTool()
    registry = {"T": remote_tool}
    monkeypatch.setattr(execution, "TOOL_REGISTRY", registry)
    payload = schema.model_dump(mode="json")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))

    point = {"x": 0.25, "gear": 30, "n": 4, "material": "composite"}
    local_f = evaluator.evaluate(point, ["f"])["f"]

    with TestClient(execution.app) as client:
        execution.app.state.schema_provider = execution.SchemaProvider(
            httpx.AsyncClient(transport=transport)
        )
        execution.app.state.problem_pool = execution.ProblemPool(registry, size=1)
        response = client.post("/evaluate", json={"inputs": point, "objectives": ["f"]})
        invalid = client.post(
            "/evaluate",
            json={"inputs": {**point, "material": "steel"}, "objectives": ["f"]},
        )

    assert response.status_code == 200, response.text
    assert response.json()["results"]["f"] == pytest.approx(local_f)
    assert local_tool.calls == remote_tool.calls == [point]
    assert [type(v) for v in remote_tool.calls[0].values()] == [
        type(v) for v in point.values()
    ]
    assert invalid.status_code == 400
