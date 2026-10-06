"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

End-to-end tests (real Ax/GEMSEO) for the values tools receive.
"""

import logging
import warnings

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import services.execution.main as execution
from mdo_framework.core.components import ToolComponent, to_tool_value
from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.core.topology import TopologicalAnalyzer
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.optimization.optimizer import BayesianOptimizer
from mdo_framework.optimization.parameter_codec import ParameterValueError

FLOAT_OUT = {"param_type": "continuous", "value_type": "float"}
GEARBOX_VARIABLES = [
    {"name": "x", "lower": 0.0, "upper": 1.0, **FLOAT_OUT},
    {
        "name": "gear",
        "param_type": "choice",
        "choices": [10, 20, 30],
        "value_type": "int",
    },
    {"name": "n", "lower": 1, "upper": 5, "param_type": "range", "value_type": "int"},
    {
        "name": "material",
        "param_type": "choice",
        "choices": ["aluminum", "composite"],
        "value_type": "str",
    },
]
DENSITY = {"aluminum": 0.2, "composite": 0.1}


def gearbox_schema(names: list[str]) -> dict:
    variables = [v for v in GEARBOX_VARIABLES if v["name"] in names]
    return {
        "tools": [{"name": "T", "fidelity": "high", "inputs": names, "outputs": ["f"]}],
        "variables": variables + [{"name": "f", **FLOAT_OUT}],
    }


class RecordingTool:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, x, gear, n, material):
        self.calls.append({"x": x, "gear": gear, "n": n, "material": material})
        steps = sum(1 for _ in range(n))  # requires a real int
        return (x - 0.3) ** 2 + 0.01 * gear + 0.1 * steps + DENSITY[material]


def build_local(tool):
    names = [v["name"] for v in GEARBOX_VARIABLES]
    schema = gearbox_schema(names)
    analyzer = TopologicalAnalyzer(schema)
    parameters = analyzer.extract_parameters(analyzer.resolve_dependencies(["f"])[0])
    builder = GraphProblemBuilder(schema)
    mda = builder.build_problem({"T": tool})
    return schema, parameters, LocalEvaluator(mda, builder.variable_specs)


def assert_declared(call):
    assert call["gear"] in {10, 20, 30} and type(call["gear"]) is int
    assert type(call["n"]) is int and 1 <= call["n"] <= 5
    assert call["material"] in DENSITY


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # optimize() writes XDSM/plot files into the cwd
    logging.disable(logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    logging.disable(logging.NOTSET)


class RecordingExecutionService:
    """Evaluator without a `.problem`: forces the RemoteDiscipline path."""

    def __init__(self):
        self.received: list[dict] = []

    def evaluate(self, parameters, objectives):
        self.received.append(dict(parameters))
        return {"f": (parameters["c"] - 3) ** 2 + parameters["z"]}


def test_remote_discipline_delivers_declared_numeric_choices():
    service = RecordingExecutionService()
    parameters = [
        {"name": "c", "type": "choice", "values": [1, 2, 3], "value_type": "int"},
        {"name": "z", "type": "range", "bounds": [0.0, 1.0], "value_type": "float"},
    ]
    result = BayesianOptimizer(
        service, parameters, [{"name": "f", "minimize": True}]
    ).optimize(n_steps=4, n_init=4)

    received_c = [p["c"] for p in service.received]
    assert set(received_c) <= {1, 2, 3}
    history_c = [trial["parameters"]["c"] for trial in result["history"]]
    assert history_c == received_c[-len(history_c) :]
    assert result["best_parameters"]["c"] in {1, 2, 3}
    expected_f = (result["best_parameters"]["c"] - 3) ** 2 + result["best_parameters"][
        "z"
    ]
    assert result["best_objectives"]["f"] == pytest.approx(expected_f)


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


def test_builder_rejects_undeclared_choice_default():
    schema = gearbox_schema(["gear"])
    schema["variables"][0] = {**schema["variables"][0], "value": 15}
    with pytest.raises(ParameterValueError):
        GraphProblemBuilder(schema).build_problem({"T": lambda gear: gear})


def test_local_optimize_and_explore_deliver_declared_values():
    tool = RecordingTool()
    _, parameters, evaluator = build_local(tool)
    optimizer = BayesianOptimizer(
        evaluator, parameters, [{"name": "f", "minimize": True}]
    )
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


def test_local_and_execution_service_pass_identical_tool_inputs(monkeypatch):
    local_tool = RecordingTool()
    schema, _, evaluator = build_local(local_tool)
    remote_tool = RecordingTool()
    registry = {"T": remote_tool}
    monkeypatch.setattr(execution, "TOOL_REGISTRY", registry)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=schema))

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
