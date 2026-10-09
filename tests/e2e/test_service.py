"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Un-mocked Optimization Service tests: real BayesianOptimizer, RemoteEvaluator
and Execution Service app. Only the graph schema is served by a mock transport.
"""

import functools

import httpx
import pytest
from fastapi.testclient import TestClient

import services.execution.main as execution
import services.optimization.main as optimization
from mdo_framework.optimization.optimizer import OptimizationConfigurationError
from mdo_framework.schema import RangeVar, StateVar, StudySchema, ToolSpec

pytestmark = pytest.mark.e2e

# Documented walkthrough (docs/user-guide/running-optimization.md)
SCHEMA = StudySchema(
    variables=[
        RangeVar(name="x", lower=0.0, upper=10.0),
        RangeVar(name="y", lower=0.0, upper=10.0),
        StateVar(name="f_xy"),
        StateVar(name="c_xy"),
    ],
    tools=[ToolSpec(name="Paraboloid", inputs=["x", "y"], outputs=["f_xy", "c_xy"])],
)
DOCUMENTED_PAYLOAD = {"objectives": [{"name": "f_xy", "minimize": True}]}


@pytest.fixture
def services(monkeypatch):
    payload = SCHEMA.model_dump(mode="json")
    graph = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    with (
        TestClient(execution.app) as execution_client,
        TestClient(optimization.app) as optimization_client,
    ):
        execution.app.state.schema_provider = execution.SchemaProvider(
            httpx.AsyncClient(transport=graph)
        )
        execution.app.state.problem_pool = execution.ProblemPool(
            execution.TOOL_REGISTRY, size=1
        )
        optimization.app.state.client = httpx.AsyncClient(transport=graph)
        real_remote_evaluator = optimization.RemoteEvaluator
        monkeypatch.setattr(
            optimization,
            "RemoteEvaluator",
            lambda url: real_remote_evaluator(url, client=execution_client),
        )
        yield optimization_client


def test_optimize_service_end_to_end(services):
    response = services.post(
        "/optimize", json={**DOCUMENTED_PAYLOAD, "n_init": 2, "n_steps": 2}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {
        "best_parameters",
        "best_objectives",
        "feasible",
        "constraints",
        "pareto_front",
        "history",
        "stop_reason",
        "evaluations",
    }
    assert "f_xy" in body["best_objectives"]
    assert set(body["best_parameters"]) == {"x", "y"}
    assert body["feasible"] is True
    assert body["pareto_front"] == []
    assert body["stop_reason"] == "budget"
    assert body["evaluations"] == {"x0": 0, "init": 2, "bo": 2, "failed": 0}
    assert [entry["status"] for entry in body["history"]] == ["completed"] * 4


def test_evaluate_x0_and_failure_limit_are_forwarded(services):
    response = services.post(
        "/optimize",
        json={
            **DOCUMENTED_PAYLOAD,
            "n_init": 2,
            "n_steps": 1,
            "evaluate_x0": True,
            "max_consecutive_failures": 2,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["history"][0]["phase"] == "x0"
    assert body["history"][0]["parameters"] == {"x": 5.0, "y": 5.0}
    assert body["evaluations"] == {"x0": 1, "init": 2, "bo": 1, "failed": 0}


def test_multi_objective_request_forwards_thresholds(services, ax_recorder):
    objectives = [
        {"name": "f_xy", "minimize": True, "threshold": 1000.0},
        {"name": "c_xy", "minimize": False, "threshold": -20.0},
    ]
    response = services.post(
        "/optimize", json={"objectives": objectives, "n_init": 3, "n_steps": 2}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pareto_front"]
    assert set(body["best_objectives"]) == {"f_xy", "c_xy"}
    assert ax_recorder.objective_thresholds() == {"f_xy": 1000.0, "c_xy": -20.0}


def test_failing_tool_aborts_with_the_partial_result(services):
    def failing(x, y):
        raise RuntimeError("solver diverged")

    execution.app.state.problem_pool = execution.ProblemPool(
        {"Paraboloid": failing}, size=1
    )
    response = services.post(
        "/optimize",
        json={
            **DOCUMENTED_PAYLOAD,
            "n_init": 3,
            "n_steps": 3,
            "max_consecutive_failures": 2,
        },
    )

    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert "No trial completed" in detail["message"]
    partial = detail["partial_result"]
    assert partial["stop_reason"] == "consecutive_failures"
    assert [entry["status"] for entry in partial["history"]] == ["failed", "failed"]
    assert "solver diverged" in partial["history"][0]["reason"]


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"objectives": [{"name": "f_xy", "fidelity": "low"}]}, "fidelity"),
        ({"objectives": [{"name": "f_xy", "minimise": False}]}, "minimise"),
        ({**DOCUMENTED_PAYLOAD, "n_iterations": 3}, "n_iterations"),
        ({**DOCUMENTED_PAYLOAD, "fidelity_parameter": "f1"}, "fidelity_parameter"),
        ({**DOCUMENTED_PAYLOAD, "n_steps": 0}, "n_steps"),
        ({**DOCUMENTED_PAYLOAD, "n_init": 0}, "n_init"),
        ({**DOCUMENTED_PAYLOAD, "max_consecutive_failures": 0}, "max_consecutive"),
    ],
)
def test_unsupported_fields_are_rejected_with_422(services, payload, field):
    response = services.post("/optimize", json=payload)
    assert response.status_code == 422
    assert field in response.text


def test_validate_reports_a_valid_study(services):
    response = services.post("/validate", json=DOCUMENTED_PAYLOAD)

    assert response.status_code == 200, response.text
    assert response.json() == {"errors": [], "warnings": [], "valid": True}


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"objectives": [{"name": "missing"}]}, "UNKNOWN_OUTPUT"),
        (
            {**DOCUMENTED_PAYLOAD, "parameter_constraints": ["x + z <= 1"]},
            "PARAMETER_CONSTRAINT_INVALID",
        ),
    ],
)
def test_invalid_study_is_rejected_before_any_tool_runs(
    services, monkeypatch, payload, code
):
    calls = []
    real_tool = execution.TOOL_REGISTRY["Paraboloid"]

    @functools.wraps(real_tool)
    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_tool(*args, **kwargs)

    monkeypatch.setitem(execution.TOOL_REGISTRY, "Paraboloid", spy)

    validated = services.post("/validate", json=payload)
    optimized = services.post("/optimize", json={**payload, "n_init": 2, "n_steps": 2})

    assert validated.status_code == 200
    assert validated.json()["valid"] is False
    assert optimized.status_code == 422
    assert optimized.json()["detail"] == validated.json()
    assert code in [finding["code"] for finding in validated.json()["errors"]]
    assert calls == []


def test_configuration_error_from_constructor_maps_to_400(services, monkeypatch):
    def failing_constructor(**kwargs):
        raise OptimizationConfigurationError("bad objective")

    monkeypatch.setattr(optimization, "BayesianOptimizer", failing_constructor)
    response = services.post("/optimize", json=DOCUMENTED_PAYLOAD)
    assert response.status_code == 400
    assert response.json()["detail"] == "bad objective"
