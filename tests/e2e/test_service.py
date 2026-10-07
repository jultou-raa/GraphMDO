"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Un-mocked Optimization Service tests: real BayesianOptimizer, RemoteEvaluator
and Execution Service app. Only the graph schema is served by a mock transport.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

import services.execution.main as execution
import services.optimization.main as optimization
from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary
from mdo_framework.optimization.optimizer import (
    AX_OBJECTIVE_KEYS,
    BayesianOptimizer,
    OptimizationConfigurationError,
)

pytestmark = pytest.mark.e2e

FLOAT = {"param_type": "continuous", "value_type": "float"}
SCHEMA = {  # documented walkthrough (docs/user-guide/running-optimization.md)
    "tools": [
        {
            "name": "Paraboloid",
            "fidelity": "high",
            "inputs": ["x", "y"],
            "outputs": ["f_xy"],
        }
    ],
    "variables": [
        {"name": "x", "lower": 0.0, "upper": 10.0, **FLOAT},
        {"name": "y", "lower": 0.0, "upper": 10.0, **FLOAT},
        {"name": "f_xy", **FLOAT},
    ],
}
DOCUMENTED_PAYLOAD = {"objectives": [{"name": "f_xy", "minimize": True}]}


@pytest.fixture
def services(monkeypatch):
    graph = httpx.MockTransport(lambda request: httpx.Response(200, json=SCHEMA))
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


def test_optimize_service_end_to_end(services, monkeypatch):
    forwarded = []
    real_execute = AxOptimizationLibrary.execute

    def spy(self, problem, **settings):
        forwarded.extend(settings["ax_objectives"])
        return real_execute(self, problem, **settings)

    monkeypatch.setattr(AxOptimizationLibrary, "execute", spy)
    response = services.post(
        "/optimize", json={**DOCUMENTED_PAYLOAD, "n_init": 2, "n_steps": 2}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "f_xy" in body["best_objectives"]
    assert set(body["best_parameters"]) == {"x", "y"}
    assert body["history"]
    assert forwarded == [{"name": "f_xy", "minimize": True}]
    assert all(set(objective) <= AX_OBJECTIVE_KEYS for objective in forwarded)


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"objectives": [{"name": "f_xy", "fidelity": "low"}]}, "fidelity"),
        ({"objectives": [{"name": "f_xy", "minimise": False}]}, "minimise"),
        ({**DOCUMENTED_PAYLOAD, "n_iterations": 3}, "n_iterations"),
        ({**DOCUMENTED_PAYLOAD, "fidelity_parameter": "f1"}, "fidelity_parameter"),
        ({**DOCUMENTED_PAYLOAD, "n_steps": 0}, "n_steps"),
        ({**DOCUMENTED_PAYLOAD, "n_init": 0}, "n_init"),
    ],
)
def test_unsupported_fields_are_rejected_with_422(services, payload, field):
    response = services.post("/optimize", json=payload)
    assert response.status_code == 422
    assert field in response.text


def test_objective_threshold_is_forwarded():
    objective = optimization.ObjectiveConfig(name="f", minimize=False, threshold=1.5)
    assert objective.to_ax() == {"name": "f", "minimize": False, "threshold": 1.5}


def test_optimizer_rejects_unsupported_objective_keys():
    with pytest.raises(OptimizationConfigurationError, match="fidelity"):
        BayesianOptimizer(
            evaluator=object(),
            parameters=[{"name": "x", "type": "range", "bounds": [0.0, 1.0]}],
            objectives=[{"name": "f", "minimize": True, "fidelity": None}],
        )


def test_configuration_error_from_constructor_maps_to_400(services, monkeypatch):
    def failing_constructor(**kwargs):
        raise OptimizationConfigurationError("bad objective")

    monkeypatch.setattr(optimization, "BayesianOptimizer", failing_constructor)
    response = services.post("/optimize", json=DOCUMENTED_PAYLOAD)
    assert response.status_code == 400
    assert response.json()["detail"] == "bad objective"
