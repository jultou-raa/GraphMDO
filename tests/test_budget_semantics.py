"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Real Ax/GEMSEO tests for the optimization budget contract:
total evaluations = 1 (x0) + n_init (Sobol) + n_steps (BoTorch).
"""

import logging
import warnings
from collections import Counter

import pytest

from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.core.topology import TopologicalAnalyzer
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.optimization import ax_algo_lib
from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary
from mdo_framework.optimization.optimizer import (
    BayesianOptimizer,
    OptimizationConfigurationError,
)

FLOAT = {"param_type": "continuous", "value_type": "float"}
BOX = {"lower": -10.0, "upper": 10.0, **FLOAT}
PARABOLOID_SCHEMA = {  # main.py demo
    "tools": [
        {
            "name": "Paraboloid",
            "fidelity": "high",
            "inputs": ["x", "y"],
            "outputs": ["f_xy", "c_xy"],
        }
    ],
    "variables": [
        {"name": "x", **BOX},
        {"name": "y", **BOX},
        {"name": "f_xy", **FLOAT},
        {"name": "c_xy", **FLOAT},
    ],
}
DISCRETE_SCHEMA = {
    "tools": [{"name": "T", "fidelity": "high", "inputs": ["m"], "outputs": ["f"]}],
    "variables": [
        {
            "name": "m",
            "param_type": "choice",
            "choices": ["a", "b", "c"],
            "value_type": "str",
        },
        {"name": "f", **FLOAT},
    ],
}


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # optimize() writes XDSM/plot files into the cwd
    logging.disable(logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def ax_clients(monkeypatch):
    clients = []
    real_client = ax_algo_lib.Client

    def capture(*args, **kwargs):
        clients.append(real_client(*args, **kwargs))
        return clients[-1]

    monkeypatch.setattr(ax_algo_lib, "Client", capture)
    return clients


@pytest.fixture
def stop_messages(monkeypatch):
    messages = []
    real_run = AxOptimizationLibrary._run

    def spy(self, problem):
        result = real_run(self, problem)
        messages.append(result[0])
        return result

    monkeypatch.setattr(AxOptimizationLibrary, "_run", spy)
    return messages


def build_optimizer(schema, tool, outputs, constraints=None):
    analyzer = TopologicalAnalyzer(schema)
    parameters = analyzer.extract_parameters(analyzer.resolve_dependencies(outputs)[0])
    builder = GraphProblemBuilder(schema)
    problem = builder.build_problem({schema["tools"][0]["name"]: tool})
    return BayesianOptimizer(
        LocalEvaluator(problem, builder.variable_specs),
        parameters,
        [{"name": outputs[0], "minimize": True}],
        constraints,
    )


def trial_kinds(client) -> Counter:
    return Counter(
        (
            trial.generator_runs[0]._generation_node_name or "x0",
            trial.status.name,
        )
        for trial in client._experiment.trials.values()
    )


@pytest.mark.parametrize(("n_init", "n_steps"), [(5, 5), (2, 3)])
def test_budget_is_x0_plus_sobol_plus_botorch(
    ax_clients, stop_messages, n_init, n_steps
):
    calls = []

    def paraboloid(x, y):
        calls.append((x, y))
        return {"f_xy": (x - 3) ** 2 + x * y + (y + 4) ** 2 - 3, "c_xy": x - y}

    optimizer = build_optimizer(
        PARABOLOID_SCHEMA,
        paraboloid,
        ["f_xy"],
        [{"name": "c_xy", "op": "<=", "bound": 0.0}],
    )
    optimizer.optimize(n_steps=n_steps, n_init=n_init)

    assert len(calls) == 1 + n_init + n_steps
    kinds = trial_kinds(ax_clients[0])
    assert kinds == {
        ("x0", "COMPLETED"): 1,
        ("GenerationStep_0_Sobol", "COMPLETED"): n_init,
        ("GenerationStep_1_BoTorch", "COMPLETED"): n_steps,
    }
    assert stop_messages == [ax_algo_lib.BUDGET_REACHED_MESSAGE]


def test_exhausted_discrete_space_terminates_and_reports_why(ax_clients, stop_messages):
    calls = []

    def tool(m):
        calls.append(m)
        return {"a": 1.0, "b": 0.5, "c": 2.0}[m]

    result = build_optimizer(DISCRETE_SCHEMA, tool, ["f"]).optimize(
        n_steps=50, n_init=2
    )

    assert set(calls) <= {"a", "b", "c"}
    assert len(calls) == len(set(calls))
    assert result["best_parameters"] == {"m": "b"}
    assert len(stop_messages) == 1
    assert stop_messages[0].startswith(ax_algo_lib.STALLED_MESSAGE_PREFIX)
    statuses = {status for _, status in trial_kinds(ax_clients[0])}
    assert statuses == {"COMPLETED"}


@pytest.mark.parametrize(("n_init", "n_steps"), [(0, 5), (5, 0), (-1, 5), (5, -2)])
def test_invalid_budget_is_rejected_before_any_tool_call(n_init, n_steps):
    calls = []

    def paraboloid(x, y):
        calls.append((x, y))
        return {"f_xy": x + y, "c_xy": x - y}

    optimizer = build_optimizer(PARABOLOID_SCHEMA, paraboloid, ["f_xy"])
    with pytest.raises(OptimizationConfigurationError, match="must be >= 1"):
        optimizer.optimize(n_steps=n_steps, n_init=n_init)
    assert calls == []
