"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

End-to-end tests (real Ax/GEMSEO) for the values tools receive.
"""

import logging
import warnings

import pytest

from mdo_framework.optimization.optimizer import BayesianOptimizer


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
