"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from gemseo.core.discipline import Discipline
from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.optimization.optimizer import BayesianOptimizer, RemoteEvaluator

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        # Setup common mock data for GEMSEO discipline
        class MockDisc(Discipline):
            def __init__(self):
                super().__init__(name="MockDisc")
                self.input_grammar.update_from_names(["x", "y", "c"])
                self.output_grammar.update_from_names(["f_xy", "g_xy"])
                self.default_inputs = {"x": np.array([0.0]), "y": np.array([0.0]), "c": np.array([0.0])}
            def _run(self, input_data):
                self.local_data["f_xy"] = np.array([1.0])
                self.local_data["g_xy"] = np.array([0.0])
        self.mock_prob = MockDisc()

        self.evaluator = LocalEvaluator(self.mock_prob)

        self.parameters = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "c", "type": "range", "bounds": [0.0, 1.0]},
        ]
        self.objectives = [{"name": "f_xy", "minimize": True}]

    def test_local_evaluator_scalar(self):
        """Tests LocalEvaluator when GEMSEO returns an array containing a scalar."""
        def dummy_run(self, input_data):
            self.local_data["f_xy"] = np.array([1.23])
            self.local_data["g_xy"] = np.array([0.0])
        self.mock_prob._run = dummy_run.__get__(self.mock_prob, type(self.mock_prob))
        res = self.evaluator.evaluate({"x": 0.5, "y": 0.5, "c": 0.0}, ["f_xy"])
        self.assertEqual(res["f_xy"], 1.23)

    def test_local_evaluator_iterable(self):
        """Tests LocalEvaluator when GEMSEO returns a flat array."""
        def dummy_run2(self, input_data):
            self.local_data["f_xy"] = np.array([4.56, 1.0])
            self.local_data["g_xy"] = np.array([0.0])
        self.mock_prob._run = dummy_run2.__get__(self.mock_prob, type(self.mock_prob))
        res = self.evaluator.evaluate({"x": 0.5, "y": 0.5, "c": 0.0}, ["f_xy"])
        self.assertEqual(res["f_xy"], 4.56)

    @patch("mdo_framework.optimization.optimizer.httpx.post")
    def test_remote_evaluator(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": {"f_xy": 2.0}}
        mock_post.return_value = mock_resp

        evaluator = RemoteEvaluator("http://fake-url")
        res = evaluator.evaluate({"x": 0.5, "y": 0.5, "c": 0.0}, ["f_xy"])

        mock_post.assert_called_once()
        self.assertEqual(res["f_xy"], 2.0)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_basic(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertIn("best_objectives", result)
        self.assertEqual(len(result["history"]), 0)
        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5, "c": 0.5})

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_bonsai_and_multi_objective(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        # Test pareto frontier handling
        mock_client.get_pareto_frontier.return_value = [
            ({"x": 0.5, "y": 0.5, "c": "A"}, {"f_xy": 42.0, "g_xy": 10.0}, 0, "0_0"),
        ]

        params = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "c", "type": "choice", "values": ["A", "B"]},
        ]
        objs = [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}]

        opt = BayesianOptimizer(self.evaluator, params, objs, use_bonsai=True)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5, "c": 0.5})
        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5, "c": 0.5})
        pass

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_pareto_none(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        mock_client.get_pareto_frontier.return_value = None

        objs = [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}]

        opt = BayesianOptimizer(self.evaluator, self.parameters, objs)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5, "c": 0.5})
        self.assertEqual(result["best_objectives"]["f_xy"], 1.0)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_exception(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        mock_client.get_best_parameterization.side_effect = Exception("Optimization failed")

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5, "c": 0.5})

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_with_constraints(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        constraints = [{"name": "g_xy", "op": "<=", "bound": 0.0}]

        opt = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            constraints=constraints,
        )
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertEqual(len(result["history"]), 0)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_fidelity_parameter_emits_warning(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        opt = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            fidelity_parameter="x",
        )

        with self.assertWarns(UserWarning):
            result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)

if __name__ == "__main__":
    unittest.main()
