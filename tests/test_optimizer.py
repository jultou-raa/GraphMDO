import unittest
from unittest.mock import MagicMock, patch

import openmdao.api as om
import torch

from mdo_framework.optimization.optimizer import (
    BayesianOptimizer,
    LocalEvaluator,
    RemoteEvaluator,
)


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        # Setup common mock data
        self.mock_prob = MagicMock(spec=om.Problem)
        self.mock_prob.get_val.return_value = 1.0

        self.design_vars = ["x", "y"]
        self.parameters = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
        ]
        self.objective = "f_xy"
        self.objectives = [{"name": "f_xy"}]
        self.evaluator = LocalEvaluator(self.mock_prob)
        self.bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.double)

    def test_local_evaluator(self):
        evaluator = LocalEvaluator(self.mock_prob)

        result = evaluator.evaluate({"x": 1.0, "y": 2.0}, ["f_xy"])

        self.mock_prob.set_val.assert_any_call("x", 1.0)
        self.mock_prob.set_val.assert_any_call("y", 2.0)
        self.mock_prob.run_model.assert_called_once()
        self.assertEqual(result["f_xy"], 1.0)

    @patch("mdo_framework.optimization.optimizer.httpx.post")
    def test_remote_evaluator(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": {"f_xy": 42.0}}
        mock_post.return_value = mock_response

        evaluator = RemoteEvaluator("http://mock-service")

        result = evaluator.evaluate({"x": 1.0, "y": 2.0}, ["f_xy"])

        mock_post.assert_called_once()
        self.assertEqual(result["f_xy"], 42.0)

    @patch("mdo_framework.optimization.optimizer.AxClient")
    def test_optimize_loop(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trial.return_value = ({"x": 0.5, "y": 0.5}, 0)
        mock_client.get_best_parameters.return_value = (
            {"x": 0.5, "y": 0.5},
            ({"f_xy": 42.0}, None),
        )

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)

        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertIn("best_objectives", result)
        self.assertEqual(len(result["history"]), 3)  # 2 init + 1 step
        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5})

    @patch("mdo_framework.optimization.optimizer.AxClient")
    def test_optimize_bonsai_and_multi_objective(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trial.return_value = ({"x": 0.5, "y": 0.5, "c": "A"}, 0)

        # Test pareto frontier handling
        mock_client.get_pareto_optimal_parameters.return_value = {
            0: (
                {"x": 0.5, "y": 0.5, "c": "A"},
                {"f_xy": (42.0, None), "g_xy": (10.0, None)},
            )
        }

        params = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "c", "type": "choice", "values": ["A", "B"]},
        ]
        objs = [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}]

        opt = BayesianOptimizer(self.evaluator, params, objs, use_bonsai=True)

        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertEqual(len(result["best_parameters"]), 1)
        self.assertEqual(result["best_parameters"][0], {"x": 0.5, "y": 0.5, "c": "A"})
        self.assertEqual(result["best_objectives"][0]["f_xy"], 42.0)
        self.assertEqual(result["best_objectives"][0]["g_xy"], 10.0)

    @patch("mdo_framework.optimization.optimizer.AxClient")
    def test_optimize_pareto_none(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trial.return_value = ({"x": 0.5, "y": 0.5}, 0)

        # Test pareto frontier empty handling
        mock_client.get_pareto_optimal_parameters.return_value = None

        params = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
        ]
        objs = [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}]

        opt = BayesianOptimizer(self.evaluator, params, objs)

        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIsNone(result["best_parameters"])

    @patch("mdo_framework.optimization.optimizer.AxClient")
    def test_optimize_exception(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trial.return_value = ({"x": 0.5, "y": 0.5}, 0)

        # Test exception
        mock_client.get_best_parameters.side_effect = Exception("Failed")

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)

        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIsNone(result["best_parameters"])


if __name__ == "__main__":
    unittest.main()
