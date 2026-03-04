import unittest
from unittest.mock import MagicMock, patch

import openmdao.api as om

from mdo_framework.optimization.optimizer import BayesianOptimizer, RemoteEvaluator
from mdo_framework.core.evaluators import LocalEvaluator


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        # Setup common mock data
        self.mock_prob = MagicMock(spec=om.Problem)
        self.mock_prob.get_val.return_value = 1.0

        self.evaluator = LocalEvaluator(self.mock_prob)

        self.parameters = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
        ]
        self.objectives = [{"name": "f_xy", "minimize": True}]

    def test_local_evaluator(self):
        evaluator = LocalEvaluator(self.mock_prob)
        res = evaluator.evaluate({"x": 0.5}, ["f_xy"])
        self.mock_prob.set_val.assert_called_with("x", 0.5)
        self.mock_prob.run_model.assert_called_once()
        self.assertEqual(res["f_xy"], 1.0)

    @patch("mdo_framework.optimization.optimizer.httpx.post")
    def test_remote_evaluator(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": {"f_xy": 2.0}}
        mock_post.return_value = mock_resp

        evaluator = RemoteEvaluator("http://fake-url")
        res = evaluator.evaluate({"x": 0.5}, ["f_xy"])

        mock_post.assert_called_once()
        self.assertEqual(res["f_xy"], 2.0)

    @patch("mdo_framework.optimization.optimizer.AxClient")
    def test_optimize_basic(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.get_next_trial.side_effect = [
            ({"x": 0.5, "y": 0.5}, 0),
            ({"x": 0.2, "y": 0.8}, 1),
            ({"x": 0.9, "y": 0.1}, 2),
        ]

        mock_client.get_best_parameters.return_value = (
            {"x": 0.5, "y": 0.5},
            ({"f_xy": (1.0, None)}, None),
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

        mock_client.get_next_trial.side_effect = [
            ({"x": 0.5, "y": 0.5, "c": "A"}, 0),
            ({"x": 0.2, "y": 0.8, "c": "B"}, 1),
            ({"x": 0.9, "y": 0.1, "c": "A"}, 2),
        ]

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

        mock_client.get_next_trial.side_effect = [
            ({"x": 0.5, "y": 0.5}, 0),
        ]

        mock_client.get_pareto_optimal_parameters.return_value = None

        objs = [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}]

        opt = BayesianOptimizer(self.evaluator, self.parameters, objs)
        result = opt.optimize(n_steps=0, n_init=1)

        self.assertIsNone(result["best_parameters"])
        self.assertIsNone(result["best_objectives"])

    @patch("mdo_framework.optimization.optimizer.AxClient")
    def test_optimize_exception(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.get_next_trial.side_effect = [
            ({"x": 0.5, "y": 0.5}, 0),
        ]

        # Force exception
        mock_client.get_best_parameters.side_effect = Exception("Optimization failed")

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        result = opt.optimize(n_steps=0, n_init=1)

        self.assertIsNone(result["best_parameters"])

    @patch("mdo_framework.optimization.optimizer.AxClient")
    def test_optimize_with_constraints(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.get_next_trial.side_effect = [
            ({"x": 0.5, "y": 0.5}, 0),
            ({"x": 0.2, "y": 0.8}, 1),
            ({"x": 0.9, "y": 0.1}, 2),
        ]

        mock_client.get_best_parameters.return_value = (
            {"x": 0.5, "y": 0.5},
            ({"f_xy": (1.0, None)}, None),
        )

        constraints = [{"name": "g_xy", "op": "<=", "bound": 0.0}]

        opt = BayesianOptimizer(
            self.evaluator, self.parameters, self.objectives, constraints=constraints
        )
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertEqual(len(result["history"]), 3)


if __name__ == "__main__":
    unittest.main()
