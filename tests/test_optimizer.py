import unittest
from unittest.mock import MagicMock, patch

import openmdao.api as om

from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.optimization.optimizer import BayesianOptimizer, RemoteEvaluator


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

    def test_local_evaluator_scalar(self):
        """Tests LocalEvaluator when Problem.get_val returns a scalar."""
        self.mock_prob.get_val.return_value = 1.23

        result = self.evaluator.evaluate({"x": 1.0, "y": 2.0}, ["f_xy"])

        self.mock_prob.set_val.assert_any_call("x", 1.0)
        self.mock_prob.set_val.assert_any_call("y", 2.0)
        self.mock_prob.run_model.assert_called_once()
        self.assertEqual(result["f_xy"], 1.23)

    def test_local_evaluator_iterable(self):
        """Tests LocalEvaluator when Problem.get_val returns an iterable (e.g. numpy array)."""
        self.mock_prob.get_val.return_value = [4.56]

        result = self.evaluator.evaluate({"x": 1.0, "y": 2.0}, ["f_xy"])

        self.mock_prob.set_val.assert_any_call("x", 1.0)
        self.mock_prob.set_val.assert_any_call("y", 2.0)
        self.mock_prob.run_model.assert_called_once()
        self.assertEqual(result["f_xy"], 4.56)

    @patch("mdo_framework.optimization.optimizer.httpx.post")
    def test_remote_evaluator(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": {"f_xy": 2.0}}
        mock_post.return_value = mock_resp

        evaluator = RemoteEvaluator("http://fake-url")
        res = evaluator.evaluate({"x": 0.5}, ["f_xy"])

        mock_post.assert_called_once()
        self.assertEqual(res["f_xy"], 2.0)

    @patch("mdo_framework.optimization.optimizer.Client")
    def test_optimize_basic(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertIn("best_objectives", result)
        self.assertEqual(len(result["history"]), 3)  # 2 init + 1 step
        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5})

    @patch("mdo_framework.optimization.optimizer.Client")
    def test_optimize_bonsai_and_multi_objective(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": "A"}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        # Test pareto frontier handling
        mock_client.get_pareto_frontier.return_value = [
            ({"x": 0.5, "y": 0.5, "c": "A"}, {"f_xy": 42.0, "g_xy": 10.0}, 0, "0_0")
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
        self.assertEqual(len(result["best_parameters"]), 1)
        self.assertEqual(result["best_parameters"][0], {"x": 0.5, "y": 0.5, "c": "A"})
        self.assertEqual(result["best_objectives"][0]["f_xy"], 42.0)
        self.assertEqual(result["best_objectives"][0]["g_xy"], 10.0)

    @patch("mdo_framework.optimization.optimizer.Client")
    def test_optimize_pareto_none(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        # Test pareto frontier empty handling
        mock_client.get_pareto_frontier.return_value = None

        objs = [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}]

        opt = BayesianOptimizer(self.evaluator, self.parameters, objs)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIsNone(result["best_parameters"])
        self.assertIsNone(result["best_objectives"])

    @patch("mdo_framework.optimization.optimizer.Client")
    def test_optimize_exception(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        # Force exception on result retrieval
        mock_client.get_best_parameterization.side_effect = Exception(
            "Optimization failed"
        )

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIsNone(result["best_parameters"])

    @patch("mdo_framework.optimization.optimizer.Client")
    def test_optimize_with_constraints(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        constraints = [{"name": "g_xy", "op": "<=", "bound": 0.0}]

        opt = BayesianOptimizer(
            self.evaluator, self.parameters, self.objectives, constraints=constraints
        )
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertEqual(len(result["history"]), 3)

    @patch("mdo_framework.optimization.optimizer.Client")
    def test_fidelity_parameter_emits_warning(self, mock_client_cls):
        """fidelity_parameter is not supported by the new Ax Client API;
        a UserWarning must be raised and optimization still completes normally."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5},
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

        with self.assertWarns(UserWarning) as ctx:
            result = opt.optimize(n_steps=1, n_init=2)

        warning_message = str(ctx.warning)
        self.assertIn("fidelity_parameter", warning_message)
        self.assertIn("'x'", warning_message)
        self.assertIn("not supported", warning_message)
        # Optimization must still complete despite the warning
        self.assertIn("best_parameters", result)


if __name__ == "__main__":
    unittest.main()
