import unittest
from unittest.mock import MagicMock, patch
import torch
from mdo_framework.optimization.optimizer import (
    BayesianOptimizer,
    LocalEvaluator,
    RemoteEvaluator,
)
import openmdao.api as om
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import GP


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        # Setup common mock data
        self.mock_prob = MagicMock(spec=om.Problem)
        self.mock_prob.get_val.return_value = 1.0

        self.design_vars = ["x", "y"]
        self.objective = "f_xy"
        self.evaluator = LocalEvaluator(self.mock_prob)
        self.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    def test_local_evaluator(self):
        evaluator = LocalEvaluator(self.mock_prob)
        x = torch.tensor([[1.0, 2.0]], dtype=torch.double)

        result = evaluator.evaluate(x, self.design_vars, self.objective)

        self.mock_prob.set_val.assert_any_call("x", 1.0)
        self.mock_prob.set_val.assert_any_call("y", 2.0)
        self.mock_prob.run_model.assert_called_once()
        self.assertEqual(result.item(), 1.0)

    @patch("mdo_framework.optimization.optimizer.httpx.post")
    def test_remote_evaluator(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": 42.0}
        mock_post.return_value = mock_response

        evaluator = RemoteEvaluator("http://mock-service")
        x = torch.tensor([[1.0, 2.0]], dtype=torch.double)

        result = evaluator.evaluate(x, self.design_vars, self.objective)

        mock_post.assert_called_once()
        self.assertEqual(result.item(), 42.0)

    @patch("mdo_framework.optimization.optimizer.optimize_acqf")
    @patch("mdo_framework.optimization.optimizer.SingleTaskGP")
    @patch("mdo_framework.optimization.optimizer.fit_gpytorch_mll")
    @patch("mdo_framework.optimization.optimizer.ExpectedImprovement")
    def test_optimize_loop(self, mock_EI, mock_fit, mock_gp_cls, mock_acqf):
        # Mocking botorch components
        mock_acqf.return_value = (torch.tensor([[0.5, 0.5]], dtype=torch.double), None)

        # We need SingleTaskGP instance to return a GaussianLikelihood AND satisfy isinstance(model, GP)
        # mock_gp_instance = mock_gp_cls.return_value

        mock_gp_instance_with_spec = MagicMock(spec=GP)
        mock_gp_cls.return_value = mock_gp_instance_with_spec

        mock_gp_instance_with_spec.likelihood = MagicMock(spec=GaussianLikelihood)

        opt = BayesianOptimizer(
            self.evaluator, self.design_vars, self.objective, self.bounds
        )

        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_x", result)
        self.assertIn("best_y", result)
        self.assertEqual(len(result["history_x"]), 3)  # 2 init + 1 step


if __name__ == "__main__":
    unittest.main()
