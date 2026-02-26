import unittest
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.optimization.optimizer import BayesianOptimizer, LocalEvaluator
import torch


class TestIntegration(unittest.TestCase):
    def test_workflow(self):
        # 1. Use schema directly (no GraphManager mock needed for translator anymore)
        schema = {
            "tools": [
                {
                    "name": "Paraboloid",
                    "fidelity": "high",
                    "inputs": ["x", "y"],
                    "outputs": ["f_xy"],
                }
            ],
            "variables": [
                {"name": "x", "value": 3.0},
                {"name": "y", "value": -4.0},
                {"name": "f_xy"},
            ],
        }

        # 2. Translate
        builder = GraphProblemBuilder(schema)

        def paraboloid_func(x, y):
            return (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0

        tool_registry = {"Paraboloid": paraboloid_func}

        prob = builder.build_problem(tool_registry)

        # Verify Problem Structure
        self.assertIn("Paraboloid", prob.model._subsystems_allprocs)

        # Check execution
        prob.run_model()
        self.assertAlmostEqual(prob.get_val("f_xy")[0], -15.0)

        # 3. Optimize with LocalEvaluator
        evaluator = LocalEvaluator(prob)

        optimizer = BayesianOptimizer(
            evaluator=evaluator,
            design_vars=["x", "y"],
            objective="f_xy",
            bounds=torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.double),
        )

        try:
            result = optimizer.optimize(n_steps=1, n_init=2)
            self.assertIn("best_y", result)
        except Exception as e:
            self.fail(f"Optimization failed: {e}")


if __name__ == "__main__":
    unittest.main()
