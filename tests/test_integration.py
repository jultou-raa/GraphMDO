import unittest

from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.core.translator import GraphProblemBuilder
import numpy as np
from mdo_framework.optimization.optimizer import BayesianOptimizer


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
                },
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
        self.assertEqual(prob.disciplines[0].name, "Paraboloid")

        # Check execution
        out = prob.execute({"x": np.array([3.0]), "y": np.array([-4.0])})
        self.assertAlmostEqual(float(np.asarray(out["f_xy"]).flat[0]), -15.0)

        # 3. Optimize with LocalEvaluator
        evaluator = LocalEvaluator(prob)

        optimizer = BayesianOptimizer(
            evaluator=evaluator,
            parameters=[
                {"name": "x", "type": "range", "bounds": [0.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 10.0]},
            ],
            objectives=[{"name": "f_xy"}],
        )

        try:
            result = optimizer.optimize(n_steps=1, n_init=2)
            self.assertIn("best_objectives", result)
        except Exception as e:
            self.fail(f"Optimization failed: {e}")


if __name__ == "__main__":
    unittest.main()
