import unittest
from unittest.mock import MagicMock
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.optimization.optimizer import BayesianOptimizer
import torch


class TestIntegration(unittest.TestCase):
    def test_workflow(self):
        # 1. Mock GraphManager
        mock_gm = MagicMock()

        # Setup mock data for Paraboloid problem
        # Tools
        mock_gm.get_tools.return_value = [{"name": "Paraboloid", "fidelity": "high"}]

        # Inputs/Outputs
        mock_gm.get_tool_inputs.return_value = ["x", "y"]
        mock_gm.get_tool_outputs.return_value = ["f_xy"]

        # 2. Translate
        builder = GraphProblemBuilder(mock_gm)

        def paraboloid_func(x, y):
            return (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0

        tool_registry = {"Paraboloid": paraboloid_func}

        prob = builder.build_problem(tool_registry)

        # Verify Problem Structure
        # Check if subsystem exists
        self.assertIn("Paraboloid", prob.model._subsystems_allprocs)

        # Check if variables exist
        # OpenMDAO 3.x+ setup() must be called before check?
        # builder.build_problem calls setup().

        # Check inputs
        # Paraboloid component should have inputs 'x', 'y' promoted
        # We can check by running
        prob.set_val("x", 3.0)
        prob.set_val("y", -4.0)
        prob.run_model()

        # Expected: (3-3)^2 + 3*(-4) + (-4+4)^2 - 3 = 0 - 12 + 0 - 3 = -15
        self.assertAlmostEqual(prob.get_val("f_xy")[0], -15.0)

        # 3. Optimize
        # Mocking BayesianOptimizer because it takes time and involves heavy botorch logic
        # But we want to test that it runs.
        # Let's run a very short optimization loop.

        optimizer = BayesianOptimizer(
            problem=prob, design_vars=["x", "y"], objective="f_xy"
        )

        # Set bounds to narrow range to speed up
        optimizer.bounds = torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.double)

        # Run optimization (1 step)
        try:
            result = optimizer.optimize(n_steps=1, n_init=2)
            self.assertIn("best_y", result)
        except Exception as e:
            self.fail(f"Optimization failed: {e}")


if __name__ == "__main__":
    unittest.main()
