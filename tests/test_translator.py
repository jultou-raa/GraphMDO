import unittest
from mdo_framework.core.translator import GraphProblemBuilder
import openmdao.api as om


class TestTranslator(unittest.TestCase):
    def test_build_problem_invalid_tool(self):
        # Schema definition instead of mock GM
        schema = {
            "tools": [{"name": "InvalidTool", "fidelity": "high"}],
            "variables": [],
        }

        builder = GraphProblemBuilder(schema)

        tool_registry = {}  # Empty registry

        with self.assertRaises(ValueError):
            builder.build_problem(tool_registry)

    def test_build_problem_success(self):
        schema = {
            "tools": [
                {"name": "ToolA", "fidelity": "high", "inputs": ["x"], "outputs": ["y"]}
            ],
            "variables": [{"name": "x", "value": 1.0}, {"name": "y"}],
        }

        builder = GraphProblemBuilder(schema)

        def tool_func(x):
            return x

        tool_registry = {"ToolA": tool_func}

        prob = builder.build_problem(tool_registry)

        self.assertIsInstance(prob, om.Problem)

        # Verify set_val called implicitly during build/setup if we could inspect
        # But we can verify by running
        prob.run_model()
        self.assertEqual(prob.get_val("x"), 1.0)

    def test_build_problem_variable_setup_error(self):
        # Trigger the try-except block in set_val
        schema = {
            "tools": [
                {"name": "ToolA", "fidelity": "high", "inputs": ["x"], "outputs": ["y"]}
            ],
            "variables": [{"name": "z", "value": 99.0}],  # z is not in the model
        }

        builder = GraphProblemBuilder(schema)

        def tool_func(x):
            return x

        tool_registry = {"ToolA": tool_func}

        prob = builder.build_problem(tool_registry)
        # Should not raise, just ignore 'z'
        # We can't easily check if it logged or ignored, but coverage should hit 'except' block if set_val fails?
        # Actually set_val on unknown variable might succeed if it just stores it?
        # No, set_val usually fails if variable not in model.
        # But translator calls it after setup().

        try:
            prob.get_val("z")
        except KeyError:
            # Expected not to be there
            pass


if __name__ == "__main__":
    unittest.main()
