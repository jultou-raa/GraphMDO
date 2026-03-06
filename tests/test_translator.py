import unittest

from mdo_framework.core.translator import GraphProblemBuilder

class TestTranslator(unittest.TestCase):
    def test_build_problem_invalid_tool(self):
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
                {
                    "name": "ToolA",
                    "fidelity": "high",
                    "inputs": ["x"],
                    "outputs": ["y"],
                },
            ],
            "variables": [{"name": "x", "value": 1.0}, {"name": "y"}],
        }

        builder = GraphProblemBuilder(schema)

        def tool_func(x):
            return x

        tool_registry = {"ToolA": tool_func}

        mda = builder.build_problem(tool_registry)

        self.assertEqual(mda.name, "MDAChain")

        # In GEMSEO, default inputs extracted from schema are stored in builder.default_inputs
        self.assertEqual(builder.default_inputs["x"], 1.0)


if __name__ == "__main__":
    unittest.main()
