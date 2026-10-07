"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import unittest

import numpy as np

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

    def test_built_problem_executes_tool(self):
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

        def paraboloid(x, y):
            return (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0

        mda = GraphProblemBuilder(schema).build_problem({"Paraboloid": paraboloid})
        out = mda.execute({"x": np.array([3.0]), "y": np.array([-4.0])})

        self.assertEqual(mda.disciplines[0].name, "Paraboloid")
        self.assertAlmostEqual(float(np.asarray(out["f_xy"]).flat[0]), -15.0)


if __name__ == "__main__":
    unittest.main()
