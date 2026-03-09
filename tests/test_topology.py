"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import unittest

from mdo_framework.core.topology import TopologicalAnalyzer


class TestTopologicalAnalyzer(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "variables": [
                {
                    "name": "x",
                    "param_type": "range",
                    "lower": 0.0,
                    "upper": 10.0,
                    "value_type": "float",
                },
                {
                    "name": "y",
                    "param_type": "choice",
                    "choices": ["A", "B"],
                    "value_type": "str",
                },
                {
                    "name": "z",
                    "param_type": "continuous",
                    "lower": 0.0,
                    "upper": 5.0,
                    "value_type": "float",
                },
                {"name": "unused_in", "param_type": "continuous"},
                {"name": "out1"},
                {"name": "out2"},
            ],
            "tools": [
                {"name": "Tool1", "inputs": ["x", "y"], "outputs": ["z"]},
                {"name": "Tool2", "inputs": ["z"], "outputs": ["out1", "out2"]},
                {"name": "UnusedTool", "inputs": ["unused_in"], "outputs": ["out3"]},
            ],
        }

    def test_resolve_dependencies_full(self):
        analyzer = TopologicalAnalyzer(self.schema)
        design_vars, req_tools = analyzer.resolve_dependencies(["out1"])

        # Unused inputs/tools should not be here
        self.assertEqual(design_vars, ["x", "y"])
        tool_names = [t["name"] for t in req_tools]
        self.assertEqual(tool_names, ["Tool2", "Tool1"])

    def test_extract_parameters(self):
        analyzer = TopologicalAnalyzer(self.schema)
        params = analyzer.extract_parameters(["x", "y"])

        self.assertEqual(len(params), 2)
        x_param = next(p for p in params if p["name"] == "x")
        self.assertEqual(x_param["type"], "range")
        self.assertEqual(x_param["bounds"], [0.0, 10.0])

        y_param = next(p for p in params if p["name"] == "y")
        self.assertEqual(y_param["type"], "choice")
        self.assertEqual(y_param["values"], ["A", "B"])

    def test_resolve_dependencies_missing_var(self):
        analyzer = TopologicalAnalyzer(self.schema)
        with self.assertRaises(ValueError):
            analyzer.resolve_dependencies(["missing_out"])


if __name__ == "__main__":
    unittest.main()
