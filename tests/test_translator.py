import unittest
from unittest.mock import MagicMock
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.db.graph_manager import GraphManager
import openmdao.api as om

class TestTranslator(unittest.TestCase):
    def test_build_problem_invalid_tool(self):
        mock_gm = MagicMock()
        mock_gm.get_tools.return_value = [{'name': 'InvalidTool', 'fidelity': 'high'}]

        builder = GraphProblemBuilder(mock_gm)

        tool_registry = {} # Empty registry

        with self.assertRaises(ValueError):
            builder.build_problem(tool_registry)

    def test_build_problem_success(self):
        mock_gm = MagicMock()
        mock_gm.get_tools.return_value = [{'name': 'ToolA', 'fidelity': 'high'}]
        mock_gm.get_tool_inputs.return_value = ['x']
        mock_gm.get_tool_outputs.return_value = ['y']

        builder = GraphProblemBuilder(mock_gm)

        def tool_func(x): return x

        tool_registry = {'ToolA': tool_func}

        prob = builder.build_problem(tool_registry)

        self.assertIsInstance(prob, om.Problem)

if __name__ == '__main__':
    unittest.main()
