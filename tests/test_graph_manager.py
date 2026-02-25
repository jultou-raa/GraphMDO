import unittest
from unittest.mock import MagicMock, patch
from mdo_framework.db.graph_manager import GraphManager

class TestGraphManager(unittest.TestCase):
    @patch('mdo_framework.db.graph_manager.FalkorDBClient')
    def test_add_variable(self, mock_client_cls):
        # Setup Mock
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()
        gm.add_variable("x", value=1.0)

        # Verify query execution
        mock_graph.query.assert_called_once()
        args, _ = mock_graph.query.call_args
        self.assertIn("MERGE (v:Variable {name: 'x'})", args[0])
        self.assertIn("SET v.value = 1.0", args[0])

    @patch('mdo_framework.db.graph_manager.FalkorDBClient')
    def test_add_tool(self, mock_client_cls):
        # Setup Mock
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()
        gm.add_tool("ToolA")

        # Verify query execution
        mock_graph.query.assert_called_once()
        args, _ = mock_graph.query.call_args
        self.assertIn("MERGE (t:Tool {name: 'ToolA', fidelity: 'high'})", args[0])

    @patch('mdo_framework.db.graph_manager.FalkorDBClient')
    def test_get_tools(self, mock_client_cls):
        # Setup Mock
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        # Mock result set
        mock_result = MagicMock()
        mock_result.result_set = [["ToolA", "high"], ["ToolB", "low"]]
        mock_graph.query.return_value = mock_result

        gm = GraphManager()
        tools = gm.get_tools()

        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]['name'], "ToolA")
        self.assertEqual(tools[1]['fidelity'], "low")

    @patch('mdo_framework.db.graph_manager.FalkorDBClient')
    def test_clear_graph(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()
        gm.clear_graph()

        mock_graph.query.assert_called_with("MATCH (n) DETACH DELETE n")

    @patch('mdo_framework.db.graph_manager.FalkorDBClient')
    def test_connect_tool_to_output(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()
        gm.connect_tool_to_output("ToolA", "VarX")

        args, _ = mock_graph.query.call_args
        self.assertIn("(t:Tool {name: 'ToolA'})", args[0])
        self.assertIn("(v:Variable {name: 'VarX'})", args[0])
        self.assertIn("(t)-[:OUTPUTS]->(v)", args[0])

    @patch('mdo_framework.db.graph_manager.FalkorDBClient')
    def test_connect_input_to_tool(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()
        gm.connect_input_to_tool("VarX", "ToolA")

        args, _ = mock_graph.query.call_args
        self.assertIn("(v:Variable {name: 'VarX'})", args[0])
        self.assertIn("(t:Tool {name: 'ToolA'})", args[0])
        self.assertIn("(v)-[:INPUTS_TO]->(t)", args[0])

    @patch('mdo_framework.db.graph_manager.FalkorDBClient')
    def test_get_variables(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        mock_result = MagicMock()
        # v.name, v.value, v.lower, v.upper
        mock_result.result_set = [["VarX", 1.0, 0.0, 10.0]]
        mock_graph.query.return_value = mock_result

        gm = GraphManager()
        variables = gm.get_variables()

        self.assertEqual(len(variables), 1)
        self.assertEqual(variables[0]['name'], "VarX")
        self.assertEqual(variables[0]['value'], 1.0)
        self.assertEqual(variables[0]['lower'], 0.0)
        self.assertEqual(variables[0]['upper'], 10.0)

    @patch('mdo_framework.db.graph_manager.FalkorDBClient')
    def test_get_tool_inputs(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        mock_result = MagicMock()
        mock_result.result_set = [["VarX"], ["VarY"]]
        mock_graph.query.return_value = mock_result

        gm = GraphManager()
        inputs = gm.get_tool_inputs("ToolA")

        self.assertEqual(inputs, ["VarX", "VarY"])

    @patch('mdo_framework.db.graph_manager.FalkorDBClient')
    def test_get_tool_outputs(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        mock_result = MagicMock()
        mock_result.result_set = [["VarZ"]]
        mock_graph.query.return_value = mock_result

        gm = GraphManager()
        outputs = gm.get_tool_outputs("ToolA")

        self.assertEqual(outputs, ["VarZ"])

if __name__ == '__main__':
    unittest.main()
