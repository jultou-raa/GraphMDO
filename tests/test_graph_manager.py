import unittest
from unittest.mock import MagicMock, patch

from mdo_framework.db.graph_manager import GraphManager


class TestGraphManager(unittest.TestCase):
    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
    def test_add_variable(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()
        gm.add_variable("x", value=1.0)

        mock_graph.query.assert_called_once()
        args, _ = mock_graph.query.call_args
        self.assertIn("MERGE (n:Variable {name: $name})", args[0])
        self.assertIn("SET n += $props", args[0])
        params = mock_graph.query.call_args[1].get("params")
        self.assertEqual(params["props"]["value"], 1.0)

    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
    def test_add_tool(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()
        gm.add_tool("ToolA")

        mock_graph.query.assert_called_once()
        args, _ = mock_graph.query.call_args
        self.assertIn("MERGE (n:Tool {name: $name})", args[0])

    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
    def test_get_tools(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        mock_result = MagicMock()
        node_a = MagicMock()
        node_a.properties = {"name": "ToolA", "fidelity": "high"}
        node_b = MagicMock()
        node_b.properties = {"name": "ToolB", "fidelity": "low"}
        mock_result.result_set = [[node_a], [node_b]]
        mock_graph.query.return_value = mock_result

        gm = GraphManager()
        tools = gm.get_tools()

        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["name"], "ToolA")
        self.assertEqual(tools[1]["fidelity"], "low")

    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
    def test_clear_graph(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()
        gm.clear_graph()

        mock_graph.query.assert_called_with("MATCH (n) DETACH DELETE n")

    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
    def test_connect_tool_to_output(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()
        gm.connect_tool_to_output("ToolA", "VarX")

        args, _ = mock_graph.query.call_args
        self.assertIn("(t:Tool {name: $tool_name})", args[0])
        self.assertIn("(v:Variable {name: $variable_name})", args[0])
        self.assertIn("(t)-[:OUTPUTS]->(v)", args[0])

    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
    def test_connect_input_to_tool(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()
        gm.connect_input_to_tool("VarX", "ToolA")

        args, _ = mock_graph.query.call_args
        self.assertIn("(v:Variable {name: $variable_name})", args[0])
        self.assertIn("(t:Tool {name: $tool_name})", args[0])
        self.assertIn("(v)-[:INPUTS_TO]->(t)", args[0])

    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
    def test_get_variables(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        mock_result = MagicMock()
        node_x = MagicMock()
        node_x.properties = {
            "name": "VarX",
            "value": 1.0,
            "lower": 0.0,
            "upper": 10.0,
            "param_type": "range",
            "choices": "null",
            "value_type": "float",
        }
        mock_result.result_set = [[node_x]]
        mock_graph.query.return_value = mock_result

        gm = GraphManager()
        variables = gm.get_variables()

        self.assertEqual(len(variables), 1)
        self.assertEqual(variables[0]["name"], "VarX")
        self.assertEqual(variables[0]["value"], 1.0)
        self.assertEqual(variables[0]["lower"], 0.0)
        self.assertEqual(variables[0]["upper"], 10.0)

    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
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

    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
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

    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
    def test_get_graph_schema(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        # We need to mock return values for sequential calls:
        # 1. get_variables -> ["VarX"]
        # 2. MATCH (t:Tool) OPTIONAL MATCH ... -> ToolA, ["VarX"], ["VarY"]

        res1 = MagicMock()
        node_x = MagicMock()
        node_x.properties = {
            "name": "VarX",
            "value": 1.0,
            "lower": 0.0,
            "upper": 10.0,
            "param_type": "range",
            "choices": "null",
            "value_type": "float",
        }
        res1.result_set = [[node_x]]

        res2 = MagicMock()
        node_a = MagicMock()
        node_a.properties = {"name": "ToolA", "fidelity": "high"}
        res2.result_set = [[node_a, ["VarX"], ["VarY"]]]

        mock_graph.query.side_effect = [res1, res2]

        gm = GraphManager()
        schema = gm.get_graph_schema()

        self.assertIn("tools", schema)
        self.assertIn("variables", schema)
        self.assertEqual(len(schema["tools"]), 1)
        self.assertEqual(schema["tools"][0]["name"], "ToolA")
        self.assertEqual(schema["tools"][0]["inputs"], ["VarX"])
        self.assertEqual(schema["tools"][0]["outputs"], ["VarY"])
        self.assertEqual(len(schema["variables"]), 1)


if __name__ == "__main__":
    unittest.main()
