"""
Security test for GraphManager.add_node to ensure no Cypher injection is possible.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock falkordb before importing GraphManager
mock_falkordb = MagicMock()
sys.modules["falkordb"] = mock_falkordb

from mdo_framework.db.graph_manager import GraphManager  # noqa: E402


class TestSecurityFix(unittest.TestCase):
    def setUp(self):
        # Clear the query cache before each test to ensure a clean state
        GraphManager._query_cache = {}

    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
    def test_add_node_valid_kinds(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()

        # Test valid kinds (standard and custom)
        gm.add_node("Variable", "x")
        gm.add_node("Tool", "t1")
        gm.add_node("Performance", "p1")

        self.assertEqual(mock_graph.query.call_count, 3)

        # Verify calls including backticks for label escaping
        self.assertIn(
            "MERGE (n:`Variable` {name: $name})",
            mock_graph.query.call_args_list[0][0][0],
        )
        self.assertIn(
            "MERGE (n:`Tool` {name: $name})", mock_graph.query.call_args_list[1][0][0]
        )
        self.assertIn(
            "MERGE (n:`Performance` {name: $name})",
            mock_graph.query.call_args_list[2][0][0],
        )

    @patch("mdo_framework.db.graph_manager.FalkorDBClient")
    def test_add_node_invalid_kind_rejection(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        mock_graph = MagicMock()
        mock_client_instance.get_graph.return_value = mock_graph

        gm = GraphManager()

        # Test invalid kind with potential injection
        with self.assertRaises(ValueError):
            gm.add_node("Variable:`PotentialInjection`", "x")

        # Test kind with spaces
        with self.assertRaises(ValueError):
            gm.add_node("Variable {name: 'injection'}", "x")

        # Test kind with special characters
        with self.assertRaises(ValueError):
            gm.add_node("Tool-A", "t1")


if __name__ == "__main__":
    unittest.main()
