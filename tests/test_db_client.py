import unittest
from unittest.mock import patch
from mdo_framework.db.client import FalkorDBClient
import os


class TestFalkorDBClient(unittest.TestCase):
    @patch("mdo_framework.db.client.FalkorDB")
    def test_singleton_pattern(self, mock_falkordb):
        # Reset singleton instance
        FalkorDBClient._instance = None

        client1 = FalkorDBClient()
        client2 = FalkorDBClient()

        self.assertIs(client1, client2)
        mock_falkordb.assert_called_once()

    @patch("mdo_framework.db.client.FalkorDB")
    def test_initialization(self, mock_falkordb):
        # Reset singleton instance
        FalkorDBClient._instance = None

        with patch.dict(
            os.environ, {"FALKORDB_HOST": "1.2.3.4", "FALKORDB_PORT": "1234"}
        ):
            client = FalkorDBClient()
            mock_falkordb.assert_called_with(host="1.2.3.4", port=1234)

            client.client.select_graph.assert_called_with("mdo_graph")

    @patch("mdo_framework.db.client.FalkorDB")
    def test_get_graph(self, mock_falkordb):
        # Reset singleton instance
        FalkorDBClient._instance = None

        client = FalkorDBClient()
        graph = client.get_graph()

        self.assertEqual(graph, client.graph)

    @patch("mdo_framework.db.client.FalkorDB")
    def test_close(self, mock_falkordb):
        # Reset singleton instance
        FalkorDBClient._instance = None

        client = FalkorDBClient()
        # Ensure calling close doesn't crash (it's a pass for now)
        client.close()


if __name__ == "__main__":
    unittest.main()
