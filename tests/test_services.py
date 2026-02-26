# ruff: noqa: E402
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

# Pre-patch GraphManager to avoid DB connection during import of services.graph.main
gm_patcher = patch("mdo_framework.db.graph_manager.GraphManager")
MockGraphManager = gm_patcher.start()
# Set the mock instance that will be assigned to 'gm' in main.py
mock_gm_instance = MockGraphManager.return_value

# Now safe to import
from services.graph.main import app as graph_app
from services.execution.main import app as execution_app
from services.optimization.main import app as optimization_app


class TestGraphService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(graph_app)
        self.mock_gm = mock_gm_instance  # The instance created at module level

    def test_create_variable(self):
        # The endpoint calls gm.add_variable
        response = self.client.post("/variables", json={"name": "x", "value": 1.0})
        self.assertEqual(response.status_code, 200)
        self.mock_gm.add_variable.assert_called_with("x", 1.0, None, None)

    def test_get_schema(self):
        self.mock_gm.get_graph_schema.return_value = {"tools": []}
        response = self.client.get("/schema")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"tools": []})


class TestExecutionService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(execution_app)

    @patch("services.execution.main.httpx.AsyncClient")
    def test_evaluate(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Return a schema for Paraboloid
        mock_resp.json.return_value = {
            "tools": [
                {"name": "Paraboloid", "inputs": ["x", "y"], "outputs": ["f_xy"]}
            ],
            "variables": [{"name": "x"}, {"name": "y"}],
        }
        mock_client.get.return_value = mock_resp

        payload = {"inputs": {"x": 3.0, "y": -4.0}, "objective": "f_xy"}

        response = self.client.post("/evaluate", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"], -15.0)


class TestOptimizationService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(optimization_app)

    @patch("mdo_framework.optimization.optimizer.BayesianOptimizer.optimize")
    def test_optimize(self, mock_optimize):
        import numpy as np

        # Mock result of optimization
        mock_optimize.return_value = {
            "best_x": np.array([0.5, 0.5]),
            "best_y": np.array(0.0),
            "history_x": np.array([[0.0, 0.0]]),
            "history_y": np.array([[0.0]]),
        }

        payload = {
            "design_vars": ["x", "y"],
            "objective": "f_xy",
            "bounds": [[0.0, 1.0], [0.0, 1.0]],
            "n_steps": 1,
            "n_init": 1,
        }

        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("best_x", response.json())


if __name__ == "__main__":
    unittest.main()
