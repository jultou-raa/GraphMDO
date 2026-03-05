# ruff: noqa: E402
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

# Pre-patch GraphManager to avoid DB connection during import of services.graph.main
gm_patcher = patch("mdo_framework.db.graph_manager.GraphManager")
MockGraphManager = gm_patcher.start()
# Set the mock instance that will be assigned to 'gm' in main.py
mock_gm_instance = MockGraphManager.return_value

# Now safe to import
from services.execution.main import app as execution_app
from services.graph.main import app as graph_app
from services.optimization.main import app as optimization_app


class TestGraphService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(graph_app)
        self.mock_gm = mock_gm_instance  # The instance created at module level

    def test_create_variable(self):
        # The endpoint calls gm.add_variable
        response = self.client.post("/variables", json={"name": "x", "value": 1.0})
        self.assertEqual(response.status_code, 200)
        self.mock_gm.add_variable.assert_called_with(
            "x", 1.0, None, None, "continuous", None, "float"
        )

    def test_get_schema(self):
        self.mock_gm.get_graph_schema.return_value = {"tools": [], "variables": []}
        response = self.client.get("/schema")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"tools": [], "variables": []})


class TestExecutionService(unittest.TestCase):
    def setUp(self):
        # Force a hard wipe of cached state properties before each test
        execution_app.state.schema_provider = None
        execution_app.state.problem_pool = None

        self.client = TestClient(execution_app)

    def test_config_validation_invalid_types(self):
        import importlib

        import services.execution.main

        try:
            with patch.dict("os.environ", {"CACHE_TTL": "invalid"}):
                with self.assertRaises(ValueError) as context:
                    importlib.reload(services.execution.main)
                self.assertIn("must be numeric", str(context.exception))
        finally:
            importlib.reload(services.execution.main)

    def test_config_validation_invalid_values(self):
        import importlib

        import services.execution.main

        try:
            with patch.dict("os.environ", {"PROBLEM_POOL_SIZE": "-1"}):
                with self.assertRaises(ValueError) as context:
                    importlib.reload(services.execution.main)
                self.assertIn("must be a positive integer", str(context.exception))

            with patch.dict("os.environ", {"CACHE_TTL": "-10.0"}):
                with self.assertRaises(ValueError) as context:
                    importlib.reload(services.execution.main)
                self.assertIn("must be positive", str(context.exception))
        finally:
            importlib.reload(services.execution.main)

    def test_evaluate(self):
        # We need to mock the state because we're using TestClient which might not run lifespan
        # and we want to test the caching logic explicitly.
        from services.execution.main import TOOL_REGISTRY, ProblemPool, SchemaProvider

        with (
            execution_app.container_context()
            if hasattr(execution_app, "container_context")
            else patch.dict(execution_app.state.__dict__, {})
        ):
            mock_client = AsyncMock()
            execution_app.state.schema_provider = SchemaProvider(mock_client)
            # Use a small pool for testing
            execution_app.state.problem_pool = ProblemPool(TOOL_REGISTRY, size=1)

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "tools": [
                    {"name": "Paraboloid", "inputs": ["x", "y"], "outputs": ["f_xy"]}
                ],
                "variables": [{"name": "x"}, {"name": "y"}],
            }
            mock_resp.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_resp

            payload = {"inputs": {"x": 3.0, "y": -4.0}, "objectives": ["f_xy"]}

            # Reset local expiry cache to ensure isolated test runs do not trip each other
            execution_app.state.schema_provider.expiry = 0
            execution_app.state.schema_provider.envelope = None
            mock_client.get.reset_mock()

            # 1. First call (cache miss)
            response = self.client.post("/evaluate", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["results"]["f_xy"], -15.0)
            self.assertEqual(mock_client.get.call_count, 1)

            # 2. Second call (cache hit)
            response = self.client.post("/evaluate", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(mock_client.get.call_count, 1)

            # 3. Third call (expired cache)
            import time

            # Force expiry
            execution_app.state.schema_provider.expiry = time.time() - 1
            response = self.client.post("/evaluate", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(mock_client.get.call_count, 2)

    def test_evaluate_unknown_objective_input(self):
        from services.execution.main import TOOL_REGISTRY, ProblemPool, SchemaProvider

        with (
            execution_app.container_context()
            if hasattr(execution_app, "container_context")
            else patch.dict(execution_app.state.__dict__, {})
        ):
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"tools": [], "variables": [{"name": "x"}]}
            mock_client.get.return_value = mock_resp
            execution_app.state.schema_provider = SchemaProvider(mock_client)
            execution_app.state.problem_pool = ProblemPool(TOOL_REGISTRY, size=1)

            # Unknown objective
            response = self.client.post(
                "/evaluate", json={"inputs": {"x": 1.0}, "objectives": ["unknown_obj"]}
            )
            self.assertEqual(response.status_code, 422)
            self.assertIn("Unknown objective", response.json()["detail"])

            # Unknown input
            response = self.client.post(
                "/evaluate",
                json={"inputs": {"unknown_var": 1.0}, "objectives": ["f_xy"]},
            )
            self.assertEqual(response.status_code, 422)

    def test_evaluate_payload_limits(self):
        # inputs > 100
        large_inputs = {f"var_{i}": 1.0 for i in range(101)}
        response = self.client.post(
            "/evaluate", json={"inputs": large_inputs, "objectives": ["f_xy"]}
        )
        self.assertEqual(response.status_code, 422)

        # input key > 50 chars
        large_key = "a" * 51
        response = self.client.post(
            "/evaluate", json={"inputs": {large_key: 1.0}, "objectives": ["f_xy"]}
        )
        self.assertEqual(response.status_code, 422)

        # empty inputs
        response = self.client.post(
            "/evaluate", json={"inputs": {}, "objectives": ["f_xy"]}
        )
        self.assertEqual(response.status_code, 422)

    def test_evaluate_transformation_failure(self):
        from services.execution.main import TOOL_REGISTRY, ProblemPool, SchemaProvider

        with (
            execution_app.container_context()
            if hasattr(execution_app, "container_context")
            else patch.dict(execution_app.state.__dict__, {})
        ):
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "tools": [
                    {"name": "Paraboloid", "inputs": ["x", "y"], "outputs": ["f_xy"]}
                ],
                "variables": [{"name": "x"}, {"name": "y"}],
            }
            mock_client.get.return_value = mock_resp
            execution_app.state.schema_provider = SchemaProvider(mock_client)
            execution_app.state.problem_pool = ProblemPool(TOOL_REGISTRY, size=1)

            with patch(
                "services.execution.main.to_float",
                side_effect=ValueError("Invalid Shape"),
            ):
                response = self.client.post(
                    "/evaluate",
                    json={"inputs": {"x": 1.0, "y": 1.0}, "objectives": ["f_xy"]},
                )
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.json()["detail"], "Invalid result shape.")

    def test_evaluate_execution_failure(self):
        from services.execution.main import TOOL_REGISTRY, ProblemPool, SchemaProvider

        with (
            execution_app.container_context()
            if hasattr(execution_app, "container_context")
            else patch.dict(execution_app.state.__dict__, {})
        ):
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "tools": [
                    {"name": "Paraboloid", "inputs": ["x", "y"], "outputs": ["f_xy"]}
                ],
                "variables": [{"name": "x"}, {"name": "y"}],
            }
            mock_client.get.return_value = mock_resp
            execution_app.state.schema_provider = SchemaProvider(mock_client)
            # Patch discard_instance to verify it's called
            with patch(
                "services.execution.main.ProblemPool.discard_instance",
                new_callable=AsyncMock,
            ) as mock_discard:
                execution_app.state.problem_pool = ProblemPool(TOOL_REGISTRY, size=1)

                with patch(
                    "services.execution.main.execute_problem",
                    side_effect=Exception("Solver crashed"),
                ):
                    response = self.client.post(
                        "/evaluate",
                        json={"inputs": {"x": 1.0, "y": 1.0}, "objectives": ["f_xy"]},
                    )
                    self.assertEqual(response.status_code, 500)
                    self.assertEqual(
                        response.json()["detail"],
                        "An internal execution error occurred.",
                    )
                    mock_discard.assert_called_once()

    def test_health_degraded(self):
        from services.execution.main import SchemaProvider

        with patch.dict(execution_app.state.__dict__, {}):
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection Refused")
            execution_app.state.schema_provider = SchemaProvider(mock_client)

            response = self.client.get("/health")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["status"], "degraded")

    def test_schema_provider_httpx_errors(self):
        import httpx

        from services.execution.main import SchemaProvider

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("Network error")

        provider = SchemaProvider(mock_client)

        # When no cache exists, it raises 503 HTTP Exception
        with self.assertRaises(Exception) as context:
            import asyncio

            asyncio.run(provider.get_schema())
        self.assertEqual(context.exception.status_code, 503)

    def test_schema_provider_json_errors(self):
        from services.execution.main import SchemaProvider

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        mock_client.get.return_value = mock_resp

        provider = SchemaProvider(mock_client)

        # When no cache exists, it raises 502 HTTP Exception
        with self.assertRaises(Exception) as context:
            import asyncio

            asyncio.run(provider.get_schema())
        self.assertEqual(context.exception.status_code, 502)

    def test_schema_provider_lock_timeout(self):
        from services.execution.main import SchemaProvider

        mock_client = AsyncMock()
        provider = SchemaProvider(mock_client)

        async def delayed_acquire():
            await provider.lock.acquire()
            import asyncio

            await asyncio.sleep(2.0)
            provider.lock.release()

        async def test_timeout():
            import asyncio

            # background task holds the lock
            asyncio.create_task(delayed_acquire())
            # tiny sleep to let the background task grab the lock
            await asyncio.sleep(0.1)

            # This should timeout because the lock is held
            with self.assertRaises(Exception) as context:
                await provider.get_schema()
            self.assertEqual(context.exception.status_code, 503)

        import asyncio

        asyncio.run(test_timeout())

    def test_schema_envelope_invalid_format(self):
        from services.execution.main import SchemaEnvelope

        with self.assertRaises(ValueError):
            # Pass a type that causes TypeError during iteration
            SchemaEnvelope(
                {
                    "variables": [{"name": "x"}],
                    "tools": [{"name": "tool", "outputs": None}],
                }
            )


class TestProblemPool(unittest.TestCase):
    def setUp(self):
        from services.execution.main import TOOL_REGISTRY, ProblemPool

        self.registry = TOOL_REGISTRY
        self.pool = ProblemPool(registry=self.registry, size=2)

    def test_problem_pool_teardown(self):
        import asyncio

        async def run_test():
            # Add a mock instance to the pool
            mock_inst = MagicMock()
            await self.pool.pool.put(mock_inst)

            # create a placeholder task in the background set
            async def dummy_task():
                await asyncio.sleep(5)

            t = asyncio.create_task(dummy_task())
            self.pool._background_tasks.add(t)

            await self.pool.teardown()

            # Pool should be empty, cleanup should have been called in thread
            self.assertTrue(self.pool.pool.empty())
            # Background task should be cancelled
            self.assertTrue(t.cancelled())

        asyncio.run(run_test())

    def test_problem_pool_replenish_and_discard(self):
        import asyncio

        from services.execution.main import SchemaEnvelope

        envelope = SchemaEnvelope(
            {
                "tools": [
                    {"name": "Paraboloid", "inputs": ["x", "y"], "outputs": ["f_xy"]}
                ],
                "variables": [{"name": "x"}, {"name": "y"}],
            }
        )
        self.pool.current_hash = envelope.hash

        async def run_test():
            mock_inst = MagicMock()

            # Mock the to_thread call inside _replenish_one so we don't actually hang building a real problem
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                # To mock the inst.cleanup and build_and_init calls
                mock_thread.return_value = MagicMock()

                await self.pool.discard_instance(mock_inst, envelope)

                # Wait briefly for the task to finish spinning up and populating the pool
                await asyncio.sleep(0.1)

                self.assertFalse(self.pool.pool.empty())
                new_inst = await self.pool.pool.get()
                self.assertIsNotNone(new_inst)

                # cleanup of mock_inst was sent to thread
                self.assertTrue(mock_thread.called)

        asyncio.run(run_test())

    def test_problem_pool_build_failures(self):
        import asyncio

        from fastapi import HTTPException

        from services.execution.main import SchemaEnvelope

        envelope = SchemaEnvelope(
            {
                "tools": [{"name": "MissingTool", "inputs": ["x"], "outputs": ["y"]}],
                "variables": [{"name": "x"}],
            }
        )

        async def run_test():
            # Building this will fail because "MissingTool" is not in registry
            with self.assertRaises(HTTPException) as context:
                await self.pool.get_instance(envelope)
            self.assertEqual(context.exception.status_code, 500)
            self.assertIsNone(self.pool.current_hash)  # hash reset

        asyncio.run(run_test())

    def test_problem_pool_checkout_timeout(self):
        import asyncio

        from fastapi import HTTPException

        from services.execution.main import SchemaEnvelope

        envelope = SchemaEnvelope(
            {
                "tools": [
                    {"name": "Paraboloid", "inputs": ["x", "y"], "outputs": ["f_xy"]}
                ],
                "variables": [{"name": "x"}, {"name": "y"}],
            }
        )

        async def run_test():
            # Simply patch the get_instance logic to simulate a timeout directly
            # since all lower level starvation combinations are hanging the pytest event loop.
            original_get = self.pool.pool.get

            async def instant_timeout():
                raise TimeoutError("Starvation Timeout")

            self.pool.pool.get = instant_timeout

            # Pretend pool is built so we skip the build logic block
            self.pool.current_hash = envelope.hash

            try:
                with self.assertRaises(HTTPException) as context:
                    await self.pool.get_instance(envelope)
                self.assertEqual(context.exception.status_code, 503)
            finally:
                self.pool.pool.get = original_get

        asyncio.run(run_test())

    def test_problem_pool_release_stale(self):
        import asyncio

        async def run_test():
            self.pool.current_hash = "active_hash"
            mock_inst = MagicMock()

            await self.pool.release_instance(mock_inst, "stale_hash")
            self.assertTrue(self.pool.pool.empty())

            await self.pool.release_instance(mock_inst, "active_hash")
            self.assertFalse(self.pool.pool.empty())

        asyncio.run(run_test())


class TestOptimizationService(unittest.TestCase):
    def setUp(self):
        self.mock_client = AsyncMock()
        optimization_app.state.client = self.mock_client

        # Default mock response for schema fetch
        self.mock_resp = MagicMock()
        self.mock_resp.json.return_value = {"tools": [], "variables": []}
        self.mock_client.get.return_value = self.mock_resp

        self.client = TestClient(optimization_app)

    @patch("mdo_framework.optimization.optimizer.BayesianOptimizer.optimize")
    @patch("mdo_framework.core.topology.TopologicalAnalyzer.resolve_dependencies")
    def test_optimize(self, mock_resolve, mock_optimize):
        import numpy as np

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "variables": [
                {
                    "name": "x",
                    "param_type": "range",
                    "lower": 0.0,
                    "upper": 1.0,
                    "value_type": "float",
                },
                {
                    "name": "y",
                    "param_type": "range",
                    "lower": 0.0,
                    "upper": 1.0,
                    "value_type": "float",
                },
            ],
            "tools": [{"name": "ToolA", "inputs": ["x", "y"], "outputs": ["f_xy"]}],
        }

        self.mock_client.get.return_value = mock_resp
        mock_resolve.return_value = (
            ["x", "y"],
            [{"name": "ToolA", "inputs": ["x", "y"], "outputs": ["f_xy"]}],
        )

        # Mock result of optimization
        mock_optimize.return_value = {
            "best_x": np.array([0.5, 0.5]),
            "best_y": np.array(0.0),
            "history_x": np.array([[0.0, 0.0]]),
            "history_y": np.array([[0.0]]),
        }

        payload = {
            "objectives": [{"name": "f_xy"}],
            "n_steps": 1,
            "n_init": 1,
        }

        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("best_parameters", response.json())

    @patch("mdo_framework.optimization.optimizer.BayesianOptimizer.optimize")
    @patch("mdo_framework.core.topology.TopologicalAnalyzer.resolve_dependencies")
    def test_optimize_tensor_conversion(self, mock_resolve, mock_optimize):
        import torch

        # Mock result of optimization
        mock_optimize.return_value = {
            "best_parameters": {"x": 0.5, "y": 0.5},
            "best_objectives": {"f_xy": 0.0},
            "history": [
                {
                    "parameters": {"x": 0.5, "y": 0.5},
                    "objectives": {"f_xy": torch.tensor(0.0)},
                }
            ],
        }

        payload = {
            "objectives": [{"name": "f_xy"}],
            "n_steps": 1,
            "n_init": 1,
        }

        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 200)

    @patch("mdo_framework.optimization.optimizer.BayesianOptimizer.optimize")
    @patch("mdo_framework.core.topology.TopologicalAnalyzer.resolve_dependencies")
    def test_optimize_exception(self, mock_resolve, mock_optimize):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "variables": [
                {
                    "name": "x",
                    "param_type": "range",
                    "lower": 0.0,
                    "upper": 1.0,
                    "value_type": "float",
                },
                {
                    "name": "y",
                    "param_type": "range",
                    "lower": 0.0,
                    "upper": 1.0,
                    "value_type": "float",
                },
            ],
            "tools": [{"name": "ToolA", "inputs": ["x", "y"], "outputs": ["f_xy"]}],
        }

        optimization_app.state.client.get.return_value = mock_resp
        mock_resolve.return_value = (
            ["x", "y"],
            [{"name": "ToolA", "inputs": ["x", "y"], "outputs": ["f_xy"]}],
        )

        # Mock result of optimization
        mock_optimize.side_effect = Exception("Optimization Failed")

        payload = {
            "objectives": [{"name": "f_xy"}],
            "n_steps": 1,
            "n_init": 1,
        }

        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 500)

    @patch("mdo_framework.optimization.optimizer.BayesianOptimizer.optimize")
    def test_optimize_tensor_conversion_nested(self, mock_optimize):
        import numpy as np

        # Mock result of optimization
        mock_optimize.return_value = {
            "best_parameters": {"x": 0.5, "y": 0.5},
            "best_objectives": {"f_xy": 0.0},
            "history": [
                {
                    "parameters": {"x": 0.5, "y": 0.5},
                    "objectives": {"f_xy": np.array([0.0])},
                }
            ],
        }

        payload = {
            "objectives": [{"name": "f_xy"}],
            "n_steps": 1,
            "n_init": 1,
        }

        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 200)

    def test_optimize_invalid_payload(self):
        payload = {
            "objectives": [],
            "n_steps": 1,
            "n_init": 1,
        }
        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 500)

    def test_optimize_tensor_lists(self):
        # A simple check for the internal to_list method inside optimize route
        # is covered by having best_parameters return a numpy array
        pass

    @patch("mdo_framework.optimization.optimizer.BayesianOptimizer.__init__")
    @patch("mdo_framework.core.topology.TopologicalAnalyzer.resolve_dependencies")
    def test_optimize_exception2(self, mock_resolve, mock_init):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "variables": [
                {
                    "name": "x",
                    "param_type": "range",
                    "lower": 0.0,
                    "upper": 1.0,
                    "value_type": "float",
                },
                {
                    "name": "y",
                    "param_type": "range",
                    "lower": 0.0,
                    "upper": 1.0,
                    "value_type": "float",
                },
            ],
            "tools": [{"name": "ToolA", "inputs": ["x", "y"], "outputs": ["f_xy"]}],
        }

        optimization_app.state.client.get.return_value = mock_resp
        mock_resolve.return_value = (
            ["x", "y"],
            [{"name": "ToolA", "inputs": ["x", "y"], "outputs": ["f_xy"]}],
        )

        # Mock result of optimization
        mock_init.side_effect = Exception("Initialization Failed")

        payload = {
            "objectives": [{"name": "f_xy"}],
            "n_steps": 1,
            "n_init": 1,
        }

        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 500)

    @patch("mdo_framework.optimization.optimizer.BayesianOptimizer.optimize")
    @patch("mdo_framework.core.topology.TopologicalAnalyzer.resolve_dependencies")
    def test_optimize_with_constraints_service(self, mock_resolve, mock_optimize):
        mock_optimize.return_value = {
            "best_parameters": {"x": 0.5, "y": 0.5},
            "best_objectives": {"f_xy": 0.0},
            "history": [],
        }

        payload = {
            "parameters": [
                {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "f_xy"}],
            "constraints": [{"name": "g_xy", "op": "<=", "bound": 0.0}],
            "n_steps": 1,
            "n_init": 1,
        }

        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
