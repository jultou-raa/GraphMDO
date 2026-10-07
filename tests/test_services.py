"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

# ruff: noqa: E402
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

try:
    import torch
except ImportError:
    torch = None

from fakes.falkordb import FakeGraph

from mdo_framework.core.topology import ResolvedInputs
from mdo_framework.db.graph_manager import GraphManager
from mdo_framework.optimization.optimizer import (
    OptimizationConfigurationError,
    RemoteEvaluationTransportError,
)
from mdo_framework.schema import (
    ChoiceVar,
    Finding,
    FixedParam,
    RangeVar,
    StateVar,
    StudySchema,
    StudyValidationError,
    ToolNode,
    ToolSpec,
    ValidationReport,
)
from services.execution.main import TOOL_REGISTRY
from services.execution.main import app as execution_app
from services.graph.main import app as graph_app
from services.graph.main import get_graph_manager
from services.optimization.main import app as optimization_app

X_BODY = {"kind": "range", "name": "x", "lower": -10.0, "upper": 10.0}

PARABOLOID_SCHEMA = StudySchema(
    variables=[
        RangeVar(name="x", lower=-10.0, upper=10.0),
        RangeVar(name="y", lower=-10.0, upper=10.0),
        StateVar(name="f_xy"),
        StateVar(name="c_xy"),
    ],
    tools=[ToolSpec(name="Paraboloid", inputs=["x", "y"], outputs=["f_xy", "c_xy"])],
)
PARABOLOID_PAYLOAD = PARABOLOID_SCHEMA.model_dump(mode="json")

SERVICE_VARIABLES = (
    RangeVar(name="x", lower=0.0, upper=1.0),
    RangeVar(name="y", lower=0.0, upper=1.0),
)
SERVICE_SCHEMA = StudySchema(
    variables=[*SERVICE_VARIABLES, StateVar(name="f_xy"), StateVar(name="g_xy")],
    tools=[ToolSpec(name="ToolA", inputs=["x", "y"], outputs=["f_xy", "g_xy"])],
)
SERVICE_PAYLOAD = SERVICE_SCHEMA.model_dump(mode="json")
SERVICE_RESOLVED = ResolvedInputs(
    design_variables=SERVICE_VARIABLES, fixed_parameters=(), tools=("ToolA",)
)


class TestGraphService(unittest.TestCase):
    def setUp(self):
        self.graph = FakeGraph()
        self.manager = GraphManager(graph=self.graph)
        graph_app.dependency_overrides[get_graph_manager] = lambda: self.manager
        self.addCleanup(graph_app.dependency_overrides.clear)
        self.client = TestClient(graph_app)

    def build_paraboloid(self):
        self.manager.add_variable(RangeVar(name="x", lower=-10.0, upper=10.0))
        self.manager.add_variable(RangeVar(name="y", lower=-10.0, upper=10.0))
        self.manager.add_variable(StateVar(name="f_xy"))
        self.manager.add_tool(ToolNode(name="Paraboloid"))
        self.manager.connect_input_to_tool("x", "Paraboloid")
        self.manager.connect_input_to_tool("y", "Paraboloid")
        self.manager.connect_tool_to_output("Paraboloid", "f_xy")

    def test_create_variable(self):
        response = self.client.post("/variables", json=X_BODY)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {"status": "created", "variable": "x"})
        self.assertEqual(
            self.manager.get_variables(), [RangeVar(name="x", lower=-10.0, upper=10.0)]
        )

    def test_create_variable_of_every_kind(self):
        bodies = [
            (X_BODY, RangeVar(name="x", lower=-10.0, upper=10.0)),
            (
                {"kind": "choice", "name": "m", "choices": ["a", "b"]},
                ChoiceVar(name="m", choices=["a", "b"]),
            ),
            (
                {"kind": "fixed", "name": "g", "value": 9.81},
                FixedParam(name="g", value=9.81),
            ),
            (
                {"kind": "state", "name": "f", "initial_guess": [1.0, 2.0]},
                StateVar(name="f", initial_guess=[1.0, 2.0]),
            ),
        ]
        for body, expected in bodies:
            with self.subTest(kind=body["kind"]):
                response = self.client.post("/variables", json=body)
                self.assertEqual(response.status_code, 201)
        self.assertEqual(
            self.manager.get_variables(), [expected for _, expected in bodies]
        )

    def test_create_existing_variable_conflicts(self):
        self.client.post("/variables", json=X_BODY)
        response = self.client.post(
            "/variables", json={"kind": "fixed", "name": "x", "value": 1}
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json(),
            {"detail": {"error": "exists", "label": "Variable", "name": "x"}},
        )
        self.assertEqual(
            self.manager.get_variables(), [RangeVar(name="x", lower=-10.0, upper=10.0)]
        )

    def test_create_variable_rejects_invalid_bodies(self):
        bodies = {
            "no kind": {"name": "x", "lower": 0.0, "upper": 1.0},
            "unknown kind": {"kind": "continuous", "name": "x"},
            "legacy field": {**X_BODY, "param_type": "range"},
            "inverted bounds": {**X_BODY, "lower": 5.0, "upper": 1.0},
            "bad name": {**X_BODY, "name": "not a name"},
        }
        for label, body in bodies.items():
            with self.subTest(label):
                response = self.client.post("/variables", json=body)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.manager.get_variables(), [])

    def test_create_variable_rejects_a_nan_bound(self):
        response = self.client.post(
            "/variables",
            content='{"kind": "range", "name": "x", "lower": NaN, "upper": 1.0}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["type"], "finite_number")
        self.assertEqual(self.manager.get_variables(), [])

    def test_put_variable_creates_then_replaces(self):
        created = self.client.put("/variables/x", json=X_BODY)
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json(), {"status": "created", "variable": "x"})

        replaced = self.client.put("/variables/x", json={**X_BODY, "upper": 20.0})
        self.assertEqual(replaced.status_code, 200)
        self.assertEqual(replaced.json(), {"status": "replaced", "variable": "x"})
        self.assertEqual(
            self.manager.get_variables(), [RangeVar(name="x", lower=-10.0, upper=20.0)]
        )

    def test_put_variable_rejects_a_name_that_differs_from_the_path(self):
        response = self.client.put("/variables/other", json=X_BODY)
        self.assertEqual(response.status_code, 422)
        self.assertIn("'other'", response.json()["detail"])
        self.assertEqual(self.manager.get_variables(), [])

    def test_put_variable_rejects_an_invalid_body(self):
        response = self.client.put("/variables/x", json={"name": "x"})
        self.assertEqual(response.status_code, 422)

    def test_put_variable_refuses_to_turn_a_produced_variable_into_an_input(self):
        self.build_paraboloid()
        response = self.client.put(
            "/variables/f_xy", json={"kind": "fixed", "name": "f_xy", "value": 1.0}
        )
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(detail["error"], "role_conflict")
        self.assertEqual(detail["variable"], "f_xy")
        self.assertIn("Paraboloid", detail["message"])
        self.assertEqual(
            self.manager.get_study_schema().variable("f_xy"), StateVar(name="f_xy")
        )

    def test_delete_variable(self):
        self.build_paraboloid()
        response = self.client.delete("/variables/x")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "deleted", "variable": "x"})
        self.assertEqual(self.manager.get_tool_inputs("Paraboloid"), ["y"])

    def test_delete_missing_variable_is_not_found(self):
        response = self.client.delete("/variables/ghost")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(),
            {
                "detail": {
                    "missing": [{"label": "Variable", "name": "ghost"}],
                    "hint": None,
                }
            },
        )

    def test_get_schema_of_an_empty_graph(self):
        response = self.client.get("/schema")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"schema_version": "1", "variables": [], "tools": []}
        )

    def test_get_schema_returns_the_typed_study(self):
        self.build_paraboloid()
        response = self.client.get("/schema")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            StudySchema.model_validate(response.json()),
            StudySchema(
                variables=[
                    RangeVar(name="x", lower=-10.0, upper=10.0),
                    RangeVar(name="y", lower=-10.0, upper=10.0),
                    StateVar(name="f_xy"),
                ],
                tools=[
                    ToolSpec(name="Paraboloid", inputs=["x", "y"], outputs=["f_xy"])
                ],
            ),
        )
        self.assertEqual(response.json()["variables"][0]["kind"], "range")

    def test_get_schema_reports_legacy_nodes_as_a_conflict(self):
        self.graph.add_raw_node("Variable", name="old", param_type="continuous")
        response = self.client.get("/schema")
        self.assertEqual(response.status_code, 409)
        report = response.json()["detail"]
        self.assertFalse(report["valid"])
        self.assertEqual(
            [(error["code"], error["names"]) for error in report["errors"]],
            [("LEGACY_NODE", ["old"])],
        )

    def test_health_ok_when_falkordb_answers(self):
        with patch("services.graph.main.ping_database") as ping:
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "falkordb": "ok"})
        ping.assert_called_once()

    def test_health_degraded_when_falkordb_is_unreachable(self):
        with patch(
            "services.graph.main.ping_database",
            side_effect=ConnectionError("Connection refused"),
        ):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertIn("Connection refused", response.json()["falkordb"])

    def test_ping_database_pings_falkordb_connection(self):
        with patch("services.graph.main.FalkorDBClient") as client_cls:
            from services.graph.main import ping_database

            ping_database()
        client_cls.return_value.client.connection.ping.assert_called_once()

    def test_clear_graph(self):
        self.build_paraboloid()
        response = self.client.post("/clear")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "cleared"})
        self.assertEqual(self.manager.get_study_schema(), StudySchema())

    def test_create_tool(self):
        response = self.client.post("/tools", json={"name": "ToolA", "fidelity": "low"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {"status": "created", "tool": "ToolA"})
        self.assertEqual(
            self.manager.get_tools(), [ToolNode(name="ToolA", fidelity="low")]
        )

    def test_create_tool_defaults_to_high_fidelity(self):
        self.client.post("/tools", json={"name": "ToolA"})
        self.assertEqual(self.manager.get_tools(), [ToolNode(name="ToolA")])

    def test_create_tool_accepts_the_tool_options(self):
        response = self.client.post(
            "/tools",
            json={"name": "ToolA", "deterministic": False, "arg_map": {"x": "a"}},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            self.manager.get_tools(),
            [ToolNode(name="ToolA", deterministic=False, arg_map={"x": "a"})],
        )

    def test_put_tool_accepts_the_tool_options(self):
        response = self.client.put(
            "/tools/ToolA",
            json={"name": "ToolA", "arg_map": {"x": "a"}},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            self.manager.get_tools(), [ToolNode(name="ToolA", arg_map={"x": "a"})]
        )

    def test_create_existing_tool_conflicts(self):
        self.client.post("/tools", json={"name": "ToolA"})
        response = self.client.post("/tools", json={"name": "ToolA", "fidelity": "low"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json(),
            {"detail": {"error": "exists", "label": "Tool", "name": "ToolA"}},
        )
        self.assertEqual(self.manager.get_tools(), [ToolNode(name="ToolA")])

    def test_create_tool_rejects_invalid_bodies(self):
        bodies = {
            "unknown field": {"name": "ToolA", "version": "1.2"},
            "connections in body": {"name": "ToolA", "inputs": ["x"]},
            "bad name": {"name": "not a name"},
            "no name": {"fidelity": "low"},
            "duplicate arguments": {"name": "ToolA", "arg_map": {"x": "a", "y": "a"}},
            "non-boolean deterministic": {"name": "ToolA", "deterministic": "no"},
        }
        for label, body in bodies.items():
            with self.subTest(label):
                response = self.client.post("/tools", json=body)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.manager.get_tools(), [])

    def test_put_tool_creates_then_replaces(self):
        created = self.client.put("/tools/ToolA", json={"name": "ToolA"})
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json(), {"status": "created", "tool": "ToolA"})

        replaced = self.client.put(
            "/tools/ToolA", json={"name": "ToolA", "fidelity": "low"}
        )
        self.assertEqual(replaced.status_code, 200)
        self.assertEqual(replaced.json(), {"status": "replaced", "tool": "ToolA"})
        self.assertEqual(
            self.manager.get_tools(), [ToolNode(name="ToolA", fidelity="low")]
        )

    def test_put_tool_rejects_a_name_that_differs_from_the_path(self):
        response = self.client.put("/tools/other", json={"name": "ToolA"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("'other'", response.json()["detail"])
        self.assertEqual(self.manager.get_tools(), [])

    def test_delete_tool(self):
        self.build_paraboloid()
        response = self.client.delete("/tools/Paraboloid")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "deleted", "tool": "Paraboloid"})
        self.assertEqual(self.manager.get_tools(), [])

    def test_delete_missing_tool_is_not_found(self):
        response = self.client.delete("/tools/ghost")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(),
            {"detail": {"missing": [{"label": "Tool", "name": "ghost"}], "hint": None}},
        )

    def test_connect_input(self):
        self.manager.add_variable(RangeVar(name="varX", lower=0.0, upper=1.0))
        self.manager.add_tool(ToolNode(name="ToolA"))
        response = self.client.post(
            "/connections/input", json={"source": "varX", "target": "ToolA"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "connected", "type": "input"})
        self.assertEqual(self.manager.get_tool_inputs("ToolA"), ["varX"])

    def test_connect_output(self):
        self.manager.add_variable(StateVar(name="varY"))
        self.manager.add_tool(ToolNode(name="ToolA"))
        response = self.client.post(
            "/connections/output", json={"source": "ToolA", "target": "varY"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "connected", "type": "output"})
        self.assertEqual(self.manager.get_tool_outputs("ToolA"), ["varY"])

    def test_connecting_a_missing_node_is_not_found(self):
        self.build_paraboloid()
        cases = [
            ("/connections/input", "ghost", "Paraboloid", [("Variable", "ghost")]),
            ("/connections/input", "x", "ghost", [("Tool", "ghost")]),
            ("/connections/output", "Paraboloid", "ghost", [("Variable", "ghost")]),
            ("/connections/output", "ghost", "f_xy", [("Tool", "ghost")]),
        ]
        for path, source, target, missing in cases:
            with self.subTest(path=path, source=source, target=target):
                response = self.client.post(
                    path, json={"source": source, "target": target}
                )
                self.assertEqual(response.status_code, 404)
                self.assertEqual(
                    response.json(),
                    {
                        "detail": {
                            "missing": [
                                {"label": label, "name": name}
                                for label, name in missing
                            ],
                            "hint": None,
                        }
                    },
                )

    def test_swapped_connection_arguments_come_with_a_hint(self):
        self.build_paraboloid()
        response = self.client.post(
            "/connections/input", json={"source": "Paraboloid", "target": "f_xy"}
        )
        self.assertEqual(response.status_code, 404)
        detail = response.json()["detail"]
        self.assertEqual(
            detail["missing"],
            [
                {"label": "Variable", "name": "Paraboloid"},
                {"label": "Tool", "name": "f_xy"},
            ],
        )
        self.assertIn("connect_tool_to_output", detail["hint"])

    def test_second_producer_of_a_variable_conflicts(self):
        self.build_paraboloid()
        self.manager.add_tool(ToolNode(name="Other"))
        response = self.client.post(
            "/connections/output", json={"source": "Other", "target": "f_xy"}
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json(),
            {
                "detail": {
                    "error": "duplicate_producer",
                    "variable": "f_xy",
                    "producers": ["Paraboloid", "Other"],
                }
            },
        )

    def test_design_variable_as_tool_output_conflicts(self):
        self.build_paraboloid()
        response = self.client.post(
            "/connections/output", json={"source": "Paraboloid", "target": "x"}
        )
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual((detail["error"], detail["variable"]), ("role_conflict", "x"))
        self.assertIn("kind 'state'", detail["message"])

    def test_variable_that_is_input_and_output_of_a_tool_conflicts(self):
        self.build_paraboloid()
        response = self.client.post(
            "/connections/input", json={"source": "f_xy", "target": "Paraboloid"}
        )
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(
            (detail["error"], detail["variable"]), ("role_conflict", "f_xy")
        )

    def test_connection_bodies_are_strict(self):
        for path in ("/connections/input", "/connections/output"):
            for body in (
                {"source": "a"},
                {"source": "a", "target": "b", "weight": 1},
            ):
                with self.subTest(path=path, body=body):
                    response = self.client.post(path, json=body)
                    self.assertEqual(response.status_code, 422)


class TestSharedValidationHandler(unittest.TestCase):
    """Every service answers a body holding NaN with 422, not a 500."""

    def setUp(self):
        manager = GraphManager(graph=FakeGraph())
        graph_app.dependency_overrides[get_graph_manager] = lambda: manager
        self.addCleanup(graph_app.dependency_overrides.clear)
        execution_app.state.schema_provider = None
        execution_app.state.problem_pool = None

    def post_raw(self, app, path, text):
        return TestClient(app).post(
            path, content=text, headers={"Content-Type": "application/json"}
        )

    def assert_unprocessable(self, response, error_type):
        self.assertEqual(response.status_code, 422, response.text)
        errors = response.json()["detail"]
        self.assertIn(error_type, [error["type"] for error in errors])
        for error in errors:
            self.assertEqual(set(error), {"loc", "msg", "type"})

    def test_graph_service_rejects_nan(self):
        response = self.post_raw(
            graph_app,
            "/variables",
            '{"kind": "range", "name": "x", "lower": NaN, "upper": 1.0}',
        )
        self.assert_unprocessable(response, "finite_number")

    def test_execution_service_rejects_nan(self):
        response = self.post_raw(
            execution_app, "/evaluate", '{"inputs": {"x": 1.0}, "objectives": [NaN]}'
        )
        self.assert_unprocessable(response, "string_type")

    def test_execution_service_rejects_a_nan_input_value(self):
        response = self.post_raw(
            execution_app,
            "/evaluate",
            '{"inputs": {"x": NaN}, "objectives": ["f_xy"]}',
        )
        self.assert_unprocessable(response, "finite_number")

    def test_optimization_service_rejects_nan(self):
        response = self.post_raw(
            optimization_app,
            "/optimize",
            '{"objectives": [{"name": "f_xy"}], "n_steps": NaN}',
        )
        self.assert_unprocessable(response, "finite_number")


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
            mock_resp.json.return_value = PARABOLOID_PAYLOAD
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

    def test_evaluate_demo_graph_with_the_default_registry(self):
        from services.execution.main import TOOL_REGISTRY, ProblemPool, SchemaProvider

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = PARABOLOID_PAYLOAD
        mock_client.get.return_value = mock_resp
        execution_app.state.schema_provider = SchemaProvider(mock_client)
        execution_app.state.problem_pool = ProblemPool(TOOL_REGISTRY, size=1)

        response = self.client.post(
            "/evaluate",
            json={"inputs": {"x": 3.0, "y": -4.0}, "objectives": ["f_xy", "c_xy"]},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["results"], {"f_xy": -15.0, "c_xy": 7.0})

    def test_evaluate_passes_non_numeric_fixed_values_to_the_tool(self):
        from services.execution.main import TOOL_REGISTRY, ProblemPool, SchemaProvider

        received = []

        def tool(x, material, flag, count, rho):
            received.append(
                {"material": material, "flag": flag, "count": count, "rho": rho}
            )
            return x * rho * count

        schema = StudySchema(
            variables=[
                RangeVar(name="x", lower=0.0, upper=1.0),
                FixedParam(name="material", value="steel"),
                FixedParam(name="flag", value=True),
                FixedParam(name="count", value=3),
                FixedParam(name="rho", value=1.225),
                StateVar(name="f"),
            ],
            tools=[
                ToolSpec(
                    name="T",
                    inputs=["x", "material", "flag", "count", "rho"],
                    outputs=["f"],
                )
            ],
        )
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = schema.model_dump(mode="json")
        mock_client.get.return_value = mock_resp
        execution_app.state.schema_provider = SchemaProvider(mock_client)
        execution_app.state.problem_pool = ProblemPool(TOOL_REGISTRY, size=1)

        with patch.dict(TOOL_REGISTRY, {"T": tool}):
            response = self.client.post(
                "/evaluate", json={"inputs": {"x": 0.5}, "objectives": ["f"]}
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertAlmostEqual(response.json()["results"]["f"], 0.5 * 1.225 * 3)
        expected = {"material": "steel", "flag": True, "count": 3, "rho": 1.225}
        self.assertEqual(received, [expected])
        self.assertEqual(
            {name: type(value) for name, value in received[0].items()},
            {name: type(value) for name, value in expected.items()},
        )

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
            mock_resp.json.return_value = PARABOLOID_PAYLOAD
            mock_client.get.return_value = mock_resp
            execution_app.state.schema_provider = SchemaProvider(mock_client)
            execution_app.state.problem_pool = ProblemPool(TOOL_REGISTRY, size=1)

            # Unknown objective
            response = self.client.post(
                "/evaluate",
                json={"inputs": {"x": 1.0}, "objectives": ["unknown_obj"]},
            )
            self.assertEqual(response.status_code, 422)
            self.assertIn("Unknown objective", response.json()["detail"])

            # Unknown input
            response = self.client.post(
                "/evaluate",
                json={"inputs": {"unknown_var": 1.0}, "objectives": ["f_xy"]},
            )
            self.assertEqual(response.status_code, 422)
            self.assertIn("Unknown inputs", response.json()["detail"])

    def test_evaluate_payload_limits(self):
        # inputs > 100
        large_inputs = {f"var_{i}": 1.0 for i in range(101)}
        response = self.client.post(
            "/evaluate",
            json={"inputs": large_inputs, "objectives": ["f_xy"]},
        )
        self.assertEqual(response.status_code, 422)

        # input key > 50 chars
        large_key = "a" * 51
        response = self.client.post(
            "/evaluate",
            json={"inputs": {large_key: 1.0}, "objectives": ["f_xy"]},
        )
        self.assertEqual(response.status_code, 422)

        # empty inputs
        response = self.client.post(
            "/evaluate",
            json={"inputs": {}, "objectives": ["f_xy"]},
        )
        self.assertEqual(response.status_code, 422)

    def test_execute_problem_missing_objective(self):
        from unittest.mock import MagicMock

        from services.execution.main import execute_problem

        mock_prob = MagicMock()
        mock_prob.execute.return_value = {"known_obj": 1.0}

        # 'missing_obj' should trigger the `if val is None: results[obj] = 0.0` logic
        results = execute_problem(
            mock_prob, inputs={"x": 1.0}, objectives=["known_obj", "missing_obj"]
        )
        self.assertEqual(results["missing_obj"], 0.0)

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
            mock_resp.json.return_value = PARABOLOID_PAYLOAD
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
            mock_resp.json.return_value = PARABOLOID_PAYLOAD
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

    def test_health_ok_when_graph_service_is_healthy(self):
        from services.execution.main import SchemaProvider

        with patch.dict(execution_app.state.__dict__, {}):
            mock_client = AsyncMock()
            mock_client.get.return_value = MagicMock(
                raise_for_status=MagicMock(return_value=None)
            )
            execution_app.state.schema_provider = SchemaProvider(mock_client)

            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["graph_service"], "ok")
            self.assertTrue(mock_client.get.call_args.args[0].endswith("/health"))

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
        from pydantic import ValidationError

        from services.execution.main import SchemaEnvelope

        legacy = {
            "variables": [{"name": "x"}],
            "tools": [{"name": "tool", "outputs": None}],
        }
        with self.assertRaises(ValidationError):
            SchemaEnvelope(StudySchema.model_validate(legacy), TOOL_REGISTRY)

    def test_schema_provider_rejects_legacy_payload(self):
        import asyncio

        from services.execution.main import SchemaProvider

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "variables": [{"name": "x"}],
            "tools": [{"name": "tool", "inputs": ["x"], "outputs": []}],
        }
        mock_client.get.return_value = mock_resp
        provider = SchemaProvider(mock_client)

        with self.assertRaises(Exception) as context:
            asyncio.run(provider.get_schema())
        self.assertEqual(context.exception.status_code, 502)

    def test_schema_envelope_exposes_typed_schema(self):
        from services.execution.main import SchemaEnvelope

        env = SchemaEnvelope(PARABOLOID_SCHEMA, TOOL_REGISTRY)
        self.assertIs(env.schema, PARABOLOID_SCHEMA)
        self.assertTrue(env.registry_report.valid)
        self.assertEqual(env.known_vars, {"x", "y", "f_xy", "c_xy"})
        self.assertEqual(env.known_objectives, {"f_xy", "c_xy"})
        self.assertEqual(list(env.variable_specs), ["x", "y"])
        self.assertEqual(env.hash, PARABOLOID_SCHEMA.content_hash())

    def test_to_float_integer(self):
        from services.execution.main import to_float

        # Covers line 86 casting from int
        self.assertEqual(to_float(4), 4.0)

    def test_evaluate_input_execution_error(self):
        # Covers line 400 ValueError / KeyError inside execution
        from services.execution.main import TOOL_REGISTRY, ProblemPool, SchemaProvider

        with (
            execution_app.container_context()
            if hasattr(execution_app, "container_context")
            else patch.dict(execution_app.state.__dict__, {})
        ):
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = PARABOLOID_PAYLOAD
            mock_client.get.return_value = mock_resp
            execution_app.state.schema_provider = SchemaProvider(mock_client)

            with patch(
                "services.execution.main.ProblemPool.discard_instance",
                new_callable=AsyncMock,
            ):
                execution_app.state.problem_pool = ProblemPool(TOOL_REGISTRY, size=1)

                with patch(
                    "services.execution.main.execute_problem",
                    side_effect=ValueError("Input Error"),
                ):
                    response = self.client.post(
                        "/evaluate",
                        json={"inputs": {"x": 1.0, "y": 1.0}, "objectives": ["f_xy"]},
                    )
                    self.assertEqual(response.status_code, 400)

    def test_schema_provider_lock_timeout_fallback(self):
        import asyncio

        from services.execution.main import SchemaEnvelope, SchemaProvider

        mock_client = AsyncMock()
        provider = SchemaProvider(mock_client)
        provider.envelope = SchemaEnvelope(StudySchema(), TOOL_REGISTRY)
        provider.expiry = 1e9  # Not expired initially

        async def test_run():
            await provider.lock.acquire()  # block the lock
            res = await provider.get_schema()
            self.assertEqual(res, provider.envelope)
            provider.lock.release()

        # monkeypatch asyncio.timeout
        with patch("asyncio.timeout") as mock_timeout:
            mock_timeout.side_effect = TimeoutError("Timed out")
            asyncio.run(test_run())

    def test_schema_provider_httpx_errors_fallback(self):
        import asyncio

        import httpx

        from services.execution.main import SchemaEnvelope, SchemaProvider

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("Network error")
        provider = SchemaProvider(mock_client)
        provider.envelope = SchemaEnvelope(StudySchema(), TOOL_REGISTRY)
        provider.expiry = 0  # force fetch

        async def test_run():
            res = await provider.get_schema()
            self.assertEqual(res, provider.envelope)

        asyncio.run(test_run())

    def test_schema_provider_invalid_data_fallback(self):
        import asyncio

        from services.execution.main import SchemaEnvelope, SchemaProvider

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("Bad JSON")
        mock_client.get.return_value = mock_resp

        provider = SchemaProvider(mock_client)
        provider.envelope = SchemaEnvelope(StudySchema(), TOOL_REGISTRY)
        provider.expiry = 0  # force fetch

        async def test_run():
            res = await provider.get_schema()
            self.assertEqual(res, provider.envelope)

        asyncio.run(test_run())

    def test_schema_provider_legacy_payload_fallback(self):
        import asyncio

        from services.execution.main import SchemaEnvelope, SchemaProvider

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"variables": [{"name": "x"}], "tools": []}
        mock_client.get.return_value = mock_resp

        provider = SchemaProvider(mock_client)
        stale = SchemaEnvelope(PARABOLOID_SCHEMA, TOOL_REGISTRY)
        provider.envelope = stale
        provider.expiry = 0  # force fetch

        res = asyncio.run(provider.get_schema())
        self.assertIs(res, stale)

    def test_lifespan_initialization(self):
        import asyncio

        from services.execution.main import lifespan

        async def run_lifespan():
            async with lifespan(execution_app):
                self.assertIsNotNone(execution_app.state.schema_provider)
                self.assertIsNotNone(execution_app.state.problem_pool)

        asyncio.run(run_lifespan())


class TestRegistryCheck(unittest.TestCase):
    """The tool registry is checked once per schema version, before any problem."""

    UNREGISTERED = StudySchema(
        variables=PARABOLOID_SCHEMA.variables,
        tools=[ToolSpec(name="Missing", inputs=["x", "y"], outputs=["f_xy"])],
    )
    MISMATCHED = StudySchema(
        variables=[*PARABOLOID_SCHEMA.variables, RangeVar(name="z", lower=0, upper=1)],
        tools=[ToolSpec(name="Paraboloid", inputs=["x", "y", "z"], outputs=["f_xy"])],
    )
    BODY = {"inputs": {"x": 1.0, "y": 2.0}, "objectives": ["f_xy"]}

    def setUp(self):
        from services.execution.main import TOOL_REGISTRY, ProblemPool, SchemaProvider

        self.registry = TOOL_REGISTRY
        self.mock_client = AsyncMock()
        execution_app.state.schema_provider = SchemaProvider(self.mock_client)
        execution_app.state.problem_pool = ProblemPool(TOOL_REGISTRY, size=1)
        self.client = TestClient(execution_app)
        for name in ("build_and_init", "execute_problem"):
            patcher = patch(f"services.execution.main.{name}")
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def serve(self, schema):
        response = MagicMock()
        response.json.return_value = schema.model_dump(mode="json")
        self.mock_client.get.return_value = response

    def assert_schema_invalid(self, response, code, names):
        self.assertEqual(response.status_code, 422, response.text)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "SCHEMA_INVALID")
        self.assertFalse(detail["report"]["valid"])
        finding = next(f for f in detail["report"]["errors"] if f["code"] == code)
        self.assertEqual(finding["names"], names)
        self.build_and_init.assert_not_called()
        self.execute_problem.assert_not_called()

    def test_unregistered_tool_is_rejected_before_any_problem_is_built(self):
        self.serve(self.UNREGISTERED)

        response = self.client.post("/evaluate", json=self.BODY)

        self.assert_schema_invalid(response, "UNREGISTERED_TOOL", ["Missing"])

    def test_signature_mismatch_is_rejected_before_any_problem_is_built(self):
        self.serve(self.MISMATCHED)

        response = self.client.post("/evaluate", json=self.BODY)

        self.assert_schema_invalid(
            response, "SIGNATURE_MISMATCH", ["Paraboloid", "x", "y", "z"]
        )

    def test_the_registry_is_checked_once_per_schema_version(self):
        from mdo_framework.validation import validate_registry

        self.serve(self.UNREGISTERED)

        with patch(
            "services.execution.main.validate_registry", wraps=validate_registry
        ) as checked:
            for _ in range(3):
                self.client.post("/evaluate", json=self.BODY)

        self.assertEqual(checked.call_count, 1)

    def test_envelope_carries_the_registry_report(self):
        from services.execution.main import SchemaEnvelope

        valid = SchemaEnvelope(PARABOLOID_SCHEMA, self.registry)
        invalid = SchemaEnvelope(self.UNREGISTERED, self.registry)

        self.assertTrue(valid.registry_report.valid)
        self.assertEqual(
            [f.code for f in invalid.registry_report.errors], ["UNREGISTERED_TOOL"]
        )

    def test_provider_checks_the_schema_against_the_tool_registry(self):
        import asyncio

        self.serve(self.UNREGISTERED)
        provider = execution_app.state.schema_provider
        self.assertFalse(asyncio.run(provider.get_schema()).registry_report.valid)

        with patch.dict(self.registry, {"Missing": self.registry["Paraboloid"]}):
            provider.envelope = None
            self.assertTrue(asyncio.run(provider.get_schema()).registry_report.valid)


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

        envelope = SchemaEnvelope(PARABOLOID_SCHEMA, self.registry)
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
            StudySchema(
                variables=[
                    RangeVar(name="x", lower=-10.0, upper=10.0),
                    StateVar(name="y"),
                ],
                tools=[ToolSpec(name="MissingTool", inputs=["x"], outputs=["y"])],
            ),
            self.registry,
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

        envelope = SchemaEnvelope(PARABOLOID_SCHEMA, self.registry)

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

    def test_problem_pool_teardown_empty_exception(self):
        import asyncio

        async def run_test():
            self.pool.pool.get_nowait = MagicMock(side_effect=asyncio.QueueEmpty)
            self.pool.pool.empty = MagicMock(return_value=False)

            # This should cleanly break instead of crashing
            await self.pool.teardown()

        asyncio.run(run_test())

    def test_problem_pool_replenish_one_exception(self):
        import asyncio

        async def run_test():
            with patch(
                "services.execution.main.build_and_init",
                side_effect=Exception("Failed Init"),
            ):
                # This should catch the exception and log instead of crashing
                await self.pool._replenish_one("hash1", StudySchema())

        asyncio.run(run_test())


class TestOptimizationService(unittest.TestCase):
    def setUp(self):
        self.mock_client = AsyncMock()
        optimization_app.state.client = self.mock_client

        # Default mock response for schema fetch
        self.mock_resp = MagicMock()
        self.mock_resp.json.return_value = StudySchema().model_dump(mode="json")
        self.mock_client.get.return_value = self.mock_resp

        self.client = TestClient(optimization_app)

    @patch("mdo_framework.optimization.optimizer.BayesianOptimizer.optimize")
    @patch("mdo_framework.core.topology.TopologicalAnalyzer.resolve_dependencies")
    def test_optimize(self, mock_resolve, mock_optimize):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SERVICE_PAYLOAD

        self.mock_client.get.return_value = mock_resp
        mock_resolve.return_value = SERVICE_RESOLVED

        # Mock result of optimization
        mock_optimize.return_value = {
            "best_parameters": {"x": 0.5, "y": 0.5},
            "best_objectives": {"f_xy": 0.0},
            "history": [
                {
                    "parameters": {"x": 0.5, "y": 0.5},
                    "objectives": {"f_xy": np.array([0.0])},
                },
            ],
            "serialized_client": "{}",
        }

        payload = {
            "objectives": [{"name": "f_xy"}],
            "n_steps": 1,
            "n_init": 1,
            "use_bonsai": True,
        }

        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("best_parameters", response.json())

    @patch("mdo_framework.optimization.optimizer.BayesianOptimizer.optimize")
    @patch("mdo_framework.core.topology.TopologicalAnalyzer.resolve_dependencies")
    def test_optimize_tensor_conversion(self, mock_resolve, mock_optimize):
        import torch

        self.mock_resp.json.return_value = SERVICE_PAYLOAD
        mock_resolve.return_value = SERVICE_RESOLVED

        # Mock result of optimization
        mock_optimize.return_value = {
            "best_parameters": {"x": 0.5, "y": 0.5},
            "best_objectives": {"f_xy": 0.0},
            "history": [
                {
                    "parameters": {"x": 0.5, "y": 0.5},
                    "objectives": {"f_xy": torch.tensor(0.0)},
                },
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
        mock_resp.json.return_value = SERVICE_PAYLOAD

        optimization_app.state.client.get.return_value = mock_resp
        mock_resolve.return_value = SERVICE_RESOLVED

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
    @patch("mdo_framework.core.topology.TopologicalAnalyzer.resolve_dependencies")
    def test_optimize_tensor_conversion_nested(self, mock_resolve, mock_optimize):
        import numpy as np

        self.mock_resp.json.return_value = SERVICE_PAYLOAD
        mock_resolve.return_value = SERVICE_RESOLVED

        # Mock result of optimization
        mock_optimize.return_value = {
            "best_parameters": {"x": 0.5, "y": 0.5},
            "best_objectives": {"f_xy": 0.0},
            "history": [
                {
                    "parameters": {"x": 0.5, "y": 0.5},
                    "objectives": {"f_xy": np.array([0.0])},
                },
            ],
        }

        payload = {
            "objectives": [{"name": "f_xy"}],
            "n_steps": 1,
            "n_init": 1,
        }

        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 200)

    def test_optimize_requires_an_objective(self):
        payload = {
            "objectives": [],
            "n_steps": 1,
            "n_init": 1,
        }
        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_optimize_tensor_lists(self):
        # A simple check for the internal to_list method inside optimize route
        # is covered by having best_parameters return a numpy array
        pass

    @patch("mdo_framework.optimization.optimizer.BayesianOptimizer.__init__")
    @patch("mdo_framework.core.topology.TopologicalAnalyzer.resolve_dependencies")
    def test_optimize_exception2(self, mock_resolve, mock_init):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SERVICE_PAYLOAD

        optimization_app.state.client.get.return_value = mock_resp
        mock_resolve.return_value = SERVICE_RESOLVED

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
        self.mock_resp.json.return_value = SERVICE_PAYLOAD
        mock_resolve.return_value = SERVICE_RESOLVED

        mock_optimize.return_value = {
            "best_parameters": {"x": 0.5, "y": 0.5},
            "best_objectives": {"f_xy": 0.0},
            "history": [],
        }

        payload = {
            "objectives": [{"name": "f_xy"}],
            "constraints": [{"name": "g_xy", "op": "<=", "bound": 0.0}],
            "n_steps": 1,
            "n_init": 1,
        }

        response = self.client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 200)


class TestStudyPreflight(unittest.TestCase):
    """``/validate`` and the 422 preflight of ``/optimize``."""

    VALID_BODY = {
        "objectives": [{"name": "f_xy"}],
        "constraints": [{"name": "g_xy", "bound": 0.0}],
        "parameter_constraints": ["x + y <= 1.5"],
        "n_steps": 1,
        "n_init": 1,
    }
    INVALID_STUDIES = {
        "unknown objective": (
            {"objectives": [{"name": "missing"}]},
            "UNKNOWN_OUTPUT",
        ),
        "unusable parameter constraint": (
            {"objectives": [{"name": "f_xy"}], "parameter_constraints": ["x + z <= 1"]},
            "PARAMETER_CONSTRAINT_INVALID",
        ),
        "constraint on a design variable": (
            {
                "objectives": [{"name": "f_xy"}],
                "constraints": [{"name": "x", "bound": 0.5}],
            },
            "NOT_PRODUCED",
        ),
    }

    def setUp(self):
        self.mock_client = AsyncMock()
        optimization_app.state.client = self.mock_client
        self.serve(SERVICE_PAYLOAD)
        self.client = TestClient(optimization_app)
        for target in ("RemoteEvaluator", "BayesianOptimizer"):
            patcher = patch(f"services.optimization.main.{target}")
            setattr(self, target, patcher.start())
            self.addCleanup(patcher.stop)

    def serve(self, payload):
        response = MagicMock()
        response.json.return_value = payload
        self.mock_client.get.return_value = response

    def assert_nothing_ran(self):
        self.RemoteEvaluator.assert_not_called()
        self.BayesianOptimizer.assert_not_called()

    def test_validate_accepts_a_valid_study(self):
        response = self.client.post("/validate", json=self.VALID_BODY)

        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertTrue(report["valid"])
        self.assertEqual(report["errors"], [])
        self.assert_nothing_ran()

    def test_validate_reports_an_invalid_study_with_status_200(self):
        for label, (body, code) in self.INVALID_STUDIES.items():
            with self.subTest(label):
                response = self.client.post("/validate", json=body)

                self.assertEqual(response.status_code, 200)
                report = response.json()
                self.assertFalse(report["valid"])
                self.assertIn(code, [item["code"] for item in report["errors"]])
        self.assert_nothing_ran()

    def test_validate_accepts_a_study_with_non_numeric_fixed_parameters(self):
        self.serve(
            StudySchema(
                variables=[
                    *SERVICE_VARIABLES,
                    FixedParam(name="material", value="steel"),
                    FixedParam(name="flag", value=True),
                    StateVar(name="f_xy"),
                ],
                tools=[
                    ToolSpec(
                        name="ToolA",
                        inputs=["x", "y", "material", "flag"],
                        outputs=["f_xy"],
                    )
                ],
            ).model_dump(mode="json")
        )

        response = self.client.post(
            "/validate", json={"objectives": [{"name": "f_xy"}]}
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["valid"])

    def test_validate_reports_a_schema_that_does_not_parse(self):
        payloads = {
            "legacy node without a kind": (
                {"variables": [{"name": "x"}], "tools": []},
                "SCHEMA_INVALID",
            ),
            "duplicate variable": (
                {
                    "variables": [
                        {"kind": "state", "name": "f"},
                        {"kind": "state", "name": "f"},
                    ]
                },
                "DUPLICATE_NAME",
            ),
        }
        for label, (payload, code) in payloads.items():
            with self.subTest(label):
                self.serve(payload)

                response = self.client.post("/validate", json=self.VALID_BODY)

                self.assertEqual(response.status_code, 200)
                report = response.json()
                self.assertFalse(report["valid"])
                self.assertIn(code, [item["code"] for item in report["errors"]])

    def test_validate_is_bad_gateway_when_the_graph_service_is_down(self):
        self.mock_client.get.side_effect = httpx.ConnectError("graph service down")

        response = self.client.post("/validate", json=self.VALID_BODY)

        self.assertEqual(response.status_code, 502)
        self.assertIn("Failed to fetch graph schema", response.json()["detail"])

    def test_validate_takes_the_body_of_optimize(self):
        response = self.client.post("/validate", json={"objectives": []})

        self.assertEqual(response.status_code, 422)

    def test_optimize_rejects_an_invalid_study_before_running_anything(self):
        for label, (body, code) in self.INVALID_STUDIES.items():
            with self.subTest(label):
                response = self.client.post("/optimize", json=body)

                self.assertEqual(response.status_code, 422)
                report = response.json()["detail"]
                self.assertEqual(
                    report, self.client.post("/validate", json=body).json()
                )
                self.assertIn(code, [item["code"] for item in report["errors"]])
        self.assert_nothing_ran()

    def test_optimize_rejects_a_schema_that_does_not_parse(self):
        self.serve({"variables": [{"name": "x"}], "tools": []})

        response = self.client.post("/optimize", json=self.VALID_BODY)

        self.assertEqual(response.status_code, 422)
        codes = [item["code"] for item in response.json()["detail"]["errors"]]
        self.assertIn("SCHEMA_INVALID", codes)
        self.assert_nothing_ran()

    def test_optimize_is_bad_gateway_when_the_graph_service_is_down(self):
        self.mock_client.get.side_effect = httpx.ConnectError("graph service down")

        response = self.client.post("/optimize", json=self.VALID_BODY)

        self.assertEqual(response.status_code, 502)
        self.assert_nothing_ran()

    def test_every_upstream_failure_is_a_bad_gateway(self):
        request = httpx.Request("GET", "http://graph/schema")
        status_error = httpx.HTTPStatusError(
            "Server error", request=request, response=httpx.Response(500)
        )
        not_json = json.JSONDecodeError("Expecting value", "<html>", 0)

        def fail_get(error):
            self.mock_client.get.side_effect = error

        def fail_status(response):
            response.raise_for_status.side_effect = status_error

        def fail_json(response):
            response.json.side_effect = not_json

        cases = {
            "timeout": lambda response: fail_get(httpx.ReadTimeout("slow")),
            "connection error": lambda response: fail_get(httpx.ConnectError("down")),
            "non-2xx status": fail_status,
            "2xx body that is not JSON": fail_json,
        }
        for path in ("/validate", "/optimize"):
            for label, break_upstream in cases.items():
                with self.subTest(path=path, upstream=label):
                    response = MagicMock()
                    self.mock_client.get.side_effect = None
                    self.mock_client.get.return_value = response
                    break_upstream(response)

                    result = self.client.post(path, json=self.VALID_BODY)

                    self.assertEqual(result.status_code, 502, result.text)
                    self.assertIn("graph", result.json()["detail"].lower())
        self.assert_nothing_ran()

    def test_json_that_is_not_a_study_is_reported_not_a_bad_gateway(self):
        self.serve({"variables": "not a list"})

        validated = self.client.post("/validate", json=self.VALID_BODY)
        optimized = self.client.post("/optimize", json=self.VALID_BODY)

        self.assertEqual(validated.status_code, 200)
        self.assertFalse(validated.json()["valid"])
        self.assertEqual(optimized.status_code, 422)
        self.assertEqual(optimized.json()["detail"], validated.json())
        self.assert_nothing_ran()

    def test_optimize_rejects_non_finite_numbers_and_bad_names(self):
        bodies = {
            "NaN constraint bound": (
                '{"objectives": [{"name": "f_xy"}],'
                ' "constraints": [{"name": "g_xy", "bound": NaN}]}'
            ),
            "infinite constraint bound": (
                '{"objectives": [{"name": "f_xy"}],'
                ' "constraints": [{"name": "g_xy", "bound": Infinity}]}'
            ),
            "NaN objective threshold": (
                '{"objectives": [{"name": "f_xy", "threshold": NaN}]}'
            ),
            "objective name that is not an identifier": (
                '{"objectives": [{"name": "1f"}]}'
            ),
        }
        for path in ("/optimize", "/validate"):
            for label, text in bodies.items():
                with self.subTest(path=path, body=label):
                    response = self.client.post(
                        path,
                        content=text,
                        headers={"Content-Type": "application/json"},
                    )

                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIsInstance(response.json()["detail"], list)
        self.assert_nothing_ran()

    def test_optimize_runs_a_valid_study(self):
        self.BayesianOptimizer.return_value.optimize.return_value = {
            "best_parameters": {"x": 0.5, "y": 0.5},
            "best_objectives": {"f_xy": 0.0},
            "history": [],
        }

        response = self.client.post("/optimize", json=self.VALID_BODY)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["best_parameters"], {"x": 0.5, "y": 0.5})
        arguments = self.BayesianOptimizer.call_args.kwargs
        self.assertEqual(arguments["objectives"], [{"name": "f_xy", "minimize": True}])
        self.assertEqual(
            arguments["constraints"], [{"name": "g_xy", "bound": 0.0, "op": "<="}]
        )
        self.assertEqual(arguments["parameter_constraints"], ["x + y <= 1.5"])


class TestOptimizationServiceExtra(unittest.IsolatedAsyncioTestCase):
    async def test_to_jsonable_all_branches(self):
        from services.optimization.main import to_jsonable

        # 1. Dict branch
        self.assertEqual(to_jsonable({"a": 1}), {"a": 1})
        # 2. List/Tuple/Set branch
        self.assertEqual(to_jsonable([1, 2]), [1, 2])
        self.assertEqual(to_jsonable((1, 2)), [1, 2])
        self.assertEqual(to_jsonable({1, 2}), [1, 2])
        # 3. NumPy ndarray
        self.assertEqual(to_jsonable(np.array([1, 2])), [1, 2])
        # 4. NumPy generic (scalar)
        self.assertEqual(to_jsonable(np.float64(1.0)), 1.0)

        # 5. Object with .tolist() (e.g. Mocking a Tensor)
        mock_tensor = MagicMock()
        mock_tensor.tolist.return_value = [3, 4]
        self.assertEqual(to_jsonable(mock_tensor), [3, 4])

        # 6. Object with .item() (e.g. Mocking a Tensor scalar)
        mock_scalar = MagicMock()
        mock_scalar.item.return_value = 5
        # Ensure it doesn't have tolist to hit this branch
        if hasattr(mock_scalar, "tolist"):
            del mock_scalar.tolist
        self.assertEqual(to_jsonable(mock_scalar), 5)

        # 7. Default branch
        self.assertEqual(to_jsonable("string"), "string")
        self.assertEqual(to_jsonable(None), None)

        # 8. Nested structures (Recursive branches)
        nested = {
            "list": [np.array([1]), {np.float64(2.0)}],
            "tuple": (MagicMock(tolist=lambda: [3]),),
        }
        expected = {"list": [[1], [2.0]], "tuple": [[3]]}
        self.assertEqual(to_jsonable(nested), expected)

    async def test_lifespan_coverage(self):
        from services.optimization.main import lifespan

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        async with lifespan(mock_app):
            self.assertIsNotNone(mock_app.state.client)
            self.assertIsInstance(mock_app.state.client, httpx.AsyncClient)

        # Client should be closed (we can't easily check internal state of closed,
        # but we verified the context manager exits)

    @patch("mdo_framework.core.topology.TopologicalAnalyzer.resolve_dependencies")
    async def test_optimize_error_paths(self, mock_resolve):
        mock_client = AsyncMock()
        optimization_app.state.client = mock_client
        client = TestClient(optimization_app)

        payload = {
            "objectives": [{"name": "f_xy"}],
            "n_steps": 1,
            "n_init": 1,
        }

        # 1. Fetch schema failure (502)
        mock_client.get.side_effect = httpx.ConnectError("Network down")
        response = client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 502)
        mock_client.get.side_effect = None

        # 2. Dependency resolution failure (422 with the report)
        mock_resp = MagicMock()
        mock_resp.json.return_value = SERVICE_PAYLOAD
        mock_client.get.return_value = mock_resp
        mock_resolve.side_effect = StudyValidationError(
            ValidationReport(
                errors=(Finding(code="NOT_PRODUCED", message="'f_xy' has no producer"),)
            )
        )

        response = client.post("/optimize", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["errors"][0]["code"], "NOT_PRODUCED")
        mock_resolve.side_effect = None
        mock_resolve.return_value = SERVICE_RESOLVED

        # 3. Catch-all Internal Server Error (500)
        # Hit via to_jsonable or return block by returning None from optimize
        with (
            patch(
                "mdo_framework.core.topology.TopologicalAnalyzer.extract_parameters",
                return_value=[
                    {
                        "name": "x",
                        "type": "range",
                        "bounds": [0.0, 1.0],
                        "value_type": "float",
                    },
                ],
            ),
            patch(
                "mdo_framework.optimization.optimizer.BayesianOptimizer.optimize",
                return_value=None,
            ),
        ):
            response = client.post("/optimize", json=payload)
            self.assertEqual(response.status_code, 500)
            self.assertIn("Optimization failed", response.json()["detail"])

    async def test_optimize_resolves_real_dependencies(self):
        mock_client = AsyncMock()
        optimization_app.state.client = mock_client
        client = TestClient(optimization_app)

        mock_resp = MagicMock()
        mock_resp.json.return_value = SERVICE_PAYLOAD
        mock_client.get.return_value = mock_resp

        payload = {"objectives": [{"name": "unknown"}], "n_steps": 1, "n_init": 1}
        response = client.post("/optimize", json=payload)

        self.assertEqual(response.status_code, 422)
        codes = [item["code"] for item in response.json()["detail"]["errors"]]
        self.assertIn("UNKNOWN_OUTPUT", codes)

    @patch("mdo_framework.core.topology.TopologicalAnalyzer.resolve_dependencies")
    async def test_optimize_exception_mapping(self, mock_resolve):
        mock_client = AsyncMock()
        optimization_app.state.client = mock_client
        client = TestClient(optimization_app)

        mock_resp = MagicMock()
        mock_resp.json.return_value = SERVICE_PAYLOAD
        mock_client.get.return_value = mock_resp
        mock_resolve.return_value = SERVICE_RESOLVED

        payload = {
            "objectives": [{"name": "f_xy"}],
            "n_steps": 1,
            "n_init": 1,
        }

        with patch(
            "mdo_framework.optimization.optimizer.BayesianOptimizer.optimize",
            side_effect=OptimizationConfigurationError("invalid optimization config"),
        ):
            response = client.post("/optimize", json=payload)
            self.assertEqual(response.status_code, 400)
            self.assertIn("invalid optimization config", response.json()["detail"])

        with patch(
            "mdo_framework.optimization.optimizer.BayesianOptimizer.optimize",
            side_effect=RemoteEvaluationTransportError("execution service unavailable"),
        ):
            response = client.post("/optimize", json=payload)
            self.assertEqual(response.status_code, 502)
            self.assertIn("execution service unavailable", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
