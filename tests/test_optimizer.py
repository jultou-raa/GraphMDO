"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import contextlib
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from gemseo.core.discipline import Discipline

from mdo_framework.core.errors import (
    MDANotConvergedError,
    ToolExecutionError,
    ToolOutputError,
)
from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.optimization.optimizer import (
    BayesianOptimizer,
    OptimizationConfigurationError,
    OptimizationExecutionError,
    RemoteDiscipline,
    RemoteEvaluationContractError,
    RemoteEvaluationTransportError,
    RemoteEvaluator,
)
from mdo_framework.schema import (
    ChoiceVar,
    ConstraintSpec,
    ObjectiveSpec,
    RangeVar,
)


class OptimizerTestCase(unittest.TestCase):
    def setUp(self):
        # Runs write their XDSM and plots in the working directory.
        workdir = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(contextlib.chdir(workdir))

        class MockDisc(Discipline):
            def __init__(self):
                super().__init__(name="MockDisc")
                self.input_grammar.update_from_names(["x", "y", "c"])
                self.output_grammar.update_from_names(["f_xy", "g_xy"])
                self.default_input_data = {
                    "x": np.array([0.0]),
                    "y": np.array([0.0]),
                    "c": np.array([0.0]),
                }

            def _run(self, input_data):
                x, y, c = (input_data[name] for name in ("x", "y", "c"))
                self.local_data["f_xy"] = (x - 0.25) ** 2 + (y - 0.5) ** 2 + c
                self.local_data["g_xy"] = x - y

        self.mock_prob = MockDisc()
        self.evaluator = LocalEvaluator(self.mock_prob)
        self.design_variables = [
            RangeVar(name="x", lower=0.0, upper=1.0),
            RangeVar(name="y", lower=0.0, upper=1.0),
            RangeVar(name="c", lower=0.0, upper=1.0),
        ]
        self.objectives = [ObjectiveSpec(name="f_xy")]


class TestLocalEvaluator(OptimizerTestCase):
    def test_normalizes_scalar_and_array_outputs(self):
        output_cases = [
            (np.array([1.23]), 1.23),
            (np.array([4.56, 1.0]), 4.56),
        ]

        for raw_output, expected_value in output_cases:
            with self.subTest(raw_output=raw_output):
                fresh_problem = self.mock_prob.__class__()

                def dummy_run(this, input_data, captured_output=raw_output):
                    this.local_data["f_xy"] = captured_output
                    this.local_data["g_xy"] = np.array([0.0])

                fresh_problem._run = dummy_run.__get__(
                    fresh_problem,
                    type(fresh_problem),
                )
                evaluator = LocalEvaluator(fresh_problem)
                result = evaluator.evaluate(
                    {"x": 0.5, "y": 0.5, "c": 0.0},
                    ["f_xy"],
                )
                self.assertEqual(result["f_xy"], expected_value)

    def test_raises_when_output_is_missing(self):
        mock_problem = MagicMock()
        mock_problem.execute.return_value = {"g_xy": np.array([0.0])}
        evaluator = LocalEvaluator(mock_problem)

        with self.assertRaises(KeyError):
            evaluator.evaluate({"x": 0.5, "y": 0.5, "c": 0.0}, ["f_xy"])


class TestRemoteEvaluator(unittest.TestCase):
    def test_happy_path(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": {"f_xy": 2.0}}
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        evaluator = RemoteEvaluator("http://fake-url", client=mock_client)
        result = evaluator.evaluate({"x": 0.5, "y": 0.5, "c": 0.0}, ["f_xy"])

        mock_client.post.assert_called_once_with(
            "http://fake-url/evaluate",
            json={
                "inputs": {"x": 0.5, "y": 0.5, "c": 0.0},
                "objectives": ["f_xy"],
            },
        )
        self.assertEqual(result, {"f_xy": 2.0})

    def test_transport_errors(self):
        url = "http://fake-url/evaluate"
        request = httpx.Request("POST", url)
        failure_cases = [
            (
                "timeout",
                MagicMock(post=MagicMock(side_effect=httpx.TimeoutException("slow"))),
            ),
            (
                "server_error",
                MagicMock(
                    post=MagicMock(
                        return_value=MagicMock(
                            raise_for_status=MagicMock(
                                side_effect=httpx.HTTPStatusError(
                                    "server error",
                                    request=request,
                                    response=httpx.Response(503, request=request),
                                )
                            )
                        )
                    )
                ),
            ),
            (
                "request_error",
                MagicMock(
                    post=MagicMock(
                        side_effect=httpx.RequestError("network down", request=request)
                    )
                ),
            ),
        ]

        for label, client in failure_cases:
            with self.subTest(case=label):
                with self.assertRaises(RemoteEvaluationTransportError):
                    RemoteEvaluator("http://fake-url", client=client).evaluate(
                        {}, ["f_xy"]
                    )

    def test_contract_errors(self):
        url = "http://fake-url/evaluate"
        request = httpx.Request("POST", url)
        failure_cases = [
            (
                "http_400",
                MagicMock(
                    post=MagicMock(
                        return_value=MagicMock(
                            raise_for_status=MagicMock(
                                side_effect=httpx.HTTPStatusError(
                                    "bad request",
                                    request=request,
                                    response=httpx.Response(400, request=request),
                                )
                            )
                        )
                    )
                ),
            ),
            (
                "invalid_json",
                MagicMock(
                    post=MagicMock(
                        return_value=MagicMock(
                            raise_for_status=MagicMock(return_value=None),
                            json=MagicMock(side_effect=ValueError("bad json")),
                        )
                    )
                ),
            ),
            (
                "missing_results",
                MagicMock(
                    post=MagicMock(
                        return_value=MagicMock(
                            raise_for_status=MagicMock(return_value=None),
                            json=MagicMock(return_value={}),
                        )
                    )
                ),
            ),
            (
                "missing_objective",
                MagicMock(
                    post=MagicMock(
                        return_value=MagicMock(
                            raise_for_status=MagicMock(return_value=None),
                            json=MagicMock(return_value={"results": {"other": 1.0}}),
                        )
                    )
                ),
            ),
            (
                "non_numeric",
                MagicMock(
                    post=MagicMock(
                        return_value=MagicMock(
                            raise_for_status=MagicMock(return_value=None),
                            json=MagicMock(return_value={"results": {"f_xy": "NaN?"}}),
                        )
                    )
                ),
            ),
        ]

        for label, client in failure_cases:
            with self.subTest(case=label):
                with self.assertRaises(RemoteEvaluationContractError):
                    RemoteEvaluator("http://fake-url", client=client).evaluate(
                        {}, ["f_xy"]
                    )

    @staticmethod
    def _answering(status_code: int, body: object) -> RemoteEvaluator:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(status_code, json=body)
        )
        client = httpx.Client(base_url="http://exec", transport=transport)
        return RemoteEvaluator("http://exec", client=client)

    def test_evaluation_errors_are_rebuilt_as_local_exceptions(self):
        cases = [
            ToolExecutionError("RuntimeError: solver diverged", tool="T"),
            ToolOutputError("non-finite values for ['f']", tool="T"),
            MDANotConvergedError("residual 1e+00 > 1e-06 (couplings: f, g)"),
        ]
        for error in cases:
            with self.subTest(code=error.code):
                evaluator = self._answering(422, {"detail": error.to_payload()})

                with self.assertRaises(type(error)) as raised:
                    evaluator.evaluate({"x": 0.5}, ["f"])

                self.assertEqual(str(raised.exception), str(error))
                self.assertEqual(raised.exception.tool, error.tool)
                self.assertIsInstance(raised.exception.__cause__, httpx.HTTPStatusError)

    def test_contract_errors_include_the_server_detail(self):
        cases = [
            (422, {"detail": "Unknown inputs: {'z'}"}, "Unknown inputs"),
            (400, {"detail": "invalid choice 'd' for 'm'"}, "invalid choice"),
            (
                422,
                {"detail": {"code": "SCHEMA_INVALID", "report": {"errors": []}}},
                "SCHEMA_INVALID",
            ),
            (422, ["not", "a", "detail"], "HTTP 422"),
        ]
        for status_code, body, expected in cases:
            with self.subTest(body=body):
                evaluator = self._answering(status_code, body)

                with self.assertRaises(RemoteEvaluationContractError) as raised:
                    evaluator.evaluate({"x": 0.5}, ["f"])

                self.assertIn(expected, str(raised.exception))
                self.assertIn(f"HTTP {status_code}", str(raised.exception))

    def test_non_json_error_body_is_a_contract_error(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(418, text="<html>teapot</html>")
        )
        client = httpx.Client(base_url="http://exec", transport=transport)

        with self.assertRaises(RemoteEvaluationContractError) as raised:
            RemoteEvaluator("http://exec", client=client).evaluate({"x": 0.5}, ["f"])

        self.assertIn("HTTP 418", str(raised.exception))

    @patch("mdo_framework.optimization.optimizer.httpx.Client")
    def test_close_only_closes_owned_client(self, mock_httpx_client):
        owned_client = MagicMock()
        mock_httpx_client.return_value = owned_client

        owned_evaluator = RemoteEvaluator("http://owned")
        owned_evaluator.close()
        owned_client.close.assert_called_once()

        external_client = MagicMock()
        external_evaluator = RemoteEvaluator("http://external", client=external_client)
        external_evaluator.close()
        external_client.close.assert_not_called()


class TestOptimizerHelpers(OptimizerTestCase):
    def test_build_design_space_contracts(self):
        from mdo_framework.optimization.optimizer import _build_design_space

        design_space = _build_design_space(
            [
                RangeVar(name="count", lower=0, upper=5, value_type="int"),
                RangeVar(name="x", lower=-1.0, upper=2.0),
                ChoiceVar(name="c", choices=["A", "B", "C"]),
            ]
        )

        self.assertEqual(design_space.variable_names, ["count", "x", "c"])
        self.assertEqual(float(design_space.get_lower_bound("count")[0]), 0.0)
        self.assertEqual(float(design_space.get_upper_bound("count")[0]), 5.0)
        self.assertEqual(float(design_space.get_upper_bound("x")[0]), 2.0)
        self.assertEqual(float(design_space.get_lower_bound("c")[0]), 0.0)
        self.assertEqual(float(design_space.get_upper_bound("c")[0]), 2.0)
        self.assertEqual(
            [design_space.get_type(name)[0] for name in ("count", "x", "c")],
            ["i", "f", "i"],
        )

    def test_optimizer_rejects_empty_objectives(self):
        with self.assertRaisesRegex(
            OptimizationConfigurationError, "At least one objective is required"
        ):
            BayesianOptimizer(self.evaluator, self.design_variables, [])

    def test_optimizer_rejects_empty_design_variables(self):
        with self.assertRaisesRegex(
            OptimizationConfigurationError, "At least one design variable"
        ):
            BayesianOptimizer(self.evaluator, [], self.objectives)

    def test_optimizer_rejects_unknown_algorithm(self):
        with self.assertRaisesRegex(
            OptimizationConfigurationError, "Unknown algorithm 'nope'"
        ):
            BayesianOptimizer(
                self.evaluator,
                self.design_variables,
                self.objectives,
                algorithm="nope",
            )

    def test_bonsai_is_only_accepted_by_a_backend_that_supports_it(self):
        with self.assertRaisesRegex(OptimizationConfigurationError, "use_bonsai"):
            BayesianOptimizer(
                self.evaluator,
                self.design_variables,
                self.objectives,
                use_bonsai=True,
                algorithm="BO_RandomSearch",
            )
        BayesianOptimizer(
            self.evaluator,
            self.design_variables,
            self.objectives,
            use_bonsai=True,
        )

    def test_optimizer_helper_error_wrapping(self):
        from mdo_framework.optimization.optimizer import _decode_parameter_value

        with self.assertRaises(OptimizationConfigurationError):
            _decode_parameter_value({"name": "c", "type": "choice", "values": []}, 0)

        with self.assertRaises(OptimizationExecutionError):
            _decode_parameter_value(
                {"name": "c", "type": "choice", "values": ["A"]},
                True,
            )


class _CountingEvaluator:
    """Evaluator without a GEMSEO problem: it drives the RemoteDiscipline path."""

    def __init__(self, function=None):
        self.function = function or (lambda parameters: {"f_xy": parameters["x"]})
        self.calls: list[dict] = []

    def evaluate(self, parameters, objectives):
        self.calls.append(dict(parameters))
        return self.function(parameters)


class TestBayesianOptimizer(OptimizerTestCase):
    def _optimizer(self, **kwargs):
        kwargs.setdefault("algorithm", "BO_RandomSearch")
        return BayesianOptimizer(
            kwargs.pop("evaluator", self.evaluator),
            kwargs.pop("design_variables", self.design_variables),
            kwargs.pop("objectives", self.objectives),
            **kwargs,
        )

    def test_budgets_are_validated_before_any_tool_call(self):
        evaluator = _CountingEvaluator()
        optimizer = self._optimizer(evaluator=evaluator)

        for name in ("n_steps", "n_init", "max_consecutive_failures"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    OptimizationConfigurationError, f"{name} must be >= 1"
                ):
                    optimizer.optimize(**{name: 0})

        self.assertEqual(evaluator.calls, [])

    def test_optimize_returns_the_result_contract(self):
        result = self._optimizer().optimize(n_steps=3, n_init=2, seed=0)

        self.assertEqual(
            set(result),
            {
                "best_parameters",
                "best_objectives",
                "feasible",
                "constraints",
                "pareto_front",
                "history",
                "stop_reason",
                "evaluations",
            },
        )
        self.assertEqual(result["stop_reason"], "budget")
        self.assertEqual(
            result["evaluations"], {"x0": 0, "init": 2, "bo": 3, "failed": 0}
        )
        self.assertEqual(len(result["history"]), 5)
        self.assertEqual(result["constraints"], {})
        self.assertEqual(result["pareto_front"], [])
        self.assertTrue(result["feasible"])
        objectives = [entry["objectives"]["f_xy"] for entry in result["history"]]
        self.assertEqual(result["best_objectives"], {"f_xy": min(objectives)})
        best = result["history"][int(np.argmin(objectives))]
        self.assertEqual(result["best_parameters"], best["parameters"])

    def test_the_start_point_is_only_evaluated_on_request(self):
        optimizer = self._optimizer()

        default = optimizer.optimize(n_steps=1, n_init=1, seed=0)
        with_x0 = optimizer.optimize(n_steps=1, n_init=1, evaluate_x0=True, seed=0)

        self.assertEqual(len(default["history"]), 2)
        self.assertEqual(len(with_x0["history"]), 3)
        self.assertEqual(with_x0["history"][0]["phase"], "x0")
        self.assertEqual(
            with_x0["history"][0]["parameters"], {"x": 0.5, "y": 0.5, "c": 0.5}
        )
        self.assertEqual(with_x0["evaluations"]["x0"], 1)

    def test_declared_initial_values_enable_the_start_point(self):
        design_variables = [
            RangeVar(name="x", lower=0.0, upper=1.0, initial=0.2),
            RangeVar(name="y", lower=0.0, upper=1.0, initial=0.8),
            RangeVar(name="c", lower=0.0, upper=1.0, initial=0.0),
        ]

        result = self._optimizer(design_variables=design_variables).optimize(
            n_steps=1, n_init=1, seed=0
        )

        self.assertEqual(result["history"][0]["phase"], "x0")
        self.assertEqual(
            result["history"][0]["parameters"], {"x": 0.2, "y": 0.8, "c": 0.0}
        )

    def test_constraints_report_margin_tolerance_and_feasibility(self):
        constraints = [ConstraintSpec(name="g_xy", bound=0.0, tolerance=0.05)]

        result = self._optimizer(constraints=constraints).optimize(
            n_steps=6, n_init=4, seed=0
        )

        self.assertTrue(result["feasible"])
        constraint = result["constraints"]["g_xy"]
        best = result["best_parameters"]
        self.assertAlmostEqual(constraint["value"], best["x"] - best["y"])
        self.assertAlmostEqual(constraint["margin"], -constraint["value"])
        self.assertTrue(constraint["satisfied"])
        self.assertEqual(constraint["tolerance"], 0.05)
        for entry in result["history"]:
            self.assertEqual(
                entry["constraints"]["g_xy"]["satisfied"],
                entry["constraints"]["g_xy"]["margin"] >= -0.05,
            )

    def test_greater_equal_constraints_use_the_user_value(self):
        constraints = [ConstraintSpec(name="g_xy", op=">=", bound=0.2)]

        result = self._optimizer(constraints=constraints).optimize(
            n_steps=6, n_init=4, seed=0
        )

        for entry in result["history"]:
            value = entry["constraints"]["g_xy"]["value"]
            self.assertAlmostEqual(
                value, entry["parameters"]["x"] - entry["parameters"]["y"]
            )
            self.assertAlmostEqual(entry["constraints"]["g_xy"]["margin"], value - 0.2)
        if result["feasible"]:
            self.assertGreaterEqual(result["constraints"]["g_xy"]["value"], 0.2)

    def test_an_infeasible_problem_is_flagged(self):
        constraints = [ConstraintSpec(name="g_xy", bound=-2.0)]

        result = self._optimizer(constraints=constraints).optimize(
            n_steps=2, n_init=2, seed=0
        )

        self.assertFalse(result["feasible"])
        self.assertFalse(result["constraints"]["g_xy"]["satisfied"])
        self.assertEqual(
            result["best_parameters"],
            min(
                result["history"],
                key=lambda entry: entry["constraints"]["g_xy"]["value"] + 2.0,
            )["parameters"],
        )

    def test_maximisation_uses_the_user_direction(self):
        objectives = [ObjectiveSpec(name="f_xy", minimize=False)]

        result = self._optimizer(objectives=objectives).optimize(
            n_steps=3, n_init=2, seed=0
        )

        values = [entry["objectives"]["f_xy"] for entry in result["history"]]
        self.assertEqual(result["best_objectives"], {"f_xy": max(values)})
        self.assertTrue(all(value >= 0.0 for value in values))

    def test_multi_objective_reports_a_pareto_front_of_user_values(self):
        objectives = [
            ObjectiveSpec(name="f_xy"),
            ObjectiveSpec(name="g_xy", minimize=False, threshold=-1.0),
        ]

        result = self._optimizer(objectives=objectives).optimize(
            n_steps=5, n_init=3, seed=0
        )

        front = result["pareto_front"]
        self.assertGreaterEqual(len(front), 1)
        self.assertEqual(set(result["best_objectives"]), {"f_xy", "g_xy"})
        history = [
            (entry["objectives"]["f_xy"], entry["objectives"]["g_xy"])
            for entry in result["history"]
        ]
        for point in front:
            f, g = point["objectives"]["f_xy"], point["objectives"]["g_xy"]
            self.assertIn((f, g), history)
            self.assertFalse(
                any(f2 <= f and g2 >= g and (f2 < f or g2 > g) for f2, g2 in history)
            )
        self.assertIn(
            {
                "parameters": result["best_parameters"],
                "objectives": result["best_objectives"],
            },
            front,
        )

    def test_a_failing_tool_is_a_failed_trial_not_an_abort(self):
        def fails_above_half(parameters):
            if parameters["x"] > 0.5:
                raise ToolExecutionError("mesh failed", tool="Mesher")
            return {"f_xy": parameters["x"]}

        evaluator = _CountingEvaluator(fails_above_half)
        design_variables = [RangeVar(name="x", lower=0.0, upper=1.0)]

        result = self._optimizer(
            evaluator=evaluator, design_variables=design_variables
        ).optimize(n_steps=6, n_init=6, max_consecutive_failures=12, seed=0)

        statuses = {entry["status"] for entry in result["history"]}
        self.assertEqual(statuses, {"completed", "failed"})
        for entry in result["history"]:
            if entry["status"] == "failed":
                self.assertIn("TOOL_FAILED", entry["reason"])
                self.assertEqual(entry["objectives"], {})
        self.assertLessEqual(result["best_parameters"]["x"], 0.5)
        self.assertEqual(
            result["evaluations"]["failed"],
            sum(entry["status"] == "failed" for entry in result["history"]),
        )

    def test_no_completed_trial_raises_with_the_partial_result(self):
        def always_fails(parameters):
            raise ToolExecutionError("mesh failed", tool="Mesher")

        optimizer = self._optimizer(evaluator=_CountingEvaluator(always_fails))

        with self.assertRaisesRegex(
            OptimizationExecutionError, "consecutive_failures.*mesh failed"
        ) as raised:
            optimizer.optimize(n_steps=5, n_init=5, max_consecutive_failures=2)

        partial = raised.exception.partial_result
        self.assertEqual(partial["stop_reason"], "consecutive_failures")
        self.assertIsNone(partial["best_parameters"])
        self.assertFalse(partial["feasible"])
        self.assertEqual(
            [entry["status"] for entry in partial["history"]], ["failed", "failed"]
        )

    def test_an_unexpected_error_aborts_and_keeps_the_history(self):
        def breaks_on_third_call(parameters):
            if len(evaluator.calls) == 3:
                raise RuntimeError("boom")
            return {"f_xy": parameters["x"]}

        evaluator = _CountingEvaluator(breaks_on_third_call)
        optimizer = self._optimizer(evaluator=evaluator)

        with self.assertRaisesRegex(OptimizationExecutionError, "boom") as raised:
            optimizer.optimize(n_steps=4, n_init=4, seed=0)

        partial = raised.exception.partial_result
        self.assertEqual(partial["stop_reason"], "aborted")
        self.assertEqual(
            [entry["status"] for entry in partial["history"]], ["completed"] * 2
        )
        self.assertIsNotNone(partial["best_parameters"])

    def test_remote_errors_keep_their_type_and_carry_the_partial_result(self):
        def unreachable_on_third_call(parameters):
            if len(evaluator.calls) == 3:
                raise RemoteEvaluationTransportError("service down")
            return {"f_xy": parameters["x"]}

        evaluator = _CountingEvaluator(unreachable_on_third_call)
        optimizer = self._optimizer(evaluator=evaluator)

        with self.assertRaises(RemoteEvaluationTransportError) as raised:
            optimizer.optimize(n_steps=4, n_init=4, seed=0)

        self.assertEqual(len(raised.exception.partial_result["history"]), 2)

    def test_configuration_errors_before_the_run_have_no_partial_result(self):
        optimizer = self._optimizer(parameter_constraints=["x <=== y"])

        with self.assertRaises(OptimizationConfigurationError) as raised:
            optimizer.optimize(n_steps=1, n_init=1)

        self.assertIsNone(raised.exception.partial_result)

    def test_choices_are_decoded_for_a_remote_evaluator(self):
        evaluator = _CountingEvaluator(
            lambda parameters: {"f_xy": 0.0 if parameters["c_str"] == "B" else 1.0}
        )
        design_variables = [
            RangeVar(name="x", lower=0.0, upper=1.0),
            ChoiceVar(name="c_str", choices=["A", "B", "C"]),
            ChoiceVar(name="c_num", choices=[42, 7]),
        ]

        result = self._optimizer(
            evaluator=evaluator, design_variables=design_variables
        ).optimize(n_steps=4, n_init=4, seed=0)

        for call in evaluator.calls:
            self.assertIn(call["c_str"], ["A", "B", "C"])
            self.assertIn(call["c_num"], [42, 7])
        self.assertEqual(result["best_parameters"]["c_str"], "B")

    def test_a_seed_makes_a_run_reproducible(self):
        optimizer = self._optimizer()

        first = optimizer.optimize(n_steps=2, n_init=2, seed=3)
        second = optimizer.optimize(n_steps=2, n_init=2, seed=3)
        other = optimizer.optimize(n_steps=2, n_init=2, seed=4)

        self.assertEqual(first["history"], second["history"])
        self.assertNotEqual(first["history"], other["history"])

    def test_a_seeded_ax_run_follows_the_budget(self):
        constraints = [ConstraintSpec(name="g_xy", bound=0.0)]

        result = BayesianOptimizer(
            self.evaluator,
            self.design_variables,
            self.objectives,
            constraints,
            parameter_constraints=["x <= y"],
        ).optimize(n_steps=2, n_init=3, evaluate_x0=True, seed=0)

        self.assertEqual(result["stop_reason"], "budget")
        self.assertEqual(
            result["evaluations"], {"x0": 1, "init": 3, "bo": 2, "failed": 0}
        )
        self.assertTrue(result["feasible"])
        for entry in result["history"]:
            self.assertLessEqual(
                entry["parameters"]["x"], entry["parameters"]["y"] + 1e-9
            )

    def test_a_seeded_ax_multi_objective_run_with_bonsai(self):
        objectives = [
            ObjectiveSpec(name="f_xy", threshold=1000.0),
            ObjectiveSpec(name="g_xy", minimize=False, threshold=-1.0),
        ]

        result = BayesianOptimizer(
            self.evaluator,
            self.design_variables,
            objectives,
            use_bonsai=True,
        ).optimize(n_steps=1, n_init=2, seed=0)

        self.assertEqual(result["stop_reason"], "budget")
        self.assertGreaterEqual(len(result["pareto_front"]), 1)
        self.assertEqual(set(result["best_objectives"]), {"f_xy", "g_xy"})

    def test_post_processing_failures_do_not_fail_the_run(self):
        with patch(
            "gemseo.settings.post.OptHistoryView_Settings",
            side_effect=Exception("Plot failed"),
        ):
            result = self._optimizer().optimize(n_steps=1, n_init=1, seed=0)

        self.assertEqual(result["stop_reason"], "budget")

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_basic(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_create_scenario.return_value = mock_scenario

        optimizer = BayesianOptimizer(
            self.evaluator,
            self.design_variables,
            self.objectives,
            constraints=[ConstraintSpec(name="g_xy", bound=0.0)],
        )
        result = optimizer.explore(n_samples=2, n_processes=1)

        mock_scenario.execute.assert_called_once_with(
            algo_name="Sobol",
            n_samples=2,
            n_processes=1,
        )
        mock_scenario.post_process.assert_called()
        self.assertIn("history", result)

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_supports_greater_equal_constraints(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_create_scenario.return_value = mock_scenario

        optimizer = BayesianOptimizer(
            self.evaluator,
            self.design_variables,
            self.objectives,
            constraints=[ConstraintSpec(name="g_xy", op=">=", bound=1.5)],
        )
        optimizer.explore(n_samples=2, n_processes=1)

        mock_scenario.add_constraint.assert_called_once_with(
            "g_xy",
            constraint_type="ineq",
            value=1.5,
            positive=True,
        )

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_wraps_execution_failures(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_scenario.execute.side_effect = ValueError("DOE failed")
        mock_create_scenario.return_value = mock_scenario

        optimizer = BayesianOptimizer(
            self.evaluator, self.design_variables, self.objectives
        )

        with self.assertRaises(OptimizationExecutionError):
            optimizer.explore()

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_supports_remote_evaluator(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_create_scenario.return_value = mock_scenario

        optimizer = BayesianOptimizer(
            RemoteEvaluator("http://test"),
            [
                RangeVar(name="x", lower=0.0, upper=1.0),
                ChoiceVar(name="c_str", choices=["A", "B", "C"]),
                ChoiceVar(name="c_num", choices=[42, 7]),
            ],
            self.objectives,
        )
        result = optimizer.explore(n_samples=2, n_processes=1)

        self.assertIn("history", result)
        mock_create_scenario.assert_called_once()

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_ignores_post_process_failures(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_scenario.post_process.side_effect = Exception("Plot failed")
        mock_create_scenario.return_value = mock_scenario

        optimizer = BayesianOptimizer(
            self.evaluator, self.design_variables, self.objectives
        )
        result = optimizer.explore()
        self.assertIn("history", result)


class TestRemoteDiscipline(unittest.TestCase):
    def test_decodes_the_design_variables_for_the_evaluator(self):
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {"y": 42.0}
        design_variables = [
            RangeVar(name="x", lower=0.0, upper=4.0),
            RangeVar(name="n", lower=0, upper=9, value_type="int"),
            ChoiceVar(name="c", choices=["A", "B"]),
        ]

        discipline = RemoteDiscipline(mock_evaluator, design_variables, ["y"])
        discipline.execute(
            {"x": np.array([2.0]), "n": np.array([3.0]), "c": np.array([1])}
        )

        self.assertEqual(discipline.local_data["y"][0], 42.0)
        mock_evaluator.evaluate.assert_called_once_with(
            {"x": 2.0, "n": 3, "c": "B"}, ["y"]
        )
        self.assertEqual(discipline.default_input_data["c"][0], 0)
        self.assertEqual(discipline.default_input_data["x"][0], 0.0)


def _line_explorer(tools, *, coupled=False, algorithm="Ax_Bayesian"):
    """BayesianOptimizer over x in [0, 1] minimizing f, built from real GEMSEO."""
    from mdo_framework.core.topology import TopologicalAnalyzer
    from mdo_framework.core.translator import GraphProblemBuilder
    from mdo_framework.schema import RangeVar, StateVar, StudySchema, ToolSpec

    variables = [RangeVar(name="x", lower=0.0, upper=1.0), StateVar(name="f")]
    specs = [ToolSpec(name="T", inputs=["x", "g"] if coupled else ["x"], outputs=["f"])]
    if coupled:
        variables.append(StateVar(name="g"))
        specs.append(ToolSpec(name="U", inputs=["f"], outputs=["g"]))
    schema = StudySchema(variables=variables, tools=specs)
    analyzer = TopologicalAnalyzer(schema)
    resolved = analyzer.resolve_dependencies(["f"])
    builder = GraphProblemBuilder(schema)
    evaluator = LocalEvaluator(builder.build_problem(tools), builder.variable_specs)
    return BayesianOptimizer(
        evaluator,
        resolved.design_variables,
        [ObjectiveSpec(name="f")],
        algorithm=algorithm,
    )


def test_explore_raises_the_typed_error_when_no_sample_evaluates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    optimizer = _line_explorer(
        {"T": lambda x, g: g + 1.0 + x, "U": lambda f: f + 1.0}, coupled=True
    )

    with pytest.raises(MDANotConvergedError):
        optimizer.explore(n_samples=2)


def test_explore_keeps_the_samples_that_evaluate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fails_above_half(x):
        if x > 0.5:
            raise RuntimeError("mesh generation failed")
        return x

    optimizer = _line_explorer({"T": fails_above_half})

    history = optimizer.explore(n_samples=8)["history"]

    inputs = history.get_view(variable_names="x").to_numpy().ravel()
    assert 0 < len(inputs) < 8
    assert (inputs <= 0.5).all()


def test_explore_keeps_a_remote_transport_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    client = httpx.Client(base_url="http://exec", transport=transport)
    optimizer = BayesianOptimizer(
        RemoteEvaluator("http://exec", client=client),
        [RangeVar(name="x", lower=0.0, upper=1.0)],
        [ObjectiveSpec(name="f")],
    )

    with pytest.raises(RemoteEvaluationTransportError, match="HTTP 503"):
        optimizer.explore(n_samples=2)


def test_a_run_ended_before_it_started_reports_that_it_did_not_run(
    tmp_path, monkeypatch
):
    from gemseo.algos.stop_criteria import MaxTimeReached

    from mdo_framework.optimization import optimizer as optimizer_module
    from mdo_framework.optimization.random_search import RandomSearchLibrary

    class StoppedLibrary(RandomSearchLibrary):
        """GEMSEO swallows a termination raised before the run starts."""

        def _pre_run(self, problem):
            super()._pre_run(problem)
            raise MaxTimeReached

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(optimizer_module.ALGORITHMS, "BO_RandomSearch", StoppedLibrary)
    optimizer = _line_explorer({"T": lambda x: x}, algorithm="BO_RandomSearch")

    with pytest.raises(OptimizationExecutionError, match="did not run") as raised:
        optimizer.optimize(n_steps=1, n_init=1)

    assert raised.value.partial_result is None


def test_failure_recorder_never_caches_a_non_deterministic_tool():
    from mdo_framework.core.components import ToolComponent
    from mdo_framework.optimization.optimizer import _FailureRecorder

    calls = []

    def noisy(x):
        calls.append(x)
        return x + len(calls)

    tool = ToolComponent("T", noisy, ["x"], ["f"], deterministic=False)
    recorder = _FailureRecorder(tool)

    first = recorder.execute({"x": np.array([0.5])})["f"]
    second = recorder.execute({"x": np.array([0.5])})["f"]

    assert len(calls) == 2
    assert first != second
