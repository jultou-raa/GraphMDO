"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import unittest
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
from gemseo.core.discipline import Discipline

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


class OptimizerTestCase(unittest.TestCase):
    def setUp(self):
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
                self.local_data["f_xy"] = np.array([1.0])
                self.local_data["g_xy"] = np.array([0.0])

        self.mock_prob = MockDisc()
        self.evaluator = LocalEvaluator(self.mock_prob)
        self.parameters = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "c", "type": "range", "bounds": [0.0, 1.0]},
        ]
        self.objectives = [{"name": "f_xy", "minimize": True}]

    def _configure_ax_client(
        self,
        mock_client_cls,
        *,
        next_trials=None,
        best_parameters=None,
        best_metrics=None,
        pareto_front=None,
    ):
        client = mock_client_cls.return_value
        client.get_next_trials.return_value = next_trials or {
            0: {"x": 0.5, "y": 0.5, "c": 0.0}
        }
        client._to_json_snapshot.return_value = {}
        client.to_json_snapshot.return_value = "{}"
        client.get_best_parameterization.return_value = (
            best_parameters or {"x": 0.5, "y": 0.5, "c": 0.0},
            best_metrics or ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        client.get_pareto_frontier.return_value = (
            [] if pareto_front is None else pareto_front
        )
        return client


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

        invalid_cases = [
            [{"name": "x", "type": "range", "bounds": [0.0]}],
            [{"name": "x", "type": "categorical"}],
            [{"name": "x", "type": "choice", "values": []}],
        ]

        for parameters in invalid_cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(OptimizationConfigurationError):
                    _build_design_space(parameters)

        design_space = _build_design_space(
            [
                {
                    "name": "count",
                    "type": "range",
                    "bounds": [0, 5],
                    "value_type": "int",
                }
            ]
        )
        self.assertEqual(float(design_space.get_lower_bound("count")[0]), 0.0)
        self.assertEqual(float(design_space.get_upper_bound("count")[0]), 5.0)

    def test_extract_best_objectives_contracts(self):
        from mdo_framework.optimization.optimizer import _extract_best_objectives

        class BrokenOptimum:
            @property
            def objective(self):
                raise RuntimeError("no objective")

        objectives = _extract_best_objectives(
            BrokenOptimum(),
            ["f_xy", "g_xy"],
            fallback_metrics={"f_xy": 1.0, "g_xy": 2.0},
        )
        self.assertEqual(objectives, {"f_xy": 1.0, "g_xy": 2.0})

        class PartialOptimum:
            objective = np.array([1.0])

        with self.assertRaises(OptimizationExecutionError):
            _extract_best_objectives(
                PartialOptimum(),
                ["f_xy", "g_xy"],
                fallback_metrics={"f_xy": 1.0},
            )

    def test_optimizer_helper_error_wrapping(self):
        from mdo_framework.optimization.optimizer import (
            _decode_parameter_value,
            _get_optimization_history,
        )

        with self.assertRaises(OptimizationConfigurationError):
            _decode_parameter_value({"name": "c", "type": "choice", "values": []}, 0)

        with self.assertRaises(OptimizationExecutionError):
            _decode_parameter_value(
                {"name": "c", "type": "choice", "values": ["A"]},
                True,
            )

        self.assertEqual(_get_optimization_history(None, object()), [])
        self.assertEqual(
            _get_optimization_history(None, MagicMock(trial_history=[{"ok": True}])),
            [{"ok": True}],
        )


class TestBayesianOptimizer(OptimizerTestCase):
    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_basic(self, mock_client_cls):
        self._configure_ax_client(mock_client_cls)

        optimizer = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        result = optimizer.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertIn("best_objectives", result)
        self.assertEqual(result["best_parameters"]["x"], 0.5)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_with_constraints(self, mock_client_cls):
        self._configure_ax_client(mock_client_cls)

        optimizer = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            constraints=[{"name": "g_xy", "op": "<=", "bound": 0.0}],
        )
        result = optimizer.optimize(n_steps=2, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertEqual(len(result["history"]), 3)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_with_parameter_constraints(self, mock_client_cls):
        client = self._configure_ax_client(mock_client_cls)

        optimizer = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            parameter_constraints=["x <= y"],
        )
        result = optimizer.optimize(n_steps=1, n_init=1)

        self.assertEqual(result["best_parameters"]["x"], 0.5)
        client.configure_experiment.assert_called()
        call_kwargs = client.configure_experiment.call_args.kwargs
        self.assertEqual(call_kwargs["parameter_constraints"], ["x <= y"])

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_multi_objective_with_bonsai(self, mock_client_cls):
        pareto_front = [
            (
                {"x": 0.5, "y": 0.5, "c": "B"},
                ({"f_xy": (42.0, None), "g_xy": (10.0, None)}, None),
                0,
                "0_0",
            )
        ]
        self._configure_ax_client(
            mock_client_cls,
            next_trials={0: {"x": 0.5, "y": 0.5, "c": "B"}},
            best_parameters={"x": 0.5, "y": 0.5, "c": "B"},
            best_metrics=({"f_xy": (42.0, None), "g_xy": (10.0, None)}, None),
            pareto_front=pareto_front,
        )

        parameters = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "c", "type": "choice", "values": ["A", "B"]},
        ]
        objectives = [
            {"name": "f_xy", "minimize": True},
            {"name": "g_xy", "minimize": False, "threshold": 100.0},
        ]

        optimizer = BayesianOptimizer(
            self.evaluator,
            parameters,
            objectives,
            use_bonsai=True,
        )
        result = optimizer.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertEqual(result["best_parameters"]["x"], 0.5)
        self.assertEqual(result["best_objectives"]["g_xy"], 10.0)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_remote_evaluator_and_choices(self, mock_client_cls):
        self._configure_ax_client(
            mock_client_cls,
            next_trials={
                0: {"x": 0.5, "c_str": "B", "c_single": "A", "c_num_single": 42}
            },
            best_parameters={
                "x": 0.5,
                "c_str": "B",
                "c_single": "A",
                "c_num_single": 42,
            },
            best_metrics={"f_xy": 42.0},
        )

        mock_evaluator = RemoteEvaluator("http://test")
        mock_evaluator.evaluate = MagicMock(
            side_effect=lambda parameters, objectives: {
                "f_xy": 0.0 if parameters["c_str"] == "B" else 1.0
            }
        )

        parameters = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "c_str", "type": "choice", "values": ["A", "B", "C"]},
            {"name": "c_single", "type": "choice", "values": ["A"]},
            {"name": "c_num_single", "type": "choice", "values": [42]},
        ]

        optimizer = BayesianOptimizer(mock_evaluator, parameters, self.objectives)
        result = optimizer.optimize(n_steps=1, n_init=1)

        self.assertEqual(result["best_parameters"]["x"], 0.5)
        evaluated_parameters = mock_evaluator.evaluate.call_args.args[0]
        self.assertIn(evaluated_parameters["c_str"], ["A", "B", "C"])
        self.assertEqual(evaluated_parameters["c_single"], "A")
        self.assertEqual(evaluated_parameters["c_num_single"], 42)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_fidelity_parameter_emits_warning(self, mock_client_cls):
        self._configure_ax_client(mock_client_cls)

        optimizer = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            fidelity_parameter="x",
        )

        with self.assertWarns(UserWarning):
            result = optimizer.optimize(n_steps=1, n_init=1)

        self.assertIn("best_parameters", result)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_pareto_none(self, mock_client_cls):
        self._configure_ax_client(mock_client_cls)
        mock_client_cls.return_value.get_pareto_frontier.return_value = None

        optimizer = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}],
        )

        with self.assertRaises(OptimizationExecutionError):
            optimizer.optimize(n_steps=1, n_init=2)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_wraps_best_parameterization_failures(self, mock_client_cls):
        client = self._configure_ax_client(mock_client_cls)
        client.get_best_parameterization.side_effect = Exception("Optimization failed")

        optimizer = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)

        with self.assertRaises(OptimizationExecutionError):
            optimizer.optimize(n_steps=1, n_init=2)

    @patch("mdo_framework.optimization.ax_algo_lib.AxOptimizationLibrary")
    def test_optimize_wraps_algorithm_execution_failures(self, mock_algo_cls):
        mock_algo = MagicMock()
        mock_algo.execute.side_effect = Exception("Algo failed")
        mock_algo_cls.return_value = mock_algo

        optimizer = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)

        with self.assertRaises(OptimizationExecutionError):
            optimizer.optimize(n_steps=1, n_init=1)

    @patch("mdo_framework.optimization.ax_algo_lib.AxOptimizationLibrary.execute")
    def test_optimize_requires_valid_optimum(self, mock_execute):
        with patch(
            "mdo_framework.optimization.optimizer.create_scenario"
        ) as mock_create_scenario:
            mock_problem = MagicMock(optimum=None)
            mock_scenario = MagicMock()
            mock_scenario.formulation.optimization_problem = mock_problem
            mock_create_scenario.return_value = mock_scenario

            optimizer = BayesianOptimizer(
                self.evaluator, self.parameters, self.objectives
            )

            with self.assertRaises(OptimizationExecutionError):
                optimizer.optimize(n_steps=1, n_init=1)

        mock_execute.assert_called_once()

    def test_optimize_raises_when_optimum_objectives_are_unreadable(self):
        objective_sets = [
            self.objectives,
            [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}],
        ]

        class DummyOptimum:
            design = np.array([0.5, 0.5, 0.5])

            @property
            def objective(self):
                raise TypeError("Bang!")

        for objectives in objective_sets:
            with self.subTest(objectives=objectives):
                optimizer = BayesianOptimizer(
                    self.evaluator, self.parameters, objectives
                )
                with patch(
                    "gemseo.algos.optimization_problem.OptimizationProblem.optimum",
                    return_value=DummyOptimum(),
                ):
                    with patch(
                        "mdo_framework.optimization.ax_algo_lib.AxOptimizationLibrary.execute"
                    ):
                        with self.assertRaises(OptimizationExecutionError):
                            optimizer.optimize(n_steps=1, n_init=1)

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_basic(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_create_scenario.return_value = mock_scenario

        optimizer = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            use_bonsai=True,
            constraints=[{"name": "g_xy", "op": "<=", "bound": 0.0}],
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
            self.parameters,
            self.objectives,
            constraints=[{"name": "g_xy", "op": ">=", "bound": 1.5}],
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

        optimizer = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)

        with self.assertRaises(OptimizationExecutionError):
            optimizer.explore()

    def test_explore_rejects_invalid_constraint_operator(self):
        optimizer = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            constraints=[{"name": "g_xy", "op": "==", "bound": 0.0}],
        )

        with self.assertRaises(ValueError):
            optimizer.explore()

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_supports_remote_evaluator(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_create_scenario.return_value = mock_scenario

        optimizer = BayesianOptimizer(
            RemoteEvaluator("http://test"),
            [
                {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
                {"name": "c_str", "type": "choice", "values": ["A", "B", "C"]},
                {"name": "c_single", "type": "choice", "values": ["A"]},
                {"name": "c_num_single", "type": "choice", "values": [42]},
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

        optimizer = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        result = optimizer.explore()
        self.assertIn("history", result)


class TestRemoteDiscipline(unittest.TestCase):
    def test_executes_with_string_inputs(self):
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {"y": 42.0}

        discipline = RemoteDiscipline(mock_evaluator, ["x"], ["y"])
        discipline.execute({"x": np.array([2.0])})

        self.assertEqual(discipline.local_data["y"][0], 42.0)
