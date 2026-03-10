"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import unittest
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.stop_criteria import MaxIterReachedException
from gemseo.core.mdo_functions.mdo_function import MDOFunction

from mdo_framework.optimization.ax_algo_lib import (
    AxOptimizationLibrary,
    AxSettings,
    _get_choice_parameter_type,
    _get_range_parameter_type,
    _require_ax_settings,
    _validate_custom_ax_parameters,
    build_from_ax_parameters,
    build_from_design_space,
    build_optimization_config,
    build_outcome_constraints,
)


class TestAxOptimizationLibrary(unittest.TestCase):
    def _add_variable(
        self,
        design_space: DesignSpace,
        name: str,
        lower_bound: float,
        upper_bound: float,
        *,
        value: float | None = None,
        integer: bool = False,
    ) -> None:
        kwargs: dict[str, Any] = {
            "lower_bound": [lower_bound],
            "upper_bound": [upper_bound],
        }
        if value is not None:
            kwargs["value"] = [value]
        if integer:
            kwargs["type_"] = cast(Any, "integer")
        design_space.add_variable(name, **kwargs)

    def _make_problem(self, *, upper_bound: float = 1.0) -> OptimizationProblem:
        design_space = DesignSpace()
        self._add_variable(design_space, "x", 0.0, upper_bound)
        problem = OptimizationProblem(design_space)

        def objective(x):
            return np.array([x[0] ** 2])

        problem.objective = MDOFunction(objective, "obj", expr="x**2")
        return problem

    def test_build_from_ax_parameters_validation_and_inference(self):
        invalid_cases = [
            [{"name": "x", "type": "choice"}],
            [{"name": "x", "type": "range"}],
            [{"name": "x", "type": "unrecognized", "bounds": [0.0, 1.0]}],
            [{"name": "x", "type": "range", "bounds": [0.0]}],
            [{"name": "x", "type": "choice", "values": []}],
        ]

        for parameters in invalid_cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    build_from_ax_parameters(parameters)

        inference_cases = [
            (
                [{"name": "x", "type": "range", "bounds": [0.0, 1.0]}],
                "float",
            ),
            ([{"name": "x", "type": "choice", "values": [1.0]}], "float"),
            ([{"name": "flag", "type": "choice", "values": [True, False]}], "bool"),
        ]

        for parameters, expected_type in inference_cases:
            with self.subTest(parameters=parameters):
                ax_parameters = build_from_ax_parameters(parameters)
                self.assertEqual(ax_parameters[0].parameter_type, expected_type)

        with self.assertRaises(ValueError):
            _get_range_parameter_type(
                {"name": "x", "type": "range", "value_type": "decimal"}
            )

        with self.assertRaises(ValueError):
            _get_choice_parameter_type(
                {"name": "choice", "type": "choice", "value_type": "decimal"},
                ["A", "B"],
            )

    def test_build_from_design_space_and_custom_layout_rules(self):
        design_space = DesignSpace()
        self._add_variable(design_space, "x", 0.0, 10.0)
        self._add_variable(design_space, "count", 0.0, 5.0, integer=True)

        physical_parameters = build_from_design_space(design_space, normalize=False)
        normalized_parameters = build_from_design_space(design_space, normalize=True)

        self.assertEqual(
            [parameter.name for parameter in physical_parameters],
            ["x", "count"],
        )
        self.assertEqual(
            [parameter.parameter_type for parameter in normalized_parameters],
            ["float", "float"],
        )

        with self.assertRaises(ValueError):
            _validate_custom_ax_parameters(
                [
                    {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
                    {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
                ],
                design_space,
                normalize=False,
            )

    def test_build_outcome_constraints_and_optimization_config(self):
        problem = self._make_problem()

        self.assertEqual(build_outcome_constraints([]), [])

        unsupported_constraint = MagicMock()
        unsupported_constraint.name = "h"
        unsupported_constraint.f_type = "eq"
        with self.assertRaises(ValueError):
            build_outcome_constraints([unsupported_constraint])

        config = build_optimization_config(
            [{"name": "a", "minimize": True}, {"name": "b", "minimize": False}],
            problem,
            [],
        )
        self.assertIsNotNone(config)

    def test_seed_database_records_existing_points(self):
        problem = self._make_problem()
        problem.database.store(np.array([0.1]), {"obj": np.array([0.1])})
        problem.database.store(np.array([0.2]), {"obj": np.array([0.2])})

        client = MagicMock()
        client.attach_baseline.return_value = 0
        client.attach_trial.return_value = 1

        algo = AxOptimizationLibrary()
        algo._seed_database(
            client,
            problem,
            problem.design_space,
            normalize=False,
            ax_parameters=[{"name": "x", "type": "range", "bounds": [0.0, 1.0]}],
        )

        client.attach_baseline.assert_called_once_with(parameters={"x": 0.1})
        client.attach_trial.assert_called_once_with(parameters={"x": 0.2})
        self.assertEqual(client.complete_trial.call_count, 2)
        self.assertEqual(len(algo.trial_history), 2)

    def test_seed_result_and_metric_normalization_contracts(self):
        normalized_metrics = AxOptimizationLibrary._normalize_best_metrics("invalid")
        self.assertEqual(normalized_metrics, {})

        seed_results = AxOptimizationLibrary._extract_seed_results(
            {
                "obj": np.array([1.0]),
                "g": np.array([-1.0, 2.0]),
            },
            {"obj", "g"},
            {"g"},
        )
        self.assertEqual(seed_results, {"obj": 1.0, "g": 2.0})

    def test_record_last_point_contracts(self):
        design_space = DesignSpace()
        self._add_variable(design_space, "x", 0.0, 1.0)
        self._add_variable(design_space, "c_str", 0.0, 2.0, integer=True)
        self._add_variable(design_space, "count", 0.0, 5.0, integer=True)

        class DummyLastPoint:
            design = np.array([0.5, 1.0, 2.0])
            objective = np.array([10.0])
            constraints = {"g": -1.0}

        problem = MagicMock()
        problem.design_space = design_space
        problem.history.last_point = DummyLastPoint()
        problem.objective.name = "obj"

        algo = AxOptimizationLibrary()
        algo._record_last_point(
            problem,
            [
                {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
                {"name": "c_str", "type": "choice", "values": ["A", "B", "C"]},
                {
                    "name": "count",
                    "type": "range",
                    "bounds": [0, 5],
                    "value_type": "int",
                },
            ],
        )

        self.assertEqual(
            algo.trial_history,
            [
                {
                    "parameters": {"x": 0.5, "c_str": "B", "count": 2},
                    "objectives": {"obj": 10.0, "g": -1.0},
                }
            ],
        )

        problem_with_no_history = self._make_problem()
        problem_with_no_history.history = MagicMock()
        type(problem_with_no_history.history).last_point = property(
            lambda self: (_ for _ in ()).throw(ValueError("no history"))
        )

        algo = AxOptimizationLibrary()
        algo._record_last_point(problem_with_no_history)
        self.assertEqual(algo.trial_history, [])

    def test_execute_trial_contracts(self):
        problem = self._make_problem()

        successful_cases = [
            (
                "scalar_int_metric",
                {"obj": 5},
                {"obj": 5.0},
            ),
            (
                "multi_dimensional_metric",
                {"obj": np.array([1.0, 3.0])},
                {"obj": 3.0},
            ),
        ]

        for label, output, expected_raw_data in successful_cases:
            with self.subTest(case=label):
                algo = AxOptimizationLibrary()
                client = MagicMock()
                problem.evaluate_functions = MagicMock(return_value=(output, None))

                budget_exhausted = algo._execute_trial(
                    client,
                    problem,
                    0,
                    {"x": 0.5},
                    {"obj"},
                )

                self.assertFalse(budget_exhausted)
                client.complete_trial.assert_called_once_with(
                    trial_index=0,
                    raw_data=expected_raw_data,
                )
                self.assertEqual(
                    algo.trial_history[-1],
                    {"parameters": {"x": 0.5}, "objectives": expected_raw_data},
                )

        abandonment_cases = [
            (
                "recoverable_error",
                ValueError("boom"),
                False,
            ),
            (
                "budget_exhausted",
                MaxIterReachedException(),
                True,
            ),
            (
                "missing_metrics",
                ({"other": 1.0}, None),
                False,
            ),
        ]

        for label, outcome, expected_budget_exhausted in abandonment_cases:
            with self.subTest(case=label):
                algo = AxOptimizationLibrary()
                client = MagicMock()
                if isinstance(outcome, Exception):
                    problem.evaluate_functions = MagicMock(side_effect=outcome)
                else:
                    problem.evaluate_functions = MagicMock(return_value=outcome)

                budget_exhausted = algo._execute_trial(
                    client,
                    problem,
                    1,
                    {"x": 0.25},
                    {"obj"},
                )

                self.assertEqual(budget_exhausted, expected_budget_exhausted)
                client.mark_trial_abandoned.assert_called_once_with(trial_index=1)

    def test_extract_best_solution_contracts(self):
        problem = self._make_problem()
        client = MagicMock()
        client.get_best_parameterization.return_value = (
            {"x": 0.5},
            ({"obj": 0.25}, None),
            0,
            "arm_0",
        )

        problem.evaluate_functions = MagicMock(side_effect=MaxIterReachedException())

        algo = AxOptimizationLibrary()
        algo._extract_best_solution(client, problem, False)
        self.assertEqual(algo.best_objectives, {"obj": 0.25})

        problem.evaluate_functions = MagicMock(side_effect=ValueError("boom"))
        algo._extract_best_solution(client, problem, False)
        self.assertEqual(algo.best_objectives, {"obj": 0.25})

        client.get_pareto_frontier.return_value = []
        with self.assertRaisesRegex(ValueError, "Pareto frontier is empty"):
            algo._extract_best_solution(client, problem, True)

    def test_normalized_design_vectors_are_unnormalized(self):
        problem = self._make_problem(upper_bound=10.0)
        problem.evaluate_functions = MagicMock(
            return_value=({"obj": np.array([25.0])}, None)
        )

        algo = AxOptimizationLibrary()
        client = MagicMock()

        algo._execute_trial(client, problem, 0, {"x": 0.5}, {"obj"}, normalize=True)
        evaluated_x = problem.evaluate_functions.call_args.args[0]
        self.assertEqual(float(evaluated_x[0]), 5.0)

        problem.evaluate_functions.reset_mock(return_value=True)
        client.get_best_parameterization.return_value = (
            {"x": 0.5},
            ({"obj": 25.0}, None),
            0,
            "arm_0",
        )

        algo._extract_best_solution(client, problem, False, normalize=True)
        optimum_x = problem.evaluate_functions.call_args.args[0]
        self.assertEqual(float(optimum_x[0]), 5.0)
        self.assertEqual(float(problem.design_space.get_current_value()[0]), 5.0)

    def test_configure_client_rejects_incompatible_custom_ax_parameters(self):
        problem = self._make_problem()
        algo = AxOptimizationLibrary(client_factory=lambda: MagicMock())

        incompatible_settings = [
            AxSettings(
                max_iter=1,
                n_init=1,
                batch_size=1,
                use_bonsai=False,
                ax_parameters=[
                    {"name": "custom_x", "type": "range", "bounds": [0.0, 1.0]}
                ],
                ax_objectives=[{"name": "obj", "minimize": True}],
                normalize_design_space=False,
            ),
            AxSettings(
                max_iter=1,
                n_init=1,
                batch_size=1,
                use_bonsai=False,
                ax_parameters=[{"name": "x", "type": "range", "bounds": [0.0, 1.0]}],
                ax_objectives=[{"name": "obj", "minimize": True}],
                normalize_design_space=True,
            ),
        ]

        for settings in incompatible_settings:
            with self.subTest(settings=settings):
                algo._settings = settings
                with self.assertRaises(ValueError):
                    algo._configure_client(problem)

    def test_run_guards_and_loop_fallback(self):
        with self.assertRaises(TypeError):
            _require_ax_settings(cast(Any, object()))

        algo = AxOptimizationLibrary(client_factory=lambda: MagicMock())
        algo._settings = AxSettings(
            max_iter=1,
            n_init=1,
            batch_size=1,
            use_bonsai=False,
            ax_parameters=[{"name": "x", "type": "range", "bounds": [0.0, 1.0]}],
            ax_objectives=[{"name": "obj", "minimize": True}],
        )
        with self.assertRaises(TypeError):
            algo._run(cast(Any, MagicMock()))

        problem = self._make_problem()
        problem.database.store(np.array([0.2]), {"obj": np.array([0.04])})
        problem.evaluate_functions = MagicMock(
            return_value=({"obj": np.array([0.04])}, None)
        )

        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {}
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.2},
            ({"obj": (0.04, None)}, None),
            0,
            "0_0",
        )

        algo = AxOptimizationLibrary(client_factory=lambda: mock_client)
        algo.execute(
            problem,
            max_iter=1,
            n_init=1,
            ax_parameters=[{"name": "x", "type": "range", "bounds": [0.0, 1.0]}],
            ax_objectives=[{"name": "obj", "minimize": True}],
        )

        self.assertGreaterEqual(len(algo.trial_history), 2)
        self.assertEqual(algo.trial_history[-1]["parameters"], {"x": 0.2})

        budget_problem = self._make_problem()
        budget_problem.evaluate_functions = MagicMock(
            side_effect=MaxIterReachedException()
        )
        budget_client = MagicMock()
        budget_client.get_next_trials.return_value = {0: {"x": 0.5}}
        budget_client.get_best_parameterization.return_value = (
            {"x": 0.5},
            ({"obj": 0.25}, None),
            0,
            "arm_0",
        )

        algo = AxOptimizationLibrary(client_factory=lambda: budget_client)
        algo._settings = AxSettings(
            max_iter=1,
            n_init=1,
            batch_size=1,
            use_bonsai=False,
            ax_parameters=[{"name": "x", "type": "range", "bounds": [0.0, 1.0]}],
            ax_objectives=[{"name": "obj", "minimize": True}],
        )
        message, status = algo._run(budget_problem)
        self.assertEqual(
            message, "Optimization stopped early: evaluation budget exhausted."
        )
        self.assertEqual(status, 0)
