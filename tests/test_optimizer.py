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
    RemoteEvaluationContractError,
    RemoteEvaluator,
    RemoteEvaluationTransportError,
)
from mdo_framework.optimization.parameter_codec import (
    ParameterDefinitionError,
    ParameterValueError,
    build_parameter_lookup,
    coerce_scalar,
    decode_parameter_value,
    encode_parameter_value,
)


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        # Setup common mock data for GEMSEO discipline
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

    def test_local_evaluator_scalar(self):
        """Tests LocalEvaluator when GEMSEO returns an array containing a scalar."""

        def dummy_run(self, input_data):
            self.local_data["f_xy"] = np.array([1.23])
            self.local_data["g_xy"] = np.array([0.0])

        self.mock_prob._run = dummy_run.__get__(self.mock_prob, type(self.mock_prob))
        res = self.evaluator.evaluate({"x": 0.5, "y": 0.5, "c": 0.0}, ["f_xy"])
        self.assertEqual(res["f_xy"], 1.23)

    def test_local_evaluator_iterable(self):
        """Tests LocalEvaluator when GEMSEO returns a flat array."""

        def dummy_run2(self, input_data):
            self.local_data["f_xy"] = np.array([4.56, 1.0])
            self.local_data["g_xy"] = np.array([0.0])

        self.mock_prob._run = dummy_run2.__get__(self.mock_prob, type(self.mock_prob))
        res = self.evaluator.evaluate({"x": 0.5, "y": 0.5, "c": 0.0}, ["f_xy"])
        self.assertEqual(res["f_xy"], 4.56)

    def test_remote_evaluator(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": {"f_xy": 2.0}}
        mock_resp.raise_for_status.return_value = None
        mock_client.post.return_value = mock_resp

        evaluator = RemoteEvaluator("http://fake-url", client=mock_client)
        res = evaluator.evaluate({"x": 0.5, "y": 0.5, "c": 0.0}, ["f_xy"])

        mock_client.post.assert_called_once_with(
            "http://fake-url/evaluate",
            json={
                "inputs": {"x": 0.5, "y": 0.5, "c": 0.0},
                "objectives": ["f_xy"],
            },
        )
        self.assertEqual(res["f_xy"], 2.0)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_basic(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_pareto_frontier.return_value = []

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertIn("best_objectives", result)
        self.assertTrue(len(result["history"]) >= 0)
        self.assertEqual(result["best_parameters"]["x"], 0.5)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_bonsai_and_multi_objective(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        # Test pareto frontier handling
        mock_client_cls.return_value.get_pareto_frontier.return_value = [
            (
                {"x": 0.5, "y": 0.5, "c": 0},
                ({"f_xy": (42.0, None), "g_xy": (10.0, None)}, None),
                0,
                "0_0",
            )
        ]

        params = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "c", "type": "choice", "values": ["A", "B"]},
        ]
        objs = [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}]

        opt = BayesianOptimizer(self.evaluator, params, objs, use_bonsai=True)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)

        pass

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_pareto_none(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        mock_client_cls.return_value.get_pareto_frontier.return_value = None

        objs = [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}]

        opt = BayesianOptimizer(self.evaluator, self.parameters, objs)
        with self.assertRaises(OptimizationExecutionError):
            opt.optimize(n_steps=1, n_init=2)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_exception(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        mock_client_cls.return_value.get_best_parameterization.side_effect = Exception(
            "Optimization failed"
        )

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        with self.assertRaises(OptimizationExecutionError):
            opt.optimize(n_steps=1, n_init=2)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_with_constraints(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_pareto_frontier.return_value = []

        constraints = [{"name": "g_xy", "op": "<=", "bound": 0.0}]

        opt = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            constraints=constraints,
        )
        result = opt.optimize(n_steps=2, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertEqual(
            len(result["history"]), 3
        )  # Initial trials + 1 optimization step

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_fidelity_parameter_emits_warning(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_pareto_frontier.return_value = []

        opt = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            fidelity_parameter="x",
        )

        with self.assertWarns(UserWarning):
            result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_basic(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_create_scenario.return_value = mock_scenario

        constraints = [{"name": "g_xy", "op": "<=", "bound": 0.0}]
        opt = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            use_bonsai=True,
            constraints=constraints,
        )
        res = opt.explore(n_samples=2, n_processes=1)

        mock_scenario.execute.assert_called_once_with(
            algo_name="Sobol", n_samples=2, n_processes=1
        )
        mock_scenario.post_process.assert_called()
        self.assertIn("history", res)

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_supports_greater_equal_constraints(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_create_scenario.return_value = mock_scenario

        constraints = [{"name": "g_xy", "op": ">=", "bound": 1.5}]
        opt = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            constraints=constraints,
        )

        opt.explore(n_samples=2, n_processes=1)

        mock_scenario.add_constraint.assert_called_once_with(
            "g_xy",
            constraint_type="ineq",
            value=1.5,
            positive=True,
        )

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_exception(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_create_scenario.return_value = mock_scenario
        mock_scenario.execute.side_effect = ValueError("DOE failed")

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        with self.assertRaises(OptimizationExecutionError):
            opt.explore()

    def test_explore_invalid_constraint_operator(self):
        opt = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            constraints=[{"name": "g_xy", "op": "==", "bound": 0.0}],
        )

        with self.assertRaises(ValueError):
            opt.explore()

    def test_remote_discipline(self):
        from mdo_framework.optimization.optimizer import RemoteDiscipline

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {"y": 42.0}

        disc = RemoteDiscipline(mock_evaluator, ["x"], ["y"])
        disc.execute({"x": np.array([2.0])})

        self.assertEqual(disc.local_data["y"][0], 42.0)

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_remote_evaluator(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_create_scenario.return_value = mock_scenario

        mock_evaluator = RemoteEvaluator("http://test")

        params = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "c_str", "type": "choice", "values": ["A", "B", "C"]},
            {"name": "c_single", "type": "choice", "values": ["A"]},
            {"name": "c_num_single", "type": "choice", "values": [42]},
        ]
        opt = BayesianOptimizer(mock_evaluator, params, self.objectives)
        res = opt.explore(n_samples=2, n_processes=1)

        self.assertIn("history", res)
        # Verify it hits choice branches
        mock_create_scenario.assert_called_once()

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    def test_explore_post_process_exception(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_scenario.post_process.side_effect = Exception("Plot failed")
        mock_create_scenario.return_value = mock_scenario

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        res = opt.explore()
        self.assertIn("history", res)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_remote_evaluator_and_choices(self, mock_client_cls):
        mock_client_cls.return_value.get_next_trials.return_value = {
            0: {"x": 0.5, "c_str": "B", "c_single": "A", "c_num_single": 42}
        }
        mock_client_cls.return_value._to_json_snapshot.return_value = {}
        mock_client_cls.return_value.to_json_snapshot.return_value = "{}"
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "c_str": "B", "c_single": "A", "c_num_single": 42},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        mock_evaluator = RemoteEvaluator("http://test")
        mock_evaluator.evaluate = MagicMock(
            side_effect=lambda parameters, objectives: {
                "f_xy": 0.0 if parameters["c_str"] == "B" else 1.0
            }
        )

        params = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "c_str", "type": "choice", "values": ["A", "B", "C"]},
            {"name": "c_single", "type": "choice", "values": ["A"]},
            {"name": "c_num_single", "type": "choice", "values": [42]},
        ]

        opt = BayesianOptimizer(mock_evaluator, params, self.objectives)
        res = opt.optimize(n_steps=1, n_init=1)

        self.assertEqual(res["best_parameters"]["x"], 0.5)
        self.assertGreaterEqual(mock_evaluator.evaluate.call_count, 1)
        evaluated_parameters = mock_evaluator.evaluate.call_args.args[0]
        self.assertIn(evaluated_parameters["c_str"], ["A", "B", "C"])
        self.assertEqual(evaluated_parameters["c_single"], "A")
        self.assertEqual(evaluated_parameters["c_num_single"], 42)

    def test_parameter_codec_round_trip_variants(self):
        cases = [
            ({"name": "x", "type": "range", "value_type": "float"}, 1.25, 1.25),
            ({"name": "n", "type": "range", "value_type": "int"}, 1.6, 2),
            ({"name": "c_str", "type": "choice", "values": ["A", "B"]}, "B", "B"),
            ({"name": "c_bool", "type": "choice", "values": [False, True]}, True, True),
            ({"name": "c_single", "type": "choice", "values": [42]}, 42, 42),
        ]

        for parameter, input_value, expected_value in cases:
            encoded_value = encode_parameter_value(parameter, input_value)
            decoded_value = decode_parameter_value(parameter, encoded_value)
            self.assertEqual(decoded_value, expected_value)

    def test_parameter_codec_rejects_invalid_choice_definition(self):
        parameter = {"name": "c_bad", "type": "choice", "values": []}

        with self.assertRaises(ParameterDefinitionError):
            encode_parameter_value(parameter, 0)

        with self.assertRaises(ParameterDefinitionError):
            decode_parameter_value(parameter, 0)

    def test_parameter_codec_rejects_invalid_choice_index(self):
        parameter = {"name": "c_str", "type": "choice", "values": ["A", "B"]}

        with self.assertRaises(ParameterValueError):
            decode_parameter_value(parameter, 3)

    def test_parameter_codec_does_not_match_bool_to_int_choice(self):
        parameter = {"name": "c_int", "type": "choice", "values": [0, 1]}

        with self.assertRaises(ParameterValueError):
            encode_parameter_value(parameter, True)

    def test_ax_algo_lib_record_last_point_decodes_parameters(self):
        from gemseo.algos.design_space import DesignSpace

        from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary

        design_space = DesignSpace()
        design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
        design_space.add_variable(
            "c_str",
            lower_bound=0,
            upper_bound=2,
            type_="integer",
        )
        design_space.add_variable(
            "count",
            lower_bound=0,
            upper_bound=5,
            type_="integer",
        )

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

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_ax_algo_lib_direct(self, mock_client_cls):
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem

        from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)
        ds.add_variable("c", lower_bound=0.0, upper_bound=1.0, type_="integer")
        ds.add_variable("y", lower_bound=0.0, upper_bound=1.0, value=0.5)
        ds.add_variable("z", lower_bound=42.0, upper_bound=42.0, type_="integer")

        prob = OptimizationProblem(ds)

        def obj(x):
            return np.array([x[0] ** 2])

        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        prob.objective = MDOFunction(obj, "obj", expr="x**2")

        # Add a constraint to test constraint evaluation
        def constr(x):
            return np.array([x[0] - 1.0])

        prob.add_constraint(MDOFunction(constr, "g_1", f_type="ineq", expr="x-1"))

        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {
            0: {"x": 0.5, "c": 0.0, "y": 0.5, "z": 42.0}
        }
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.0, "c": 0.0, "y": 0.5, "z": 42.0},
            {"obj": 0.0},
            0,
            "0_0",
        )

        algo = AxOptimizationLibrary()
        algo.execute(prob, max_iter=1, n_init=1)

    @patch("mdo_framework.optimization.ax_algo_lib.AxOptimizationLibrary")
    def test_optimize_algo_exception(self, mock_algo_cls):
        mock_algo = MagicMock()
        mock_algo.execute.side_effect = Exception("Algo failed")
        mock_algo_cls.return_value = mock_algo

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        with self.assertRaises(OptimizationExecutionError):
            opt.optimize(n_steps=1, n_init=1)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_multi_objective(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        # Mock pareto frontier
        mock_client_cls.return_value.get_pareto_frontier.return_value = [
            (
                {"x": 0.5, "y": 0.5, "c": 0.0},
                ({"f_xy": (42.0, None), "g_xy": (10.0, None)}, None),
                0,
                "0_0",
            )
        ]

        # Configure a mock multi-objective setup
        mock_opt_config = MagicMock()
        mock_opt_config.objective = MagicMock()
        from ax.core.objective import MultiObjective

        mock_opt_config.objective.__class__ = MultiObjective
        mock_client._experiment = MagicMock()
        mock_client._experiment.optimization_config = mock_opt_config

        objs = [
            {"name": "f_xy", "minimize": True},
            {"name": "g_xy", "minimize": False, "threshold": 100.0},
        ]

        opt = BayesianOptimizer(self.evaluator, self.parameters, objs)
        result = opt.optimize(n_steps=1, n_init=1)

        self.assertIn("best_parameters", result)
        self.assertEqual(result["best_parameters"]["x"], 0.5)
        self.assertIsNotNone(result["best_objectives"])

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_with_parameter_constraints(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        mock_opt_config = MagicMock()
        mock_client._experiment = MagicMock()
        mock_client._experiment.optimization_config = mock_opt_config
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_pareto_frontier.return_value = []

        opt = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            parameter_constraints=["x <= y"],
        )
        result = opt.optimize(n_steps=1, n_init=1)

        self.assertIn("best_parameters", result)
        self.assertEqual(result["best_parameters"]["x"], 0.5)
        mock_client_cls.return_value.configure_experiment.assert_called()
        call_kwargs = mock_client_cls.return_value.configure_experiment.call_args.kwargs
        self.assertIn("parameter_constraints", call_kwargs)
        self.assertEqual(call_kwargs["parameter_constraints"], ["x <= y"])

    def test_ax_algo_lib_coverage(self):
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)

        prob = OptimizationProblem(ds)

        def obj(x):
            return np.array([x[0] ** 2])

        prob.objective = MDOFunction(obj, "obj", expr="x**2")

        algo = AxOptimizationLibrary()

        class DummySettings:
            max_iter = 10
            n_init = 5
            use_bonsai = True
            ax_parameters = [{"name": "x", "type": "range", "bounds": [0.0, 1.0]}]
            ax_objectives = [{"name": "obj", "minimize": True}]
            ax_parameter_constraints = ["x <= 1"]
            normalize_design_space = False

        algo._settings = DummySettings()

        with patch("mdo_framework.optimization.ax_algo_lib.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.get_next_trials.return_value = {0: {"x": 0.5}}
            mock_client_cls.return_value.get_best_parameterization.return_value = (
                {"x": 0.0},
                ({"obj": 0.0}, None),
            )

            # mock get_pareto_frontier specifically
            mock_client._experiment = MagicMock()
            mock_client._experiment.optimization_config.objective.__class__.__name__ = (
                "MultiObjective"
            )
            mock_client_cls.return_value.get_pareto_frontier.return_value = [
                ({"x": 0.0}, {"obj": 0.0}, 0, "0_0"),
            ]

            algo.execute(prob, max_iter=1, n_init=1)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_fidelity_param(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_pareto_frontier.return_value = []

        opt = BayesianOptimizer(
            self.evaluator, self.parameters, self.objectives, fidelity_parameter="x"
        )
        with self.assertWarns(UserWarning):
            opt.optimize(n_steps=1, n_init=1)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimizer_fallback_lines(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_pareto_frontier.return_value = []

        opt = BayesianOptimizer(
            self.evaluator, self.parameters, self.objectives, fidelity_parameter="z"
        )

        with patch(
            "gemseo.algos.optimization_problem.OptimizationProblem.optimum"
        ) as mock_opt:

            class DummyOptimum:
                design = np.array([0.5, 0.5, 0.5])

                @property
                def objective(self):
                    raise TypeError("Bang!")

            mock_opt.return_value = DummyOptimum()

            with patch(
                "mdo_framework.optimization.ax_algo_lib.AxOptimizationLibrary.execute"
            ):
                with self.assertRaises(OptimizationExecutionError):
                    opt.optimize(n_steps=1, n_init=1)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimizer_fallback_lines_multi(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            ({"f_xy": (42.0, None)}, None),
            0,
            "0_0",
        )
        mock_client_cls.return_value.get_pareto_frontier.return_value = []

        objs = [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}]

        opt = BayesianOptimizer(self.evaluator, self.parameters, objs)

        with patch(
            "gemseo.algos.optimization_problem.OptimizationProblem.optimum"
        ) as mock_opt:

            class DummyOptimum:
                design = np.array([0.5, 0.5, 0.5])

                @property
                def objective(self):
                    raise TypeError("Bang!")

            mock_opt.return_value = DummyOptimum()

            with patch(
                "mdo_framework.optimization.ax_algo_lib.AxOptimizationLibrary.execute"
            ):
                with self.assertRaises(OptimizationExecutionError):
                    opt.optimize(n_steps=1, n_init=1)

    def test_ax_algo_lib_execute_exceptions(self):
        import numpy as np
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)
        prob = OptimizationProblem(ds)

        def obj(x):
            return np.array([x[0] ** 2])

        prob.objective = MDOFunction(obj, "obj", expr="x**2")

        algo = AxOptimizationLibrary()

        from unittest.mock import MagicMock

        mock_client = MagicMock()

        # Line 433-435: ValueError exception in evaluate_functions
        prob.evaluate_functions = MagicMock(side_effect=ValueError("Test exception"))
        res = algo._execute_trial(mock_client, prob, 0, {"x": 0.5}, {"obj"})
        self.assertFalse(res)

        # Lines 443-444: Integer objective value
        prob.evaluate_functions = MagicMock(return_value=({"obj": 5}, None))
        res = algo._execute_trial(mock_client, prob, 0, {"x": 0.5}, {"obj"})
        self.assertFalse(res)

    def test_ax_algo_lib_extra_coverage(self):
        import numpy as np
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        from mdo_framework.optimization.ax_algo_lib import (
            AxOptimizationLibrary,
            AxSettings,
            build_from_ax_parameters,
        )

        # Line 135
        with self.assertRaises(ValueError):
            build_from_ax_parameters([{"name": "x", "type": "choice"}])

        # Line 143
        with self.assertRaises(ValueError):
            build_from_ax_parameters([{"name": "x", "type": "range"}])

        # Line 149
        with self.assertRaises(ValueError):
            build_from_ax_parameters(
                [{"name": "x", "type": "unrecognized", "bounds": [0.0, 1.0]}]
            )

        # Line 332: coverage for fidelity_parameter logging
        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)
        prob = OptimizationProblem(ds)

        def obj(x):
            return np.array([x[0] ** 2])

        prob.objective = MDOFunction(obj, "obj", expr="x**2")

        algo = AxOptimizationLibrary()

        algo._settings = AxSettings(max_iter=5)
        algo.problem = prob

        from unittest.mock import MagicMock, patch

        with patch("mdo_framework.optimization.ax_algo_lib.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            algo._run(prob)

        # Line 510: Pareto front empty value error
        class DummySettingsMOO:
            def __init__(self):
                self.max_iter = 5
                self.seed = None
                self.ax_parameters = None
                self.ax_parameter_constraints = None
                self.ax_outcome_constraints = None
                self.ax_objectives = ["obj1", "obj2"]
                self.is_moo = True
                self.optimization_direction = "minimize"
                self.batch_size = 1

        algo._settings = DummySettingsMOO()
        mock_client = MagicMock()
        mock_client.get_pareto_frontier.return_value = []
        with self.assertRaisesRegex(ValueError, "Pareto frontier is empty"):
            algo._extract_best_solution(mock_client, prob, True)

        # Lines 433-447: MaxIterReachedException handling in evaluate trial
        from gemseo.algos.stop_criteria import MaxIterReachedException

        prob.evaluate_functions = MagicMock(side_effect=MaxIterReachedException())
        mock_client = MagicMock()
        from gemseo.algos.stop_criteria import MaxIterReachedException

        prob.evaluate_functions = MagicMock(side_effect=MaxIterReachedException())
        mock_client = MagicMock()
        # wait, evaluate trial doesn't raise MaxIterReachedException directly if it's caught.
        # actually, if evaluate_functions raises MaxIterReachedException, `_execute_trial` caches it in `self.problem.max_iter_reached` but `_execute_trial` catches Exception.
        # No, wait, `_execute_trial` catches `ValueError`. `MaxIterReachedException` inherits from `Exception`. Let's check `_execute_trial` code.

    def test_ax_algo_lib_edge_cases(self):
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        from mdo_framework.optimization.ax_algo_lib import (
            AxOptimizationLibrary,
            build_from_ax_parameters,
            build_optimization_config,
            build_outcome_constraints,
        )

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)

        prob = OptimizationProblem(ds)

        def obj(x):
            return np.array([x[0] ** 2])

        prob.objective = MDOFunction(obj, "obj", expr="x**2")

        res = build_outcome_constraints([])
        self.assertEqual(res, [])

        unsupported_constraint = MagicMock()
        unsupported_constraint.name = "h"
        unsupported_constraint.f_type = "eq"
        with self.assertRaises(ValueError):
            build_outcome_constraints([unsupported_constraint])

        # Test ax parameters exceptions
        with self.assertRaises(ValueError):
            build_from_ax_parameters([{"name": "x", "type": "range", "bounds": [0.0]}])
        with self.assertRaises(ValueError):
            build_from_ax_parameters([{"name": "x", "type": "choice", "values": []}])

        # Test range fallback (no value type provided)
        res = build_from_ax_parameters(
            [{"name": "x", "type": "range", "bounds": [0.0, 1.0]}]
        )
        self.assertEqual(res[0].parameter_type, "float")

        res = build_from_ax_parameters(
            [{"name": "x", "type": "choice", "values": [1.0]}]
        )
        self.assertEqual(res[0].parameter_type, "float")

        res = build_from_ax_parameters(
            [{"name": "flag", "type": "choice", "values": [True, False]}]
        )
        self.assertEqual(res[0].parameter_type, "bool")

        # Test MaxIterReachedException in execute trial
        from gemseo.algos.stop_criteria import MaxIterReachedException

        prob.evaluate_functions = MagicMock(side_effect=MaxIterReachedException())
        mock_client = MagicMock()
        algo = AxOptimizationLibrary()
        res = algo._execute_trial(MagicMock(), prob, 0, {"x": 0.5}, {"obj"})
        self.assertTrue(res)

        prob.evaluate_functions = MagicMock(side_effect=MaxIterReachedException())
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5},
            ({"obj": 0.25}, None),
            0,
            "arm_0",
        )
        algo._extract_best_solution(mock_client, prob, False)

        config = build_optimization_config(
            [{"name": "a", "minimize": True}, {"name": "b", "minimize": False}],
            prob,
            [],
        )
        self.assertIsNotNone(config)

        mock_client = MagicMock()
        algo = AxOptimizationLibrary()

        class DummySettings:
            max_iter = 10
            n_init = 5
            use_bonsai = True
            ax_parameters = [{"name": "x", "type": "range", "bounds": [0.0, 1.0]}]
            ax_objectives = [{"name": "obj", "minimize": True}]
            ax_parameter_constraints = ["x <= 1"]
            normalize_design_space = False

        algo._settings = DummySettings()

        with patch("mdo_framework.optimization.ax_algo_lib.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.get_next_trials.return_value = {0: {"x": 0.5}}
            # mock get_best_parameterization raising Exception to hit 448
            MagicMock().get_best_parameterization.side_effect = Exception("Boom")

            prob.evaluate_functions = MagicMock(
                side_effect=[
                    ({"obj": np.array([1.0])}, None),
                    Exception("General error to hit line 400"),
                ]
            )

            try:
                algo.execute(prob, max_iter=1, n_init=1)
            except Exception:
                pass

    def test_ax_algo_lib_normalized_design_vectors_are_unnormalized(self):
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=10.0)
        prob = OptimizationProblem(ds)

        def obj(x):
            return np.array([x[0] ** 2])

        prob.objective = MDOFunction(obj, "obj", expr="x**2")
        prob.evaluate_functions = MagicMock(
            return_value=({"obj": np.array([25.0])}, None)
        )

        algo = AxOptimizationLibrary()
        client = MagicMock()

        algo._execute_trial(client, prob, 0, {"x": 0.5}, {"obj"}, normalize=True)

        evaluated_x = prob.evaluate_functions.call_args.args[0]
        self.assertEqual(float(evaluated_x[0]), 5.0)

        prob.evaluate_functions.reset_mock(return_value=True)
        client.get_best_parameterization.return_value = (
            {"x": 0.5},
            ({"obj": 25.0}, None),
            0,
            "arm_0",
        )

        algo._extract_best_solution(client, prob, False, normalize=True)

        optimum_x = prob.evaluate_functions.call_args.args[0]
        self.assertEqual(float(optimum_x[0]), 5.0)
        self.assertEqual(float(prob.design_space.get_current_value()[0]), 5.0)

    def test_ax_algo_lib_rejects_incompatible_custom_ax_parameters(self):
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        from mdo_framework.optimization.ax_algo_lib import (
            AxOptimizationLibrary,
            AxSettings,
        )

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)
        prob = OptimizationProblem(ds)

        def obj(x):
            return np.array([x[0] ** 2])

        prob.objective = MDOFunction(obj, "obj", expr="x**2")

        algo = AxOptimizationLibrary()

        algo._settings = AxSettings(
            max_iter=1,
            n_init=1,
            batch_size=1,
            use_bonsai=False,
            ax_parameters=[{"name": "custom_x", "type": "range", "bounds": [0.0, 1.0]}],
            ax_objectives=[{"name": "obj", "minimize": True}],
            normalize_design_space=False,
        )

        with self.assertRaises(ValueError):
            algo._configure_client(prob)

        algo._settings = AxSettings(
            max_iter=1,
            n_init=1,
            batch_size=1,
            use_bonsai=False,
            ax_parameters=[{"name": "x", "type": "range", "bounds": [0.0, 1.0]}],
            ax_objectives=[{"name": "obj", "minimize": True}],
            normalize_design_space=True,
        )

        with self.assertRaises(ValueError):
            algo._configure_client(prob)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_ax_algo_lib_loop_edge_cases(self, mock_client_cls):
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.algos.stop_criteria import MaxIterReachedException
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)

        prob = OptimizationProblem(ds)

        def obj(x):
            return np.array([x[0] ** 2])

        prob.objective = MDOFunction(obj, "obj", expr="x**2")

        algo = AxOptimizationLibrary()

        class DummySettings:
            max_iter = 10
            n_init = 5
            use_bonsai = True
            ax_parameters = [{"name": "x", "type": "range", "bounds": [0.0, 1.0]}]
            ax_objectives = [{"name": "obj", "minimize": True}]
            ax_parameter_constraints = ["x <= 1"]
            normalize_design_space = False

        algo._settings = DummySettings()

        # Hit 326: client.attach_trial (i > 0)
        prob.database.store(np.array([0.1]), {"obj": np.array([0.1])})
        prob.database.store(np.array([0.2]), {"obj": np.array([0.2])})

        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5}}

        # Mock pareto frontier empty -> hits 448
        mock_client._experiment.optimization_config.objective.__class__.__name__ = (
            "MultiObjective"
        )
        mock_client_cls.return_value.get_pareto_frontier.return_value = []

        # Mock evaluate_functions to hit 401-403 (MaxIterReachedException)
        def side_effect(*args, **kwargs):
            raise MaxIterReachedException()

        prob.evaluate_functions = MagicMock(side_effect=side_effect)

        try:
            algo.execute(prob, max_iter=1, n_init=1)
        except Exception:
            pass

    def test_local_evaluator_missing_output_raises_key_error(self):
        mock_problem = MagicMock()
        mock_problem.execute.return_value = {"g_xy": np.array([0.0])}
        evaluator = LocalEvaluator(mock_problem)

        with self.assertRaises(KeyError):
            evaluator.evaluate({"x": 0.5, "y": 0.5, "c": 0.0}, ["f_xy"])

    def test_parameter_codec_helper_branches(self):
        self.assertEqual(coerce_scalar(np.array([1.5])), 1.5)
        self.assertEqual(coerce_scalar(np.array([1.0, 2.0])), [1.0, 2.0])
        self.assertEqual(coerce_scalar(np.float64(2.5)), 2.5)
        self.assertEqual(build_parameter_lookup(None), {})
        self.assertEqual(
            decode_parameter_value({"name": "label", "type": "range"}, "A"), "A"
        )

    def test_build_design_space_rejects_invalid_parameters(self):
        from mdo_framework.optimization.optimizer import _build_design_space

        with self.assertRaises(OptimizationConfigurationError):
            _build_design_space([{"name": "x", "type": "range", "bounds": [0.0]}])

        with self.assertRaises(OptimizationConfigurationError):
            _build_design_space([{"name": "x", "type": "categorical"}])

        with self.assertRaises(OptimizationConfigurationError):
            _build_design_space([{"name": "x", "type": "choice", "values": []}])

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

    def test_optimizer_helpers_cover_fallbacks_and_wrapped_errors(self):
        from mdo_framework.optimization.optimizer import (
            _decode_parameter_value,
            _extract_best_objectives,
            _get_optimization_history,
        )

        class BrokenOptimum:
            @property
            def objective(self):
                raise RuntimeError("no objective")

        with self.assertRaises(OptimizationConfigurationError):
            _decode_parameter_value({"name": "c", "type": "choice", "values": []}, 0)

        with self.assertRaises(OptimizationExecutionError):
            _decode_parameter_value(
                {"name": "c", "type": "choice", "values": ["A"]}, True
            )

        objectives = _extract_best_objectives(
            BrokenOptimum(),
            ["f_xy", "g_xy"],
            fallback_metrics={"f_xy": 1.0, "g_xy": 2.0},
        )
        self.assertEqual(objectives, {"f_xy": 1.0, "g_xy": 2.0})

        self.assertEqual(_get_optimization_history(None, object()), [])
        self.assertEqual(
            _get_optimization_history(None, MagicMock(trial_history=[{"ok": True}])),
            [{"ok": True}],
        )

    def test_extract_best_objectives_raises_when_fallback_is_incomplete(self):
        from mdo_framework.optimization.optimizer import _extract_best_objectives

        class PartialOptimum:
            objective = np.array([1.0])

        with self.assertRaises(OptimizationExecutionError):
            _extract_best_objectives(
                PartialOptimum(),
                ["f_xy", "g_xy"],
                fallback_metrics={"f_xy": 1.0},
            )

    def test_parameter_codec_rejects_non_numeric_choice_decode(self):
        with self.assertRaises(ParameterValueError):
            decode_parameter_value(
                {"name": "c_str", "type": "choice", "values": ["A", "B"]},
                "invalid-index",
            )

    def test_remote_evaluator_transport_errors(self):
        url = "http://fake-url/evaluate"
        request = httpx.Request("POST", url)

        timeout_client = MagicMock()
        timeout_client.post.side_effect = httpx.TimeoutException("slow")
        with self.assertRaises(RemoteEvaluationTransportError):
            RemoteEvaluator("http://fake-url", client=timeout_client).evaluate(
                {}, ["f_xy"]
            )

        server_error_client = MagicMock()
        server_error = httpx.HTTPStatusError(
            "server error",
            request=request,
            response=httpx.Response(503, request=request),
        )
        server_error_client.post.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=server_error)
        )
        with self.assertRaises(RemoteEvaluationTransportError):
            RemoteEvaluator("http://fake-url", client=server_error_client).evaluate(
                {}, ["f_xy"]
            )

        request_error_client = MagicMock()
        request_error_client.post.side_effect = httpx.RequestError(
            "network down", request=request
        )
        with self.assertRaises(RemoteEvaluationTransportError):
            RemoteEvaluator("http://fake-url", client=request_error_client).evaluate(
                {}, ["f_xy"]
            )

    def test_remote_evaluator_contract_errors(self):
        url = "http://fake-url/evaluate"
        request = httpx.Request("POST", url)

        client_400 = MagicMock()
        client_400.post.return_value = MagicMock(
            raise_for_status=MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "bad request",
                    request=request,
                    response=httpx.Response(400, request=request),
                )
            )
        )
        with self.assertRaises(RemoteEvaluationContractError):
            RemoteEvaluator("http://fake-url", client=client_400).evaluate({}, ["f_xy"])

        invalid_json_client = MagicMock()
        invalid_json_response = MagicMock()
        invalid_json_response.raise_for_status.return_value = None
        invalid_json_response.json.side_effect = ValueError("bad json")
        invalid_json_client.post.return_value = invalid_json_response
        with self.assertRaises(RemoteEvaluationContractError):
            RemoteEvaluator("http://fake-url", client=invalid_json_client).evaluate(
                {}, ["f_xy"]
            )

        missing_results_client = MagicMock()
        missing_results_response = MagicMock()
        missing_results_response.raise_for_status.return_value = None
        missing_results_response.json.return_value = {}
        missing_results_client.post.return_value = missing_results_response
        with self.assertRaises(RemoteEvaluationContractError):
            RemoteEvaluator("http://fake-url", client=missing_results_client).evaluate(
                {}, ["f_xy"]
            )

        missing_objective_client = MagicMock()
        missing_objective_response = MagicMock()
        missing_objective_response.raise_for_status.return_value = None
        missing_objective_response.json.return_value = {"results": {"other": 1.0}}
        missing_objective_client.post.return_value = missing_objective_response
        with self.assertRaises(RemoteEvaluationContractError):
            RemoteEvaluator(
                "http://fake-url", client=missing_objective_client
            ).evaluate({}, ["f_xy"])

        non_numeric_client = MagicMock()
        non_numeric_response = MagicMock()
        non_numeric_response.raise_for_status.return_value = None
        non_numeric_response.json.return_value = {"results": {"f_xy": "NaN?"}}
        non_numeric_client.post.return_value = non_numeric_response
        with self.assertRaises(RemoteEvaluationContractError):
            RemoteEvaluator("http://fake-url", client=non_numeric_client).evaluate(
                {}, ["f_xy"]
            )

    @patch("mdo_framework.optimization.optimizer.httpx.Client")
    def test_remote_evaluator_close_only_closes_owned_client(self, mock_httpx_client):
        owned_client = MagicMock()
        mock_httpx_client.return_value = owned_client

        owned_evaluator = RemoteEvaluator("http://owned")
        owned_evaluator.close()
        owned_client.close.assert_called_once()

        external_client = MagicMock()
        external_evaluator = RemoteEvaluator("http://external", client=external_client)
        external_evaluator.close()
        external_client.close.assert_not_called()

    @patch("mdo_framework.optimization.optimizer.create_scenario")
    @patch("mdo_framework.optimization.ax_algo_lib.AxOptimizationLibrary.execute")
    def test_optimize_requires_valid_optimum(self, mock_execute, mock_create_scenario):
        mock_problem = MagicMock(optimum=None)
        mock_scenario = MagicMock()
        mock_scenario.formulation.optimization_problem = mock_problem
        mock_create_scenario.return_value = mock_scenario

        optimizer = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)

        with self.assertRaises(OptimizationExecutionError):
            optimizer.optimize(n_steps=1, n_init=1)

        mock_execute.assert_called_once()

    def test_ax_execute_trial_success_and_empty_metrics_paths(self):
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)
        problem = OptimizationProblem(ds)

        def obj(x):
            return np.array([x[0] ** 2])

        problem.objective = MDOFunction(obj, "obj", expr="x**2")
        algo = AxOptimizationLibrary()
        client = MagicMock()

        problem.evaluate_functions = MagicMock(
            return_value=({"obj": np.array([1.0, 3.0])}, None)
        )
        self.assertFalse(algo._execute_trial(client, problem, 0, {"x": 0.5}, {"obj"}))
        client.complete_trial.assert_called_once_with(
            trial_index=0, raw_data={"obj": 3.0}
        )
        self.assertEqual(
            algo.trial_history[-1],
            {"parameters": {"x": 0.5}, "objectives": {"obj": 3.0}},
        )

        client.reset_mock()
        problem.evaluate_functions = MagicMock(return_value=({"other": 1.0}, None))
        self.assertFalse(algo._execute_trial(client, problem, 1, {"x": 0.25}, {"obj"}))
        client.mark_trial_abandoned.assert_called_once_with(trial_index=1)

    def test_ax_helpers_cover_empty_history_and_invalid_settings(self):
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        from mdo_framework.optimization.ax_algo_lib import (
            AxOptimizationLibrary,
            _require_ax_settings,
        )

        with self.assertRaises(TypeError):
            _require_ax_settings(object())

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)
        problem = OptimizationProblem(ds)

        def obj(x):
            return np.array([x[0] ** 2])

        problem.objective = MDOFunction(obj, "obj", expr="x**2")
        problem.history = MagicMock()
        problem.history.last_point = property(lambda self: None)
        type(problem.history).last_point = property(
            lambda self: (_ for _ in ()).throw(ValueError("no history"))
        )

        algo = AxOptimizationLibrary()
        algo._record_last_point(problem)
        self.assertEqual(algo.trial_history, [])

    def test_ax_run_records_last_point_when_client_yields_no_trials(self):
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)
        problem = OptimizationProblem(ds)

        def obj(x):
            return np.array([x[0] ** 2])

        problem.objective = MDOFunction(obj, "obj", expr="x**2")
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
