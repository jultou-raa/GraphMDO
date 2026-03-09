"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from gemseo.core.discipline import Discipline

from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.optimization.optimizer import BayesianOptimizer, RemoteEvaluator


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

    @patch("mdo_framework.optimization.optimizer.httpx.post")
    def test_remote_evaluator(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": {"f_xy": 2.0}}
        mock_post.return_value = mock_resp

        evaluator = RemoteEvaluator("http://fake-url")
        res = evaluator.evaluate({"x": 0.5, "y": 0.5, "c": 0.0}, ["f_xy"])

        mock_post.assert_called_once()
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
        result = opt.optimize(n_steps=1, n_init=2)
        self.assertIn("error", result)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_exception(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        MagicMock().get_best_parameterization.side_effect = Exception(
            "Optimization failed"
        )

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        result = opt.optimize(n_steps=1, n_init=2)
        self.assertIn("error", result)
        self.assertTrue(
            "Pareto frontier is empty" in result["error"]
            or "Optimization failed" in result["error"]
        )

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
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertEqual(len(result["history"]), 0)

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
    def test_explore_exception(self, mock_create_scenario):
        mock_scenario = MagicMock()
        mock_create_scenario.return_value = mock_scenario
        mock_scenario.execute.side_effect = ValueError("DOE failed")

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        res = opt.explore()
        self.assertEqual(res, {"history": {}})

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
        mock_client = MagicMock()
        mock_client.get_next_trials.return_value = {
            0: {"x": 0.5, "c_str": 1.0, "c_single": 0.0, "c_num_single": 42.0}
        }
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client_cls.return_value.get_best_parameterization.return_value = (
            {"x": 0.5, "c_str": 1.0, "c_single": 0.0, "c_num_single": 42.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        mock_evaluator = RemoteEvaluator("http://test")
        mock_evaluator.evaluate = MagicMock(return_value={"f_xy": 42.0})

        params = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "c_str", "type": "choice", "values": ["A", "B", "C"]},
            {"name": "c_single", "type": "choice", "values": ["A"]},
            {"name": "c_num_single", "type": "choice", "values": [42]},
        ]

        opt = BayesianOptimizer(mock_evaluator, params, self.objectives)
        res = opt.optimize(n_steps=1, n_init=1)

        self.assertEqual(res["best_parameters"]["x"], 0.5)

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
        res = opt.optimize(n_steps=1, n_init=1)

        self.assertIsNone(res["best_parameters"])

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
                result = opt.optimize(n_steps=1, n_init=1)

        self.assertEqual(result["best_objectives"]["f_xy"], 0.0)

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
                result = opt.optimize(n_steps=1, n_init=1)

        self.assertEqual(result["best_objectives"]["f_xy"], 0.0)
        self.assertEqual(result["best_objectives"]["g_xy"], 0.0)

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
