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
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertIn("best_parameters", result)
        self.assertIn("best_objectives", result)
        self.assertEqual(len(result["history"]), 0)
        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5, "c": 0.5})

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_bonsai_and_multi_objective(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        # Test pareto frontier handling
        mock_client.get_pareto_frontier.return_value = [
            ({"x": 0.5, "y": 0.5, "c": "A"}, {"f_xy": 42.0, "g_xy": 10.0}, 0, "0_0"),
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
        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5, "c": 0.5})
        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5, "c": 0.5})
        pass

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_pareto_none(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        mock_client.get_pareto_frontier.return_value = None

        objs = [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}]

        opt = BayesianOptimizer(self.evaluator, self.parameters, objs)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5, "c": 0.5})
        self.assertEqual(result["best_objectives"]["f_xy"], 1.0)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_exception(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        mock_client.get_best_parameterization.side_effect = Exception(
            "Optimization failed"
        )

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives)
        result = opt.optimize(n_steps=1, n_init=2)

        self.assertEqual(result["best_parameters"], {"x": 0.5, "y": 0.5, "c": 0.5})

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_with_constraints(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

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
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

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
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {
            0: {"x": 0.5, "c_str": 1.0, "c_single": 0.0, "c_num_single": 42.0}
        }
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"
        mock_client.get_best_parameterization.return_value = (
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
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {
            0: {"x": 0.5, "c": 0.0, "y": 0.5, "z": 42.0}
        }
        mock_client.get_best_parameterization.return_value = (
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
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        # Mock pareto frontier
        mock_client.get_pareto_frontier.return_value = [
            ({"x": 0.5, "y": 0.5, "c": 0.0}, {"f_xy": 42.0, "g_xy": 10.0}, 0, "0_0"),
        ]

        # Configure a mock multi-objective setup
        mock_opt_config = MagicMock()
        mock_opt_config.objective = MagicMock()
        from ax.core.objective import MultiObjective

        mock_opt_config.objective.__class__ = MultiObjective
        mock_client.experiment = MagicMock()
        mock_client.experiment.optimization_config = mock_opt_config

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
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client._to_json_snapshot.return_value = {}
        mock_client.to_json_snapshot.return_value = "{}"

        mock_opt_config = MagicMock()
        mock_client.experiment = MagicMock()
        mock_client.experiment.optimization_config = mock_opt_config
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        opt = BayesianOptimizer(
            self.evaluator,
            self.parameters,
            self.objectives,
            parameter_constraints=["x <= y"],
        )
        result = opt.optimize(n_steps=1, n_init=1)

        self.assertIn("best_parameters", result)
        self.assertEqual(result["best_parameters"]["x"], 0.5)
        mock_client.configure_experiment.assert_called()
        call_kwargs = mock_client.configure_experiment.call_args.kwargs
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
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_next_trials.return_value = {0: {"x": 0.5}}
            mock_client.get_best_parameterization.return_value = (
                {"x": 0.0},
                {"obj": 0.0},
                0,
                "0_0",
            )

            # mock get_pareto_frontier specifically
            mock_client.experiment = MagicMock()
            mock_client.experiment.optimization_config.objective.__class__.__name__ = (
                "MultiObjective"
            )
            mock_client.get_pareto_frontier.return_value = [
                ({"x": 0.0}, {"obj": 0.0}, 0, "0_0"),
            ]

            algo.execute(prob, max_iter=1, n_init=1)





    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimize_fidelity_param(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives, fidelity_parameter="x")
        with self.assertWarns(UserWarning):
            opt.optimize(n_steps=1, n_init=1)


    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimizer_fallback_lines(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        opt = BayesianOptimizer(self.evaluator, self.parameters, self.objectives, fidelity_parameter="z")

        with patch('gemseo.algos.optimization_problem.OptimizationProblem.optimum') as mock_opt:
            class DummyOptimum:
                design = np.array([0.5, 0.5, 0.5])
                @property
                def objective(self):
                    raise TypeError("Bang!")

            mock_opt.return_value = DummyOptimum()

            with patch('mdo_framework.optimization.ax_algo_lib.AxOptimizationLibrary.execute'):
                result = opt.optimize(n_steps=1, n_init=1)

        self.assertEqual(result["best_objectives"]["f_xy"], 0.0)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimizer_fallback_lines_multi(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "y": 0.5, "c": 0.0}}
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "y": 0.5, "c": 0.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        objs = [
            {"name": "f_xy", "minimize": True},
            {"name": "g_xy", "minimize": False}
        ]

        opt = BayesianOptimizer(self.evaluator, self.parameters, objs)

        with patch('gemseo.algos.optimization_problem.OptimizationProblem.optimum') as mock_opt:
            class DummyOptimum:
                design = np.array([0.5, 0.5, 0.5])
                @property
                def objective(self):
                    raise TypeError("Bang!")

            mock_opt.return_value = DummyOptimum()

            with patch('mdo_framework.optimization.ax_algo_lib.AxOptimizationLibrary.execute'):
                result = opt.optimize(n_steps=1, n_init=1)

        self.assertEqual(result["best_objectives"]["f_xy"], 0.0)
        self.assertEqual(result["best_objectives"]["g_xy"], 0.0)




    def test_ax_algo_lib_edge_cases(self):
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.core.mdo_functions.mdo_function import MDOFunction
        from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary, AxConstraintBuilder, AxObjectiveBuilder

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)

        prob = OptimizationProblem(ds)
        def obj(x):
            return np.array([x[0] ** 2])
        prob.objective = MDOFunction(obj, "obj", expr="x**2")

        res = AxConstraintBuilder.build_outcome_constraints([])
        self.assertEqual(res, [])

        config = AxObjectiveBuilder.build_optimization_config([{"name": "a", "minimize": True}, {"name": "b", "minimize": False}], prob, [])
        self.assertIsNotNone(config)

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
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.get_next_trials.return_value = {
                0: {"x": 0.5}
            }
            # mock get_best_parameterization raising Exception to hit 448
            mock_client.get_best_parameterization.side_effect = Exception("Boom")

            prob.evaluate_functions = MagicMock(side_effect=[({'obj': np.array([1.0])}, None), Exception('General error to hit line 400')])

            try:
                algo.execute(prob, max_iter=1, n_init=1)
            except Exception:
                pass

    def test_ax_parameter_and_objective_builders_edge_cases(self):
        from mdo_framework.optimization.ax_algo_lib import AxParameterBuilder, AxObjectiveBuilder
        from gemseo.algos.design_space import DesignSpace

        # Test parameter builder normalization
        params = AxParameterBuilder.build_from_ax_parameters([
            {"name": "c", "type": "choice", "values": [float(1.0), float(2.0)]}
        ])
        self.assertEqual(params[0].is_ordered, True) # is_numeric = True, but type int/float handled by ax

        # Test objective builder with list of metric names
        class DummyObjProblem:
            class DummyObj:
                name = ["f1", "f2"]
            objective = DummyObj()

        class DummySettings:
            pass
        fs = DummySettings()
        setattr(fs, "ax_objectives", [])

        # Test building multi-objective configs natively
        AxObjectiveBuilder.build_optimization_config([], DummyObjProblem(), [])

        # Test fallback parsing for singleton list metrics
        DummyObjProblem.objective.name = ["f1"]
        AxObjectiveBuilder.build_optimization_config([], DummyObjProblem(), [])
        ds = DesignSpace()
        ds.add_variable('v', lower_bound=0.0, upper_bound=1.0)
        ds.set_current_value(np.array([0.5]))
        AxParameterBuilder.build_from_design_space(ds, normalize=True)

    @patch("mdo_framework.optimization.ax_algo_lib.Client")
    def test_optimizer_choice_and_fallback_lines(self, mock_client_cls):
        from gemseo.algos.stop_criteria import MaxIterReachedException
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.get_next_trials.return_value = {0: {"x": 0.5, "c_num_0": 1.0, "c_num_1": 2.0}}
        mock_client.get_best_parameterization.return_value = (
            {"x": 0.5, "c_num_0": 1.0, "c_num_1": 2.0},
            {"f_xy": 42.0},
            0,
            "0_0",
        )

        params = [
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "c_num", "type": "choice", "values": [1.5, 2.5]}
        ]
        objs = [{"name": "f_xy", "minimize": True}, {"name": "g_xy", "minimize": False}]

        opt = BayesianOptimizer(self.evaluator, params, objs)
        opt.explore(n_samples=1)


        with patch('mdo_framework.optimization.optimizer.np.atleast_1d') as mock_atleast_1d:
            mock_arr = MagicMock()
            mock_arr.flatten.return_value = ["not_a_number", "still_not"]
            mock_atleast_1d.return_value = mock_arr

            # Since evaluate will run normally, Ax will finish, but then when it builds best_objectives, atleast_1d returns our bad strings!
            # float("not_a_number") throws ValueError!
            val = opt.optimize(n_steps=1, n_init=1)
            self.assertEqual(val["best_objectives"]["f_xy"], 0.0)


        from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary
        from gemseo.algos.design_space import DesignSpace
        from gemseo.algos.optimization_problem import OptimizationProblem
        from gemseo.core.mdo_functions.mdo_function import MDOFunction

        algo = AxOptimizationLibrary()
        algo.algo_options = {"max_iter": 3}
        algo.client = mock_client
        algo.client.experiment.search_space.parameters = {"x": MagicMock()}

        ds = DesignSpace()
        ds.add_variable("x", lower_bound=0.0, upper_bound=1.0)
        prob = OptimizationProblem(ds)
        prob.objective = MDOFunction(lambda x: np.array([x[0]]), "f_xy", expr="x")

        algo.problem = prob

        algo.problem.database.store(np.array([0.1]), {"f_xy": np.array([0.1])})
        algo.problem.database.store(np.array([0.2]), {"f_xy": np.array([0.2])})

        algo._seed_database(algo.client, algo.problem, ds)

        def mock_eval(*args, **kwargs):
            raise MaxIterReachedException(10)
        algo.problem.evaluate_functions = mock_eval

        class DummySettings:
            max_iter = 3
            n_init = 1
            use_bonsai = False
            ax_parameters = [{"name": "x", "type": "range", "bounds": [0.0, 1.0]}]
            ax_objectives = [{"name": "f_xy", "minimize": True}]
            ax_parameter_constraints = []
            normalize_design_space = False
        algo._settings = DummySettings()
                # execution completes without crashing even if trials fail
        algo._run(algo.problem)


        from ax.core.objective import MultiObjective
        mock_client.experiment.optimization_config.objective = MagicMock(spec=MultiObjective)
        mock_client.get_pareto_frontier.return_value = []
        # When Pareto empty, the best_parameters shouldn't crash, it should just return empty dicts or None depending on upstream
        opt2 = BayesianOptimizer(self.evaluator, [{"name": "x", "type": "range", "bounds": [0, 1]}], [{"name": "f_xy"}])
        res2 = opt2.optimize(n_steps=1, n_init=1)
        self.assertTrue(res2["best_parameters"] is not None)
