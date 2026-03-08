"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import logging
import warnings
from typing import Any, cast

import numpy as np
import traceback

# Suppress warnings that clutter production logs (e.g., pandas FutureWarning in Ax)

from ax.adapter.registry import Generators as Models
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from ax.core.map_metric import MapMetric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.generation_strategy.generation_node import GenerationStep
from ax.generation_strategy.generation_strategy import (
    GenerationStrategy,
)
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from gemseo.algos.evaluation_problem import EvaluationProblem
from gemseo.algos.opt.base_optimization_library import (
    BaseOptimizationLibrary,
    OptimizationAlgorithmDescription,
)
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.stop_criteria import MaxIterReachedException

logger = logging.getLogger(__name__)

from typing import TypedDict


class AxParameterDict(TypedDict, total=False):
    name: str
    type: str
    bounds: list[float]
    values: list[Any]
    is_ordered: bool


class AxObjectiveDict(TypedDict, total=False):
    name: str | list[str]
    minimize: bool
    maximize_objective: bool
    threshold: float


_DEFAULT_MAX_THRESHOLD = -1e6
_DEFAULT_MIN_THRESHOLD = 1e6


def _get_param_name(var_name: str, i: int, size: int) -> str:
    """Constructs the Ax parameter name for array elements."""
    return f"{var_name}_{i}" if size > 1 else var_name


class AxSettings(BaseOptimizerSettings):
    """Settings for Ax optimization."""

    max_iter: int = 10
    batch_size: int = 1
    n_init: int = 5
    use_bonsai: bool = False
    ax_parameters: list[dict[str, Any]] | None = None
    ax_objectives: list[dict[str, Any]] | None = None
    ax_parameter_constraints: list[str] | None = None
    normalize_design_space: bool = False

    """Builder for Ax parameter configurations."""


class AxConfigurationFactory:
    """Factory for building Ax platform configurations from GEMSEO inputs."""

    @staticmethod
    def build_from_ax_parameters(
        ax_parameters: list[AxParameterDict],
    ) -> list[RangeParameterConfig | ChoiceParameterConfig]:
        ax_params = []
        for p in ax_parameters:
            if p["type"] == "range":
                if len(p["bounds"]) != 2:
                    raise ValueError(f"Range parameter {p['name']} requires 2 bounds.")
                ax_params.append(
                    RangeParameterConfig(
                        name=p["name"],
                        bounds=(p["bounds"][0], p["bounds"][1]),
                        parameter_type=p.get("value_type", "float"),
                    )
                )
            elif p["type"] == "choice":
                if not p.get("values"):
                    raise ValueError(
                        f"Choice parameter {p['name']} requires a list of values."
                    )
                ax_params.append(
                    ChoiceParameterConfig(
                        name=p["name"],
                        values=p["values"],
                        parameter_type=p.get(
                            "value_type",
                            "str" if isinstance(p["values"][0], str) else "float",
                        ),
                    )
                )
        return ax_params

    @staticmethod
    def build_from_design_space(
        design_space: Any, normalize: bool
    ) -> list[RangeParameterConfig]:
        ax_params = []
        for var_name in design_space.variable_names:
            size = design_space.variable_sizes[var_name]
            l_b, u_b = (
                design_space.get_lower_bound(var_name),
                design_space.get_upper_bound(var_name),
            )
            if normalize:
                # Assuming GEMSEO handles normalization internally or we just pass float.
                # This was a bug in original code too, let's keep it simple since we just want to avoid unexpected kwargs.
                l_b, u_b = [0.0] * size, [1.0] * size
            # Introspect GEMSEO variable types to preserve integers
            var_type_list = design_space.variable_types.get(var_name, [])
            for i in range(size):
                param_name = _get_param_name(var_name, i, size)
                # Determine if this specific component is an integer
                is_int = False
                if i < len(var_type_list) and var_type_list[i] is int:
                    is_int = True

                # If normalizing, we usually force float because normalized bounds are 0.0 to 1.0
                p_type = "int" if (is_int and not normalize) else "float"

                if p_type == "int":
                    ax_params.append(
                        RangeParameterConfig(
                            name=param_name,
                            bounds=(int(l_b[i]), int(u_b[i])),
                            parameter_type=p_type,
                        )
                    )
                else:
                    ax_params.append(
                        RangeParameterConfig(
                            name=param_name,
                            bounds=(float(l_b[i]), float(u_b[i])),
                            parameter_type=p_type,
                        )
                    )
        return ax_params

    @staticmethod
    def build_optimization_config(
        ax_objectives: list[AxObjectiveDict] | None,
        problem: OptimizationProblem,
        ax_outcome_constraints: list[OutcomeConstraint],
    ) -> OptimizationConfig | MultiObjectiveOptimizationConfig:
        if ax_objectives and len(ax_objectives) > 1:
            ax_objs = []
            objective_thresholds = []
            for obj in ax_objectives:
                metric_name = obj["name"]
                minimize = obj.get("minimize", problem.minimize_objective)
                threshold = obj.get("threshold", None)

                ax_objs.append(
                    Objective(
                        metric=MapMetric(name=metric_name),
                        minimize=minimize,
                    )
                )

                if threshold is not None:
                    from ax.core.optimization_config import ObjectiveThreshold

                    objective_thresholds.append(
                        ObjectiveThreshold(
                            metric=MapMetric(name=metric_name),
                            bound=float(threshold),
                            relative=False,
                            op=ComparisonOp.LEQ if minimize else ComparisonOp.GEQ,
                        )
                    )

            return MultiObjectiveOptimizationConfig(
                objective=MultiObjective(objectives=ax_objs),
                objective_thresholds=objective_thresholds
                if objective_thresholds
                else None,
                outcome_constraints=ax_outcome_constraints,
            )
        else:
            if ax_objectives and len(ax_objectives) == 1:
                metric_name = ax_objectives[0]["name"]
                minimize = ax_objectives[0].get("minimize", problem.minimize_objective)
            else:
                metric_name = (
                    problem.objective.name[0]
                    if isinstance(problem.objective.name, list)
                    else problem.objective.name
                )
                minimize = problem.minimize_objective

            return OptimizationConfig(
                objective=Objective(
                    metric=MapMetric(name=metric_name), minimize=minimize
                ),
                outcome_constraints=ax_outcome_constraints,
            )

    @staticmethod
    def build_outcome_constraints(
        constraints: list[Any],
    ) -> list[OutcomeConstraint]:
        ax_outcome_constraints = []
        for c in constraints:
            if c.f_type == "ineq":
                ax_outcome_constraints.append(
                    OutcomeConstraint(
                        metric=MapMetric(name=c.name),
                        op=ComparisonOp.LEQ,
                        bound=0.0,
                        relative=False,
                    )
                )
        return ax_outcome_constraints


class AxOptimizationLibrary(BaseOptimizationLibrary):
    """Ax Platform Optimization algorithm wrapper for GEMSEO."""

    LIBRARY_NAME = "Ax_Platform"

    ALGORITHM_INFOS = {
        "Ax_Bayesian": OptimizationAlgorithmDescription(
            algorithm_name="Ax_Bayesian",
            internal_algorithm_name="Ax_Bayesian",
            library_name=LIBRARY_NAME,
            description="Bayesian Optimization using Ax platform.",
            website="https://ax.dev/",
            Settings=AxSettings,
            handle_equality_constraints=False,
            handle_inequality_constraints=True,
            handle_multiobjective=True,
            handle_integer_variables=True,
            positive_constraints=False,
            require_gradient=False,
            for_linear_problems=False,
        )
    }

    def __init__(
        self, algo_name: str = "Ax_Bayesian", client_factory=None, config_factory=None
    ) -> None:
        super().__init__(algo_name=algo_name)
        self.client_factory = client_factory or Client
        self.config_factory = config_factory or AxConfigurationFactory

    def _get_generation_strategy(
        self, use_bonsai: bool, n_init: int
    ) -> GenerationStrategy:
        if use_bonsai:
            logger.warning("Experimental feature BONSAI algorithm is activated.")
            return GenerationStrategy(
                name="bonsai",
                nodes=[
                    GenerationStep(
                        generator=Models.SOBOL,
                        num_trials=n_init,
                        min_trials_observed=n_init,
                    ),
                    GenerationStep(
                        generator=Models.BOTORCH_MODULAR,
                        num_trials=-1,
                    ),
                ],
            )
        return GenerationStrategy(
            name="botorch_modular",
            nodes=[
                GenerationStep(
                    generator=Models.SOBOL,
                    num_trials=n_init,
                    min_trials_observed=n_init,
                ),
                GenerationStep(
                    generator=Models.BOTORCH_MODULAR,
                    num_trials=-1,
                    generator_kwargs={
                        "botorch_acqf_class": qLogNoisyExpectedImprovement,
                    },
                ),
            ],
        )

    @staticmethod
    def _extract_seed_params(x_seed: np.ndarray, design_space: Any) -> dict[str, float]:
        seed_params: dict[str, float] = {}
        seed_offset = 0
        for var_name in design_space.variable_names:
            size = design_space.variable_sizes[var_name]
            for j in range(size):
                param_name = _get_param_name(var_name, j, size)
                seed_params[param_name] = float(x_seed[seed_offset + j])
            seed_offset += size
        return seed_params

    @staticmethod
    def _extract_seed_results(
        output: dict[str, Any], metric_names: set[str], c_names: set[str]
    ) -> dict[str, float]:
        seed_results = {}
        for metric in metric_names:
            val = output.get(metric)
            if val is not None:
                if isinstance(val, np.ndarray) and val.size > 1:
                    if metric in c_names:
                        logger.warning(
                            "Multi-dimensional array detected for constraint '%s'. "
                            "Using np.max for aggregation, which may create a non-smooth gradient landscape. "
                            "Consider defining a smooth aggregation function natively within the discipline.",
                            metric,
                        )
                    seed_results[metric] = float(np.max(val))
                else:
                    seed_results[metric] = (
                        float(val) if not isinstance(val, np.ndarray) else float(val[0])
                    )
        return seed_results

    def _seed_database(
        self, client: Client, problem: OptimizationProblem, design_space: Any
    ) -> None:
        obj_names = problem.objective.name
        if not isinstance(obj_names, list):
            obj_names = [obj_names]

        c_names = {c.name for c in problem.constraints}
        metric_names = set(obj_names) | c_names

        for i, (x_hash, output) in enumerate(problem.database.items()):
            x_seed = x_hash.unwrap()
            seed_params = self._extract_seed_params(x_seed, design_space)
            seed_results = self._extract_seed_results(output, metric_names, c_names)

            if seed_results:
                if i == 0:
                    trial_idx = client.attach_baseline(parameters=seed_params)
                else:
                    trial_idx = client.attach_trial(parameters=seed_params)
                client.complete_trial(trial_index=trial_idx, raw_data=seed_results)

    def _configure_client(self, problem: OptimizationProblem) -> Client:
        n_init = getattr(self._settings, "n_init", 5)
        use_bonsai = getattr(self._settings, "use_bonsai", False)
        ax_objectives = getattr(self._settings, "ax_objectives", None)
        ax_parameter_constraints = getattr(
            self._settings, "ax_parameter_constraints", None
        )

        design_space = problem.design_space
        gs = self._get_generation_strategy(use_bonsai, n_init)

        client = self.client_factory()

        ax_parameters = getattr(self._settings, "ax_parameters", None)
        if ax_parameters:
            ax_params = self.config_factory.build_from_ax_parameters(ax_parameters)
        else:
            normalize = getattr(self._settings, "normalize_design_space", False)
            ax_params = self.config_factory.build_from_design_space(
                design_space, normalize
            )

        client.configure_experiment(
            name="gemseo_ax_opt",
            parameters=ax_params,
            parameter_constraints=ax_parameter_constraints,
        )

        ax_outcome_constraints = self.config_factory.build_outcome_constraints(
            problem.constraints
        )

        opt_config = self.config_factory.build_optimization_config(
            ax_objectives, problem, ax_outcome_constraints
        )

        client.set_optimization_config(opt_config)
        client.set_generation_strategy(gs)
        return client

    def _execute_trial(
        self,
        client: Client,
        problem: OptimizationProblem,
        trial_index: int,
        parameters: dict[str, Any],
        obj_names: list[str],
    ) -> bool:
        design_space = problem.design_space
        x = np.zeros(design_space.dimension)
        offset = 0
        for var_name in design_space.variable_names:
            size = design_space.variable_sizes[var_name]
            for i in range(size):
                param_name = _get_param_name(var_name, i, size)
                x[offset + i] = parameters[param_name]
            offset += size

        try:
            out_dict, _ = problem.evaluate_functions(
                x, design_vector_is_normalized=False
            )
        except MaxIterReachedException:
            client.mark_trial_abandoned(trial_index=trial_index)
            return True
        except ValueError as e:
            logger.error("Failed to evaluate point: %s\n%s", e, traceback.format_exc())
            client.mark_trial_abandoned(trial_index=trial_index)
            return False

        res = {}
        for obj_name in obj_names:
            if obj_name in out_dict:
                obj_val = out_dict[obj_name]
                if isinstance(obj_val, (list, np.ndarray)) and len(obj_val) == 1:
                    res[obj_name] = float(obj_val[0])
                elif isinstance(obj_val, (int, float)):
                    res[obj_name] = float(obj_val)

        client.complete_trial(trial_index=trial_index, raw_data=res)
        return False

    def _extract_best_solution(
        self, client: Client, problem: OptimizationProblem
    ) -> None:
        design_space = problem.design_space
        is_moo = False
        ax_objectives = getattr(self._settings, "ax_objectives", []) or []
        if len(ax_objectives) > 1:
            is_moo = True

        if is_moo:
            pareto_front = client.get_pareto_frontier()
            if pareto_front:
                logger.warning(
                    "Multi-objective optimization completed. The Pareto frontier contains %d points. "
                    "Arbitrarily extracting the first lexicographical boundary point to update the problem's design space. "
                    "The full Pareto front should be exported via problem.database.",
                    len(pareto_front),
                )
                best_parameters = pareto_front[0][0]
            else:
                raise ValueError("Pareto frontier is empty")
        else:
            best_parameters, best_obj, trial_idx, arm_name = (
                client.get_best_parameterization()
            )

        x_opt = np.zeros(design_space.dimension)
        offset = 0
        for var_name in design_space.variable_names:
            size = design_space.variable_sizes[var_name]
            for i in range(size):
                param_name = _get_param_name(var_name, i, size)
                x_opt[offset + i] = best_parameters[param_name]
            offset += size

        try:
            problem.evaluate_functions(x_opt, design_vector_is_normalized=False)
        except MaxIterReachedException:
            pass
        design_space.set_current_value(x_opt)

    def _run(self, problem: EvaluationProblem) -> tuple[str, int]:
        """Executes the optimization algorithm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)

            problem = cast(OptimizationProblem, problem)

            client = self._configure_client(problem)
            self._seed_database(client, problem, problem.design_space)

            max_iter = getattr(self._settings, "max_iter", 10)
            total_trials = max_iter
            budget_exhausted = False

            obj_names = problem.objective.name
            if not isinstance(obj_names, list):
                obj_names = [obj_names]

            for _ in range(total_trials):
                if budget_exhausted:
                    break
                batch_size = getattr(self._settings, "batch_size", 1)
                trials = client.get_next_trials(max_trials=batch_size)
                for trial_index, parameters in trials.items():
                    budget_exhausted = self._execute_trial(
                        client, problem, trial_index, parameters, obj_names
                    )
                    if budget_exhausted:
                        break

            self._extract_best_solution(client, problem)

        return "Optimization completed successfully.", 0


AxOptimizationLibrary.algo_dict = {
    "Ax_Bayesian": {
        "algorithm_name": "Ax_Bayesian",
        "internal_algorithm_name": "Ax_Bayesian",
        "library_name": "Ax_Platform",
        "description": "Bayesian Optimization using Ax platform.",
        "website": "https://ax.dev/",
        "Settings": AxSettings,
        "handle_equality_constraints": False,
        "handle_inequality_constraints": True,
        "handle_multiobjective": True,
        "handle_integer_variables": True,
        "positive_constraints": False,
        "require_gradient": False,
        "for_linear_problems": False,
    }
}
