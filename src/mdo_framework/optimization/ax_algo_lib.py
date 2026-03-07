"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import logging
import warnings
from typing import Any, cast

import numpy as np

# Suppress warnings that clutter production logs (e.g., pandas FutureWarning in Ax)
warnings.filterwarnings("ignore", category=FutureWarning)

from ax.adapter.registry import Generators as Models
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from ax.core.map_metric import MapMetric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    ObjectiveThreshold,
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


class AxSettings(BaseOptimizerSettings):
    """Settings for Ax optimization."""

    max_iter: int = 10
    n_init: int = 5
    use_bonsai: bool = False
    ax_parameters: list[dict[str, Any]] | None = None
    ax_objectives: list[dict[str, Any]] | None = None
    ax_parameter_constraints: list[str] | None = None
    normalize_design_space: bool = False


class AxParameterBuilder:
    """Builder for Ax parameter configurations."""

    @staticmethod
    def build_from_ax_parameters(ax_parameters: list[dict[str, Any]]) -> list[Any]:
        """Builds Ax parameters from a given list of configurations."""
        ax_params = []
        for p in ax_parameters:
            if p["type"] == "range":
                ax_params.append(
                    RangeParameterConfig(
                        name=p["name"],
                        parameter_type=p.get("value_type", "float"),
                        bounds=p["bounds"],
                    )
                )
            elif p["type"] == "choice":
                ax_params.append(
                    ChoiceParameterConfig(
                        name=p["name"],
                        parameter_type=p.get("value_type", "float"),
                        values=p["values"],
                        is_ordered=True,
                    )
                )
        return ax_params

    @staticmethod
    def build_from_design_space(design_space: Any, normalize: bool) -> list[Any]:
        """Builds Ax parameters directly from GEMSEO's DesignSpace."""
        from gemseo.algos.design_space_utils import get_value_and_bounds

        ax_params = []
        x_0, lb_full, ub_full = get_value_and_bounds(design_space, normalize)

        offset = 0
        for var_name in design_space.variable_names:
            size = design_space.variable_sizes[var_name]
            l_b = lb_full[offset : offset + size]
            u_b = ub_full[offset : offset + size]
            offset += size
            for i in range(size):
                param_name = f"{var_name}_{i}" if size > 1 else var_name

                is_float = True
                if normalize:
                    is_float = True
                p_type = "float" if is_float else "int"

                if l_b[i] == u_b[i]:
                    ax_params.append(
                        ChoiceParameterConfig(
                            name=param_name,
                            parameter_type=p_type,
                            values=[float(l_b[i]) if is_float else int(l_b[i])],
                            is_ordered=True,
                        )
                    )
                else:
                    ax_params.append(
                        RangeParameterConfig(
                            name=param_name,
                            parameter_type=p_type,
                            bounds=(
                                float(l_b[i]) if is_float else int(l_b[i]),
                                float(u_b[i]) if is_float else int(u_b[i]),
                            ),
                        )
                    )
        return ax_params


class AxObjectiveBuilder:
    """Builder for Ax objective configurations."""

    @staticmethod
    def build_optimization_config(
        ax_objectives: list[dict[str, Any]] | None,
        problem: OptimizationProblem,
        ax_outcome_constraints: list[OutcomeConstraint],
    ) -> OptimizationConfig:
        """Builds the OptimizationConfig (Single or Multi-Objective)."""
        # Fallback to problem if ax_objectives is None
        if not ax_objectives:
            # Check if problem has multiple objectives
            if (
                isinstance(problem.objective.name, list)
                and len(problem.objective.name) > 1
            ):
                ax_objectives = [
                    {"name": n, "minimize": True} for n in problem.objective.name
                ]
            else:
                obj_name = problem.objective.name
                if isinstance(obj_name, list):
                    obj_name = obj_name[0]
                return OptimizationConfig(
                    objective=Objective(
                        metric=MapMetric(name=obj_name),
                        minimize=True,
                    ),
                    outcome_constraints=ax_outcome_constraints,
                )

        if len(ax_objectives) == 1:
            obj = ax_objectives[0]
            return OptimizationConfig(
                objective=Objective(
                    metric=MapMetric(name=obj["name"]),
                    minimize=obj.get("minimize", True),
                ),
                outcome_constraints=ax_outcome_constraints,
            )

        # Multi-Objective
        objectives = []
        objective_thresholds = []
        for obj in ax_objectives:
            minimize = obj.get("minimize", True)
            objectives.append(
                Objective(
                    metric=MapMetric(name=obj["name"]),
                    minimize=minimize,
                )
            )
            # Default threshold if not provided: a loose upper bound or tight lower bound
            # In multi-objective, Ax typically requires objective thresholds to define the reference point
            # for hypervolume calculations.
            threshold_val = obj.get("threshold")
            if threshold_val is None:
                # If no threshold provided, AxClient will auto-infer it later,
                # but MultiObjectiveOptimizationConfig requires them to be OutcomeConstraints.
                # Actually, Ax 0.4+ allows thresholds to be optional or handled internally.
                # Let's provide a loose bound: if minimizing, upper bound is high.
                threshold_val = 1e6 if minimize else -1e6

            objective_thresholds.append(
                ObjectiveThreshold(
                    metric=MapMetric(name=obj["name"]),
                    bound=threshold_val,
                    relative=False,
                    op=ComparisonOp.LEQ if minimize else ComparisonOp.GEQ,
                )
            )

        return MultiObjectiveOptimizationConfig(
            objective=MultiObjective(objectives=objectives),
            objective_thresholds=objective_thresholds,
            outcome_constraints=ax_outcome_constraints,
        )


class AxConstraintBuilder:
    """Builder for Ax constraint configurations."""

    @staticmethod
    def build_outcome_constraints(constraints: list[Any]) -> list[OutcomeConstraint]:
        """Builds Ax OutcomeConstraints from GEMSEO constraints."""
        ax_outcome_constraints = []
        for cstr in constraints:
            if cstr.f_type == "ineq":
                ax_outcome_constraints.append(
                    OutcomeConstraint(
                        metric=MapMetric(name=cstr.name),
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

    def __init__(self, algo_name: str = "Ax_Bayesian") -> None:
        """Constructor."""
        super().__init__(algo_name=algo_name)

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

    def _seed_database(
        self, client: Client, problem: OptimizationProblem, design_space: Any
    ) -> None:
        obj_names = problem.objective.name
        if not isinstance(obj_names, list):
            obj_names = [obj_names]

        metric_names = set(obj_names) | {c.name for c in problem.constraints}
        for i, (x_hash, output) in enumerate(problem.database.items()):
            x_seed = x_hash.unwrap()
            seed_params: dict[str, int | float] = {}
            seed_offset = 0
            for var_name in design_space.variable_names:
                size = design_space.variable_sizes[var_name]
                for j in range(size):
                    param_name = f"{var_name}_{j}" if size > 1 else var_name
                    seed_params[param_name] = float(x_seed[seed_offset + j])
                seed_offset += size

            seed_results = {}
            for metric in metric_names:
                val = output.get(metric)
                if val is not None:
                    seed_results[metric] = (
                        float(np.max(val))
                        if isinstance(val, np.ndarray)
                        else float(val)
                    )

            if seed_results:
                if i == 0:
                    trial_idx = client.attach_baseline(parameters=seed_params)
                else:
                    trial_idx = client.attach_trial(parameters=seed_params)
                client.complete_trial(trial_index=trial_idx, raw_data=seed_results)

    def _run(self, problem: EvaluationProblem) -> tuple[str, int]:
        """Executes the optimization algorithm."""
        problem = cast(OptimizationProblem, problem)
        max_iter = getattr(self._settings, "max_iter", 10)
        n_init = getattr(self._settings, "n_init", 5)
        use_bonsai = getattr(self._settings, "use_bonsai", False)
        ax_objectives = getattr(self._settings, "ax_objectives", None)
        ax_parameter_constraints = getattr(
            self._settings, "ax_parameter_constraints", None
        )

        design_space = problem.design_space

        gs = self._get_generation_strategy(use_bonsai, n_init)

        client = Client()

        ax_parameters = getattr(self._settings, "ax_parameters", None)
        if ax_parameters:
            ax_params = AxParameterBuilder.build_from_ax_parameters(ax_parameters)
        else:
            normalize = getattr(self._settings, "normalize_design_space", True)
            ax_params = AxParameterBuilder.build_from_design_space(
                design_space, normalize
            )

        client.configure_experiment(
            name="gemseo_ax_opt",
            parameters=ax_params,
            parameter_constraints=ax_parameter_constraints,
        )

        ax_outcome_constraints = AxConstraintBuilder.build_outcome_constraints(
            problem.constraints
        )

        opt_config = AxObjectiveBuilder.build_optimization_config(
            ax_objectives, problem, ax_outcome_constraints
        )

        client.set_optimization_config(opt_config)

        client.set_generation_strategy(gs)

        self._seed_database(client, problem, design_space)

        total_trials = max_iter
        budget_exhausted = False

        obj_names = problem.objective.name
        if not isinstance(obj_names, list):
            obj_names = [obj_names]

        for _ in range(total_trials):
            if budget_exhausted:
                break
            trials = client.get_next_trials(max_trials=1)
            for trial_index, parameters in trials.items():
                x = np.zeros(design_space.dimension)
                offset = 0
                for var_name in design_space.variable_names:
                    size = design_space.variable_sizes[var_name]
                    for i in range(size):
                        param_name = f"{var_name}_{i}" if size > 1 else var_name
                        x[offset + i] = parameters[param_name]
                    offset += size

                try:
                    out_dict, _ = problem.evaluate_functions(
                        x, design_vector_is_normalized=False
                    )
                except MaxIterReachedException:
                    client.mark_trial_abandoned(trial_index=trial_index)
                    budget_exhausted = True
                    break
                except Exception as e:
                    import traceback

                    logger.error(
                        f"Failed to evaluate point: {e}\n{traceback.format_exc()}"
                    )
                    client.mark_trial_abandoned(trial_index=trial_index)
                    continue

                results = {}
                for obj_name in obj_names:
                    obj_val = out_dict[obj_name]
                    results[obj_name] = (
                        float(obj_val[0])
                        if isinstance(obj_val, np.ndarray)
                        else float(obj_val)
                    )

                for cstr in problem.constraints:
                    if cstr.f_type == "ineq":
                        val = out_dict[cstr.name]
                        results[cstr.name] = (
                            float(np.max(val))
                            if isinstance(val, np.ndarray)
                            else float(val)
                        )

                client.complete_trial(trial_index=trial_index, raw_data=results)

        try:
            is_moo = False
            if hasattr(
                client.experiment.optimization_config, "objective"
            ) and isinstance(
                client.experiment.optimization_config.objective, MultiObjective
            ):
                is_moo = True

            if is_moo:
                pareto_front = client.get_pareto_frontier()
                if pareto_front:
                    # In MOO we get a list of parameterizations, we just take the first one and put it to design space
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
                    param_name = f"{var_name}_{i}" if size > 1 else var_name
                    x_opt[offset + i] = best_parameters[param_name]
                offset += size

            try:
                problem.evaluate_functions(x_opt, design_vector_is_normalized=False)
            except (MaxIterReachedException, Exception):
                pass
            design_space.set_current_value(x_opt)
            return "Optimization completed successfully.", 0
        except Exception as e:
            return f"Optimization failed: {str(e)}", 1
