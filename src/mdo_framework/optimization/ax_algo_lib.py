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
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
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
    normalize_design_space: bool = False


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

    def _run(self, problem: EvaluationProblem) -> tuple[str, int]:
        """Executes the optimization algorithm."""
        problem = cast(OptimizationProblem, problem)
        max_iter = getattr(self._settings, "max_iter", 10)
        n_init = getattr(self._settings, "n_init", 5)
        use_bonsai = getattr(self._settings, "use_bonsai", False)

        design_space = problem.design_space

        # Determine client setup
        if use_bonsai:
            logger.warning("Experimental feature BONSAI algorithm is activated.")
            gs = GenerationStrategy(
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
        else:
            gs = GenerationStrategy(
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

        client = Client()

        # Map GEMSEO design space to Ax parameters
        ax_params = []
        ax_parameters = getattr(self._settings, "ax_parameters", None)
        if ax_parameters:
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
        else:
            from gemseo.algos.design_space_utils import get_value_and_bounds

            normalize = getattr(self._settings, "normalize_design_space", True)
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

        client.configure_experiment(
            name="gemseo_ax_opt",
            parameters=ax_params,
        )

        # Objective – GEMSEO always minimises internally (it negates the
        # function when the user requests maximisation), so we must tell Ax to
        # minimise as well.  The previous string-based configure_optimization()
        # defaulted to *maximize*, which caused BoTorch to explore corners
        # instead of converging to the minimum.
        obj_name = problem.objective.name

        ax_outcome_constraints = []
        for cstr in problem.constraints:
            if cstr.f_type == "ineq":
                ax_outcome_constraints.append(
                    OutcomeConstraint(
                        metric=MapMetric(name=cstr.name),
                        op=ComparisonOp.LEQ,
                        bound=0.0,
                        relative=False,
                    )
                )

        client.set_optimization_config(
            OptimizationConfig(
                objective=Objective(
                    metric=MapMetric(name=obj_name),
                    minimize=True,
                ),
                outcome_constraints=ax_outcome_constraints,
            )
        )

        client.set_generation_strategy(gs)

        # Seed Ax with points already in the GEMSEO database when _run is called.
        # _pre_run evaluates x0 before _run, so the database contains at least
        # the starting point. Attaching it as baseline keeps the Ax surrogate and
        # feasibility tracking consistent with what the GEMSEO logger reports.
        # database.items() yields (HashableNdarray, dict[metric, ndarray]) pairs.
        metric_names = {obj_name} | {c.name for c in problem.constraints}
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

        # n_init Sobol + n_steps BoTorch = total budget passed as max_iter by the caller.
        total_trials = max_iter
        budget_exhausted = False

        for _ in range(total_trials):
            if budget_exhausted:
                break
            trials = client.get_next_trials(max_trials=1)
            for trial_index, parameters in trials.items():
                # Reconstruct GEMSEO input array format
                x = np.zeros(design_space.dimension)
                offset = 0
                for var_name in design_space.variable_names:
                    size = design_space.variable_sizes[var_name]
                    for i in range(size):
                        param_name = f"{var_name}_{i}" if size > 1 else var_name
                        x[offset + i] = parameters[param_name]
                    offset += size

                # Evaluate using the OptimizationProblem
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
            best_parameters, best_obj, trial_idx, arm_name = (
                client.get_best_parameterization()
            )

            # Map best_parameters back to GEMSEO arrays and set as optimum
            x_opt = np.zeros(design_space.dimension)
            offset = 0
            for var_name in design_space.variable_names:
                size = design_space.variable_sizes[var_name]
                for i in range(size):
                    param_name = f"{var_name}_{i}" if size > 1 else var_name
                    x_opt[offset + i] = best_parameters[param_name]
                offset += size

            # Register the optimum on the design space.
            # The budget may already be exhausted, so skip re-evaluation.
            try:
                problem.evaluate_functions(x_opt, design_vector_is_normalized=False)
            except (MaxIterReachedException, Exception):
                pass
            design_space.set_current_value(x_opt)
            return "Optimization completed successfully.", 0
        except Exception as e:
            return f"Optimization failed: {str(e)}", 1
