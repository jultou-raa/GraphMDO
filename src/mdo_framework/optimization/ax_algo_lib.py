"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import logging
import traceback
import warnings
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple, TypedDict, cast

import numpy as np
from ax.adapter.registry import Generators as Models
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from ax.core.map_metric import MapMetric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.generation_strategy.generation_node import GenerationStep
from ax.generation_strategy.generation_strategy import GenerationStrategy
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.evaluation_problem import EvaluationProblem
from gemseo.algos.opt.base_optimization_library import (
    BaseOptimizationLibrary,
    OptimizationAlgorithmDescription,
)
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.stop_criteria import MaxIterReachedException

logger = logging.getLogger(__name__)

# Exceptions that indicate a recoverable evaluation failure.
# Unknown exceptions (e.g., MemoryError, SystemExit) should still propagate.
_RECOVERABLE_EVAL_ERRORS = (ValueError, RuntimeError, ArithmeticError)


class _AxParameterDictBase(TypedDict):
    name: str
    type: str


class AxParameterDict(_AxParameterDictBase, total=False):
    bounds: list[float]
    values: list[Any]
    is_ordered: bool
    value_type: str


class _AxObjectiveDictBase(TypedDict):
    name: str


class AxObjectiveDict(_AxObjectiveDictBase, total=False):
    minimize: bool
    threshold: float


def _get_param_name(var_name: str, i: int, size: int) -> str:
    """Constructs the Ax parameter name for array elements."""
    return f"{var_name}_{i}" if size > 1 else var_name


def _normalize_name_list(names: str | list[str]) -> list[str]:
    """Ensures an objective name or list of names is always a list."""
    return names if isinstance(names, list) else [names]


def _get_expected_parameter_names(design_space: DesignSpace) -> list[str]:
    """Builds the parameter names expected by the design-space mapping."""
    parameter_names = []
    for var_name in design_space.variable_names:
        size = design_space.variable_sizes[var_name]
        for index in range(size):
            parameter_names.append(_get_param_name(var_name, index, size))
    return parameter_names


def _get_range_parameter_type(parameter: AxParameterDict) -> Literal["float", "int"]:
    """Validates the range parameter type accepted by Ax."""
    parameter_type = parameter.get("value_type", "float")
    if parameter_type not in {"float", "int"}:
        raise ValueError(
            f"Range parameter {parameter['name']} requires value_type 'float' or 'int'."
        )
    return cast(Literal["float", "int"], parameter_type)


def _infer_choice_parameter_type(
    values: list[Any],
) -> Literal["float", "int", "str", "bool"]:
    """Infers the Ax choice parameter type from the first value."""
    first_value = values[0]
    if isinstance(first_value, bool):
        return "bool"
    if isinstance(first_value, str):
        return "str"
    if isinstance(first_value, int):
        return "int"
    return "float"


def _get_choice_parameter_type(
    parameter: AxParameterDict,
    values: list[Any],
) -> Literal["float", "int", "str", "bool"]:
    """Validates the choice parameter type accepted by Ax."""
    parameter_type = parameter.get("value_type", _infer_choice_parameter_type(values))
    if parameter_type not in {"float", "int", "str", "bool"}:
        raise ValueError(
            f"Choice parameter {parameter['name']} requires value_type 'float', 'int', 'str' or 'bool'."
        )
    return cast(Literal["float", "int", "str", "bool"], parameter_type)


def _validate_custom_ax_parameters(
    ax_parameters: list[AxParameterDict],
    design_space: DesignSpace,
    normalize: bool,
) -> None:
    """Rejects custom parameter layouts that the design-space mapper cannot honor."""
    if normalize:
        raise ValueError(
            "normalize_design_space=True is only supported when Ax parameters are "
            "derived from the GEMSEO design space."
        )

    actual_names = [parameter["name"] for parameter in ax_parameters]
    duplicate_names = sorted(
        {name for name in actual_names if actual_names.count(name) > 1}
    )
    if duplicate_names:
        raise ValueError(
            "Duplicate Ax parameter names are not supported: "
            f"{', '.join(duplicate_names)}."
        )

    expected_names = set(_get_expected_parameter_names(design_space))
    actual_name_set = set(actual_names)
    missing_names = sorted(expected_names - actual_name_set)
    extra_names = sorted(actual_name_set - expected_names)
    if missing_names or extra_names:
        details = []
        if missing_names:
            details.append(f"missing {missing_names}")
        if extra_names:
            details.append(f"unexpected {extra_names}")
        raise ValueError(
            "Custom ax_parameters must match the GEMSEO design-space parameter names; "
            + "; ".join(details)
            + "."
        )


def _build_design_vector(
    parameters: Mapping[str, Any],
    design_space: DesignSpace,
    normalize: bool,
) -> np.ndarray:
    """Maps Ax parameters back to a physical GEMSEO design vector."""
    design_vector = np.zeros(design_space.dimension)
    offset = 0
    for var_name in design_space.variable_names:
        size = design_space.variable_sizes[var_name]
        for index in range(size):
            param_name = _get_param_name(var_name, index, size)
            design_vector[offset + index] = float(parameters[param_name])
        offset += size
    if normalize:
        return design_space.unnormalize_vect(design_vector)
    return design_vector


def _require_ax_settings(settings: BaseOptimizerSettings | None) -> "AxSettings":
    """Returns Ax settings or fails fast with an explicit runtime error."""
    if not isinstance(settings, AxSettings):
        raise TypeError(
            "AxOptimizationLibrary expects AxSettings in self._settings, "
            f"got {type(settings).__name__}."
        )
    return settings


class _ConfiguredClient(NamedTuple):
    """Bundles a configured Ax Client with derived optimization metadata."""

    client: Client
    is_moo: bool


class AxSettings(BaseOptimizerSettings):
    """Settings for Ax optimization."""

    max_iter: int = 10
    batch_size: int = 1
    n_init: int = 5
    use_bonsai: bool = False
    ax_parameters: list[AxParameterDict] | None = None
    ax_objectives: list[AxObjectiveDict] | None = None
    ax_parameter_constraints: list[str] | None = None
    normalize_design_space: bool = False


def build_from_ax_parameters(
    ax_parameters: list[AxParameterDict],
) -> list[RangeParameterConfig | ChoiceParameterConfig]:
    ax_params = []
    for p in ax_parameters:
        if p["type"] == "range":
            bounds = p.get("bounds")
            if bounds is None or len(bounds) != 2:
                raise ValueError(
                    f"Range parameter {p['name']} requires exactly 2 bounds."
                )
            ax_params.append(
                RangeParameterConfig(
                    name=p["name"],
                    bounds=(bounds[0], bounds[1]),
                    parameter_type=_get_range_parameter_type(p),
                )
            )
        elif p["type"] == "choice":
            values = p.get("values")
            if not values:
                raise ValueError(
                    f"Choice parameter {p['name']} requires a list of values."
                )
            ax_params.append(
                ChoiceParameterConfig(
                    name=p["name"],
                    values=values,
                    parameter_type=_get_choice_parameter_type(p, values),
                    is_ordered=p.get("is_ordered"),
                )
            )
        else:
            raise ValueError(
                f"Unsupported Ax parameter type for {p['name']}: {p['type']}."
            )
    return ax_params


def build_from_design_space(
    design_space: DesignSpace, normalize: bool
) -> list[RangeParameterConfig]:
    if normalize:
        lb_full = np.concatenate(
            [design_space.get_lower_bound(v) for v in design_space.variable_names]
        )
        ub_full = np.concatenate(
            [design_space.get_upper_bound(v) for v in design_space.variable_names]
        )
        lb_normalized = design_space.normalize_vect(lb_full)
        ub_normalized = design_space.normalize_vect(ub_full)
        offset = 0
        bounds_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for var_name in design_space.variable_names:
            size = design_space.variable_sizes[var_name]
            bounds_map[var_name] = (
                lb_normalized[offset : offset + size],
                ub_normalized[offset : offset + size],
            )
            offset += size
    else:
        bounds_map = {
            v: (design_space.get_lower_bound(v), design_space.get_upper_bound(v))
            for v in design_space.variable_names
        }

    ax_params = []
    for var_name in design_space.variable_names:
        size = design_space.variable_sizes[var_name]
        l_b, u_b = bounds_map[var_name]
        var_type_list = design_space.variable_types.get(var_name, [])
        for i in range(size):
            param_name = _get_param_name(var_name, i, size)
            is_int = i < len(var_type_list) and var_type_list[i] is int
            # Normalized space is always float; integers are preserved only in physical space.
            p_type = "int" if (is_int and not normalize) else "float"
            cast_fn = int if p_type == "int" else float
            ax_params.append(
                RangeParameterConfig(
                    name=param_name,
                    bounds=(cast_fn(l_b[i]), cast_fn(u_b[i])),
                    parameter_type=p_type,
                )
            )
    return ax_params


def _build_moo_config(
    ax_objectives: list[AxObjectiveDict],
    problem: OptimizationProblem,
    ax_outcome_constraints: list[OutcomeConstraint],
) -> MultiObjectiveOptimizationConfig:
    ax_objs = []
    objective_thresholds = []
    for obj in ax_objectives:
        metric_name = obj["name"]
        minimize = obj.get("minimize", problem.minimize_objective)
        threshold = obj.get("threshold", None)
        ax_objs.append(Objective(metric=MapMetric(name=metric_name), minimize=minimize))
        if threshold is not None:
            op = ComparisonOp.LEQ if minimize else ComparisonOp.GEQ
            objective_thresholds.append(
                ObjectiveThreshold(
                    metric=MapMetric(name=metric_name),
                    bound=float(threshold),
                    relative=False,
                    op=op,  # type: ignore[arg-type]  # Pyright widens IntEnum to int
                )
            )
    return MultiObjectiveOptimizationConfig(
        objective=MultiObjective(objectives=ax_objs),
        objective_thresholds=objective_thresholds if objective_thresholds else None,
        outcome_constraints=ax_outcome_constraints,
    )


def _build_soo_config(
    ax_objectives: list[AxObjectiveDict] | None,
    problem: OptimizationProblem,
    ax_outcome_constraints: list[OutcomeConstraint],
) -> OptimizationConfig:
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
        objective=Objective(metric=MapMetric(name=metric_name), minimize=minimize),
        outcome_constraints=ax_outcome_constraints,
    )


def build_optimization_config(
    ax_objectives: list[AxObjectiveDict] | None,
    problem: OptimizationProblem,
    ax_outcome_constraints: list[OutcomeConstraint],
) -> OptimizationConfig | MultiObjectiveOptimizationConfig:
    if ax_objectives and len(ax_objectives) > 1:
        return _build_moo_config(ax_objectives, problem, ax_outcome_constraints)
    return _build_soo_config(ax_objectives, problem, ax_outcome_constraints)


def build_outcome_constraints(
    constraints: list[Any],
) -> list[OutcomeConstraint]:
    ax_outcome_constraints = []
    for c in constraints:
        if c.f_type == "ineq":
            ax_outcome_constraints.append(
                OutcomeConstraint(
                    metric=MapMetric(name=c.name),
                    op=ComparisonOp.LEQ,  # type: ignore[arg-type]  # Pyright widens IntEnum to int
                    bound=0.0,
                    relative=False,
                )
            )
            continue
        raise ValueError(
            f"Unsupported constraint type for {c.name}: {c.f_type}. Only 'ineq' is supported."
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

    def __init__(self, algo_name: str = "Ax_Bayesian", client_factory=None) -> None:
        super().__init__(algo_name=algo_name)
        self.client_factory = client_factory or Client
        self.trial_history: list[dict[str, dict[str, Any]]] = []

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
    def _extract_seed_params(
        x_seed: np.ndarray, design_space: DesignSpace
    ) -> dict[str, float]:
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
        self,
        client: Client,
        problem: OptimizationProblem,
        design_space: DesignSpace,
        normalize: bool,
    ) -> None:
        obj_names = _normalize_name_list(problem.objective.name)

        c_names = {c.name for c in problem.constraints}
        metric_names = set(obj_names) | c_names

        for i, (x_hash, output) in enumerate(problem.database.items()):
            x_seed = x_hash.unwrap()
            if normalize:
                x_seed = design_space.normalize_vect(x_seed)
            seed_params = self._extract_seed_params(x_seed, design_space)
            seed_results = self._extract_seed_results(output, metric_names, c_names)

            if seed_results:
                if i == 0:
                    trial_idx = client.attach_baseline(parameters=seed_params)
                else:
                    trial_idx = client.attach_trial(parameters=seed_params)
                client.complete_trial(trial_index=trial_idx, raw_data=seed_results)
                self.trial_history.append(
                    {
                        "parameters": dict(seed_params),
                        "objectives": dict(seed_results),
                    }
                )

    def _configure_client(self, problem: OptimizationProblem) -> _ConfiguredClient:
        settings = _require_ax_settings(self._settings)

        design_space = problem.design_space
        gs = self._get_generation_strategy(settings.use_bonsai, settings.n_init)

        client = self.client_factory()

        if settings.ax_parameters:
            _validate_custom_ax_parameters(
                settings.ax_parameters,
                design_space,
                settings.normalize_design_space,
            )
            ax_params = build_from_ax_parameters(settings.ax_parameters)
        else:
            ax_params = build_from_design_space(
                design_space, settings.normalize_design_space
            )

        client.configure_experiment(
            name="gemseo_ax_opt",
            parameters=ax_params,
            parameter_constraints=settings.ax_parameter_constraints,
        )

        ax_outcome_constraints = build_outcome_constraints(list(problem.constraints))

        opt_config = build_optimization_config(
            settings.ax_objectives, problem, ax_outcome_constraints
        )

        client.set_optimization_config(opt_config)
        client.set_generation_strategy(gs)
        # Derive MOO flag from the optimization config type we built, to avoid
        # accessing the Client's private _experiment attribute.
        is_moo = isinstance(opt_config, MultiObjectiveOptimizationConfig)
        return _ConfiguredClient(client=client, is_moo=is_moo)

    def _execute_trial(
        self,
        client: Client,
        problem: OptimizationProblem,
        trial_index: int,
        parameters: Mapping[str, Any],
        metric_names: set[str],
        normalize: bool = False,
    ) -> bool:
        design_space = problem.design_space
        x = _build_design_vector(parameters, design_space, normalize)

        try:
            out_dict, _ = problem.evaluate_functions(
                x, design_vector_is_normalized=False
            )
        except MaxIterReachedException:
            client.mark_trial_abandoned(trial_index=trial_index)
            return True
        except _RECOVERABLE_EVAL_ERRORS as e:
            logger.error("Failed to evaluate point: %s\n%s", e, traceback.format_exc())
            client.mark_trial_abandoned(trial_index=trial_index)
            return False

        res = {}
        for metric in metric_names:
            if metric in out_dict:
                val = out_dict[metric]
                if isinstance(val, (int, float)):
                    res[metric] = float(val)
                elif isinstance(val, (list, np.ndarray)):
                    if len(val) == 1:
                        res[metric] = float(val[0])
                    else:
                        logger.warning(
                            "Multi-dimensional output for metric '%s' (size=%d); "
                            "aggregating with np.max.",
                            metric,
                            len(val),
                        )
                        res[metric] = float(np.max(val))

        if not res:
            logger.error(
                "No metric data extracted for trial %d; abandoning. "
                "Check that objective/constraint names match the function output keys.",
                trial_index,
            )
            client.mark_trial_abandoned(trial_index=trial_index)
            return False

        client.complete_trial(trial_index=trial_index, raw_data=res)
        trial_parameters = dict(parameters)
        self.trial_history.append(
            {
                "parameters": trial_parameters,
                "objectives": dict(res),
            }
        )
        return False

    def _extract_best_solution(
        self,
        client: Client,
        problem: OptimizationProblem,
        is_moo: bool,
        normalize: bool = False,
    ) -> None:
        design_space = problem.design_space

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
            best_parameters, *_ = client.get_best_parameterization()

        x_opt = _build_design_vector(best_parameters, design_space, normalize)

        try:
            problem.evaluate_functions(x_opt, design_vector_is_normalized=False)
        except MaxIterReachedException:
            pass
        except _RECOVERABLE_EVAL_ERRORS:
            logger.warning(
                "Could not evaluate the optimal point:\n%s", traceback.format_exc()
            )
        design_space.set_current_value(x_opt)

    def _record_last_point(self, problem: OptimizationProblem) -> None:
        """Appends the latest GEMSEO point when Ax produces no executable trial."""
        try:
            last_point = problem.history.last_point
        except ValueError:
            return

        design_space = problem.design_space
        parameters: dict[str, Any] = {}
        offset = 0
        for var_name in design_space.variable_names:
            size = design_space.variable_sizes[var_name]
            values = last_point.design[offset : offset + size]
            parameters[var_name] = values[0] if size == 1 else values.tolist()
            offset += size

        objectives = {}
        objective_names = _normalize_name_list(problem.objective.name)
        objective_values = np.atleast_1d(last_point.objective).flatten()
        for index, objective_name in enumerate(objective_names):
            if index < len(objective_values):
                objectives[objective_name] = float(objective_values[index])

        for constraint_name, constraint_value in last_point.constraints.items():
            objectives[constraint_name] = float(constraint_value)

        self.trial_history.append(
            {
                "parameters": parameters,
                "objectives": objectives,
            }
        )

    def _run(self, problem: EvaluationProblem) -> tuple[str, int]:
        """Executes the optimization algorithm."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"ax\.")
            self.trial_history = []

            if not isinstance(problem, OptimizationProblem):
                raise TypeError(
                    f"Expected OptimizationProblem, got {type(problem).__name__}"
                )

            settings = _require_ax_settings(self._settings)

            configured = self._configure_client(problem)
            self._seed_database(
                configured.client,
                problem,
                problem.design_space,
                normalize=settings.normalize_design_space,
            )

            budget_exhausted = False

            obj_names = _normalize_name_list(problem.objective.name)
            c_names = {c.name for c in problem.constraints}
            metric_names = set(obj_names) | c_names

            for _ in range(settings.max_iter):
                if budget_exhausted:
                    break
                executed_trial = False
                trials = configured.client.get_next_trials(
                    max_trials=settings.batch_size
                )
                for trial_index, parameters in trials.items():
                    executed_trial = True
                    budget_exhausted = self._execute_trial(
                        configured.client,
                        problem,
                        trial_index,
                        parameters,
                        metric_names,
                        normalize=settings.normalize_design_space,
                    )
                    if budget_exhausted:
                        break
                if not executed_trial:
                    self._record_last_point(problem)

            self._extract_best_solution(
                configured.client,
                problem,
                configured.is_moo,
                normalize=settings.normalize_design_space,
            )

        if budget_exhausted:
            return "Optimization stopped early: evaluation budget exhausted.", 0
        return "Optimization completed successfully.", 0
