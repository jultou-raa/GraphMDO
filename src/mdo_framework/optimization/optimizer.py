"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import logging
import warnings
from typing import Any, Protocol, TypeAlias

import httpx
import numpy as np
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import Discipline
from gemseo.typing import StrKeyMapping

from mdo_framework.core.errors import EvaluationError, evaluation_error_from_payload
from mdo_framework.optimization.ax_algo_lib import AxObjectiveDict
from mdo_framework.optimization.parameter_codec import (
    ParameterDefinitionError,
    ParameterValueError,
    build_parameter_lookup,
)
from mdo_framework.optimization.parameter_codec import (
    coerce_scalar as _shared_coerce_scalar,
)
from mdo_framework.optimization.parameter_codec import (
    decode_parameter_value as _shared_decode_parameter_value,
)

logger = logging.getLogger(__name__)

ScalarValue: TypeAlias = bool | int | float | str
AX_OBJECTIVE_KEYS = frozenset(
    AxObjectiveDict.__required_keys__ | AxObjectiveDict.__optional_keys__
)


class OptimizationConfigurationError(ValueError):
    """Raised when the optimization request is invalid for the current backend."""


class OptimizationExecutionError(RuntimeError):
    """Raised when optimization cannot produce a valid result."""


class RemoteEvaluationTransportError(RuntimeError):
    """Raised when the execution service cannot be reached reliably."""


class RemoteEvaluationContractError(TypeError):
    """Raised when the execution service response breaks the expected contract."""


def _validate_objectives(objectives: list[dict[str, Any]]) -> None:
    """Rejects objective keys the Ax backend does not understand."""
    if not objectives:
        raise OptimizationConfigurationError("At least one objective is required.")
    for objective in objectives:
        if "name" not in objective:
            raise OptimizationConfigurationError(
                f"Objective {objective!r} is missing the required 'name' key."
            )
        if unknown := set(objective) - AX_OBJECTIVE_KEYS:
            raise OptimizationConfigurationError(
                f"Objective {objective['name']!r} has unsupported keys: "
                f"{sorted(unknown)}. Supported keys: {sorted(AX_OBJECTIVE_KEYS)}."
            )


def _get_optimization_history(
    scenario: Any | None, algo: Any | None = None
) -> list[dict[str, dict[str, Any]]]:
    """Returns explicit Ax trial history when available."""
    trial_history = getattr(algo, "trial_history", None)
    if trial_history is not None:
        return trial_history
    return []


def _coerce_scalar(value: Any) -> Any:
    return _shared_coerce_scalar(value)


def _decode_parameter_value(parameter: dict[str, Any], raw_value: Any) -> ScalarValue:
    """Decode a GEMSEO design-space value to the user-facing parameter value."""
    try:
        return _shared_decode_parameter_value(parameter, raw_value)
    except ParameterDefinitionError as exc:
        raise OptimizationConfigurationError(str(exc)) from exc
    except ParameterValueError as exc:
        raise OptimizationExecutionError(str(exc)) from exc


def _build_design_space(parameters: list[dict[str, Any]]) -> DesignSpace:
    """Build a GEMSEO design space with explicit integer encoding for choices."""
    design_space = DesignSpace()
    for parameter in parameters:
        parameter_name = parameter["name"]
        if parameter["type"] == "range":
            bounds = parameter.get("bounds")
            if bounds is None or len(bounds) != 2:
                raise OptimizationConfigurationError(
                    f"Range parameter {parameter_name} requires exactly two bounds."
                )
            extra_args = {}
            if parameter.get("value_type") == "int":
                extra_args["type_"] = "integer"
            design_space.add_variable(
                parameter_name,
                lower_bound=bounds[0],
                upper_bound=bounds[1],
                **extra_args,
            )
            continue

        if parameter["type"] != "choice":
            raise OptimizationConfigurationError(
                f"Unsupported parameter type for {parameter_name}: {parameter['type']}."
            )

        choices = parameter.get("values", [])
        if not choices:
            raise OptimizationConfigurationError(
                f"Choice parameter {parameter_name} requires at least one value."
            )
        design_space.add_variable(
            parameter_name,
            value=0,
            lower_bound=0,
            upper_bound=max(len(choices) - 1, 0),
            type_="integer",
        )

    return design_space


def _add_constraints_to_scenario(
    scenario: Any, constraints: list[dict[str, Any]]
) -> None:
    """Normalize user constraints to GEMSEO inequality constraints."""
    for constraint in constraints:
        operator = constraint["op"]
        if operator not in {"<=", ">="}:
            raise OptimizationConfigurationError(
                f"Unsupported constraint operator {operator!r} for {constraint['name']}."
            )
        scenario.add_constraint(
            constraint["name"],
            constraint_type="ineq",
            value=float(constraint["bound"]),
            positive=operator == ">=",
        )


def _extract_best_parameters(
    optimum: Any,
    design_space: DesignSpace,
    parameters: list[dict[str, Any]],
) -> dict[str, ScalarValue]:
    best_parameters: dict[str, ScalarValue] = {}
    offset = 0
    for parameter in parameters:
        name = parameter["name"]
        size = design_space.variable_sizes[name]
        raw_value = optimum.design[offset : offset + size]
        best_parameters[name] = _decode_parameter_value(
            parameter,
            raw_value[0] if size == 1 else raw_value.tolist(),
        )
        offset += size
    return best_parameters


def _extract_best_objectives(
    optimum: Any,
    objective_names: list[str],
    fallback_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    try:
        objective_values = np.atleast_1d(optimum.objective).flatten()
    except Exception:
        objective_values = np.array([])

    extracted = {
        objective_name: float(objective_values[index])
        for index, objective_name in enumerate(objective_names)
        if index < objective_values.size
    }
    if len(extracted) == len(objective_names):
        return extracted

    if fallback_metrics:
        merged = dict(extracted)
        for objective_name in objective_names:
            if objective_name in fallback_metrics:
                merged[objective_name] = float(fallback_metrics[objective_name])
        if len(merged) == len(objective_names):
            return merged

    raise OptimizationExecutionError(
        "Optimization completed but GEMSEO optimum does not expose all objectives."
    )


class Evaluator(Protocol):
    def evaluate(
        self,
        parameters: dict[str, Any],
        objectives: list[str],
    ) -> dict[str, float]:
        """Evaluates the requested objectives given the design parameters."""
        ...


def _response_detail(response: httpx.Response) -> Any:
    """Returns the ``detail`` of a JSON error response, ``None`` otherwise."""
    try:
        body = response.json()
    except ValueError:
        return None
    return body.get("detail") if isinstance(body, dict) else None


class RemoteEvaluator:
    """Evaluates the design parameters remotely by communicating with the Execution microservice.

    Args:
        service_url: The URL of the execution service.

    """

    def __init__(
        self,
        service_url: str,
        client: httpx.Client | None = None,
        timeout: httpx.Timeout | None = None,
    ):
        self.service_url = service_url.rstrip("/")
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=self.service_url,
            timeout=timeout or httpx.Timeout(30.0, connect=5.0),
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def evaluate(
        self,
        parameters: dict[str, Any],
        objectives: list[str],
    ) -> dict[str, float]:
        payload = {
            "inputs": parameters,
            "objectives": objectives,
        }
        try:
            response = self.client.post(f"{self.service_url}/evaluate", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RemoteEvaluationTransportError(
                "Execution service request timed out."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise RemoteEvaluationTransportError(
                    f"Execution service returned HTTP {exc.response.status_code}."
                ) from exc
            detail = _response_detail(exc.response)
            if (error := evaluation_error_from_payload(detail)) is not None:
                raise error from exc
            raise RemoteEvaluationContractError(
                "Execution service rejected the evaluation request with "
                f"HTTP {exc.response.status_code}"
                + (f": {detail}" if detail is not None else ".")
            ) from exc
        except httpx.RequestError as exc:
            raise RemoteEvaluationTransportError(
                "Execution service request failed."
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RemoteEvaluationContractError(
                "Execution service returned invalid JSON."
            ) from exc

        results = data.get("results")
        if not isinstance(results, dict):
            raise RemoteEvaluationContractError(
                "Execution service response is missing a 'results' object."
            )

        missing_objectives = [name for name in objectives if name not in results]
        if missing_objectives:
            raise RemoteEvaluationContractError(
                "Execution service response is missing requested objectives: "
                + ", ".join(missing_objectives)
                + "."
            )

        normalized_results: dict[str, float] = {}
        for objective_name in objectives:
            try:
                normalized_results[objective_name] = float(results[objective_name])
            except (TypeError, ValueError) as exc:
                raise RemoteEvaluationContractError(
                    f"Execution service returned a non-numeric value for {objective_name}."
                ) from exc
        return normalized_results


class RemoteDiscipline(Discipline):
    def __init__(
        self,
        evaluator: Evaluator,
        inputs: list[dict[str, Any]] | list[str],
        outputs: list[str],
    ):
        super().__init__(name="RemoteExecution")
        self.evaluator = evaluator
        if inputs and isinstance(inputs[0], str):
            self.input_names = list(inputs)
            self.parameter_definitions = build_parameter_lookup(
                [
                    {"name": name, "type": "range", "value_type": "float"}
                    for name in self.input_names
                ]
            )
        else:
            self.parameter_definitions = build_parameter_lookup(inputs)
            self.input_names = [parameter["name"] for parameter in inputs]
        self.output_names = outputs
        self.input_grammar.update_from_names(self.input_names)
        self.output_grammar.update_from_names(self.output_names)
        for in_name in self.input_names:
            default_value = (
                0
                if self.parameter_definitions[in_name].get("type") == "choice"
                else 0.0
            )
            self.default_input_data[in_name] = np.array([default_value])

    def _run(self, input_data: dict[str, np.ndarray]) -> None:
        params = {
            parameter_name: _decode_parameter_value(
                self.parameter_definitions[parameter_name],
                parameter_value.tolist()[0]
                if parameter_value.size == 1
                else parameter_value.tolist(),
            )
            for parameter_name, parameter_value in input_data.items()
        }
        results = self.evaluator.evaluate(params, self.output_names)
        for k, v in results.items():
            self.local_data[k] = np.atleast_1d(v)


class _FailureRecorder(Discipline):
    """Runs a discipline and keeps the evaluation errors it raises.

    GEMSEO's DOE skips a sample whose evaluation raises a ``ValueError`` and
    only logs it, so the typed error would otherwise be lost.
    """

    def __init__(self, discipline: Discipline) -> None:
        super().__init__(name=discipline.name)
        self._discipline = discipline
        self.failures: list[EvaluationError] = []
        self.input_grammar.update_from_names(discipline.input_grammar.names)
        self.output_grammar.update_from_names(discipline.output_grammar.names)
        self.default_input_data.update(discipline.default_input_data)

    def _run(self, input_data: StrKeyMapping) -> dict[str, Any]:
        try:
            output_data = self._discipline.execute(input_data)
        except EvaluationError as error:
            self.failures.append(error)
            raise
        return {name: output_data[name] for name in self.output_grammar}


class BayesianOptimizer:
    """Bayesian Optimizer using Ax Platform.

    Args:
        evaluator: Local or Remote implementation of Evaluator protocol.
        parameters: Dict defining the variables bounds, choices, and types.
        objectives: Dict defining the targeted metrics and their directions.
        constraints: Dict defining boundaries mapped out of GEMSEO evaluations.
        fidelity_parameter: Name of variable designating multi-fidelity.
        use_bonsai: Toggle for experimental algorithmic execution.
        parameter_constraints: List of string-based constraints on the search space parameters.

    """

    def __init__(
        self,
        evaluator: Evaluator,
        parameters: list[dict[str, Any]],
        objectives: list[dict[str, Any]],
        constraints: list[dict[str, Any]] | None = None,
        fidelity_parameter: str | None = None,
        use_bonsai: bool = False,
        parameter_constraints: list[str] | None = None,
    ) -> None:
        _validate_objectives(objectives)
        self.evaluator = evaluator
        self.parameters = parameters
        self.objectives = objectives
        self.constraints = constraints or []
        self.fidelity_parameter = fidelity_parameter
        self.use_bonsai = use_bonsai
        self.parameter_constraints = parameter_constraints

    def _build_output_names(self) -> list[str]:
        return [objective["name"] for objective in self.objectives] + [
            constraint["name"] for constraint in self.constraints
        ]

    def _build_discipline(self) -> Discipline:
        if hasattr(self.evaluator, "problem"):
            return self.evaluator.problem
        return RemoteDiscipline(
            self.evaluator,
            self.parameters,
            self._build_output_names(),
        )

    def _prepare_scenario_context(self) -> tuple[Discipline, DesignSpace, list[str]]:
        return (
            self._build_discipline(),
            _build_design_space(self.parameters),
            [objective["name"] for objective in self.objectives],
        )

    def _create_scenario(
        self,
        *,
        discipline: Discipline,
        design_space: DesignSpace,
        objective_names: list[str],
        scenario_type: str | None = None,
        maximize_objective: bool | list[bool] | None = None,
        name: str | None = None,
    ) -> Any:
        scenario_kwargs: dict[str, Any] = {
            "formulation_name": "MDF",
            "objective_name": objective_names,
            "design_space": design_space,
        }
        if scenario_type is not None:
            scenario_kwargs["scenario_type"] = scenario_type
        if maximize_objective is not None:
            scenario_kwargs["maximize_objective"] = maximize_objective
        if name is not None:
            scenario_kwargs["name"] = name

        scenario = create_scenario([discipline], **scenario_kwargs)
        _add_constraints_to_scenario(scenario, self.constraints)
        return scenario

    def explore(self, n_samples: int = 10, n_processes: int = 1) -> dict[str, Any]:
        """Runs a Design of Experiments (DOE) exploration using GEMSEO DOEScenario.

        Args:
            n_samples: Number of samples to evaluate.
            n_processes: Number of concurrent processes.

        Returns:
            A dictionary containing the exploration history.
        """
        discipline, design_space, objective_names = self._prepare_scenario_context()
        recorder = _FailureRecorder(discipline)

        scenario = self._create_scenario(
            discipline=recorder,
            design_space=design_space,
            objective_names=objective_names,
            scenario_type="DOE",
        )

        try:
            scenario.execute(
                algo_name="Sobol",
                n_samples=n_samples,
                n_processes=n_processes,
            )

            # Post-process
            try:
                from gemseo.settings.post import ScatterPlotMatrix_Settings

                scenario.post_process(
                    "ScatterPlotMatrix",
                    settings_model=ScatterPlotMatrix_Settings(save=True, show=False),
                )
            except Exception as pp_err:
                logger.warning(f"Failed to post-process DOE: {pp_err}")

            return {
                "history": scenario.to_dataset(),
            }
        except (
            OptimizationConfigurationError,
            OptimizationExecutionError,
            RemoteEvaluationContractError,
            RemoteEvaluationTransportError,
        ):
            raise
        except Exception as e:
            evaluated = len(scenario.formulation.optimization_problem.database)
            if recorder.failures and not evaluated:
                raise recorder.failures[0] from e
            logger.error(f"Exploration failed: {e}")
            raise OptimizationExecutionError(f"Exploration failed: {str(e)}") from e

    def optimize(self, n_steps: int = 10, n_init: int = 5) -> dict[str, Any]:
        """Runs Bayesian optimization using a GEMSEO MDOScenario.

        Args:
            n_steps: Bayesian (BoTorch) iterations, at least 1.
            n_init: Initial Sobol trials, at least 1. The start point x0 is
                evaluated in addition to these trials.

        The tools are called at most ``1 + n_init + n_steps`` times. Fewer calls
        happen only when Ax stops proposing new designs, e.g. in an exhausted
        discrete space.

        Raises:
            OptimizationConfigurationError: If ``n_steps`` or ``n_init`` is < 1.
        """
        for budget_name, budget in (("n_steps", n_steps), ("n_init", n_init)):
            if budget < 1:
                raise OptimizationConfigurationError(
                    f"{budget_name} must be >= 1, got {budget}."
                )
        if self.fidelity_parameter is not None:
            warnings.warn("fidelity_parameter is ignored.")
        discipline, design_space, objective_names = self._prepare_scenario_context()

        # Explicitly configure maximize_objective per user request.
        # GEMSEO maximize_objective expects a single boolean or a list of booleans
        maximize_objective = [not o.get("minimize", True) for o in self.objectives]
        if len(maximize_objective) == 1:
            maximize_objective = maximize_objective[0]

        scenario = self._create_scenario(
            discipline=discipline,
            design_space=design_space,
            objective_names=objective_names,
            maximize_objective=maximize_objective,
            name="MDOScenario_Ax",
        )

        algo = None
        try:
            from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary

            problem = scenario.formulation.optimization_problem

            algo = AxOptimizationLibrary()
            algo.execute(
                problem,
                max_iter=1 + n_init + n_steps,
                n_init=n_init,
                use_bonsai=self.use_bonsai,
                ax_parameters=self.parameters,
                ax_objectives=self.objectives,
                ax_parameter_constraints=self.parameter_constraints,
            )

            # Generate XDSM diagram
            scenario.xdsmize(show_html=False)
            # Generate Post-Processing
            try:
                from gemseo.settings.post import OptHistoryView_Settings

                scenario.post_process(
                    settings_model=OptHistoryView_Settings(save=True, show=False)
                )
            except Exception as pp_err:
                logger.warning(f"Failed to post-process: {pp_err}")

            optimum = problem.optimum
            if optimum is None:
                raise OptimizationExecutionError(
                    "Optimization completed without a valid GEMSEO optimum."
                )

            best_params = _extract_best_parameters(
                optimum,
                design_space,
                self.parameters,
            )
            best_objectives = _extract_best_objectives(
                optimum,
                objective_names,
                getattr(algo, "best_objectives", None),
            )

            return {
                "best_parameters": best_params,
                "best_objectives": best_objectives,
                "history": _get_optimization_history(scenario, algo),
            }
        except (
            OptimizationConfigurationError,
            OptimizationExecutionError,
            RemoteEvaluationContractError,
            RemoteEvaluationTransportError,
        ):
            raise
        except Exception as e:
            import traceback

            logger.error(f"Optimization failed: {e}\n{traceback.format_exc()}")
            raise OptimizationExecutionError(f"Optimization failed: {str(e)}") from e
