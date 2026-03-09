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

import mdo_framework.optimization.ax_algo_lib  # noqa: F401

logger = logging.getLogger(__name__)

ScalarValue: TypeAlias = bool | int | float | str


class OptimizationConfigurationError(ValueError):
    """Raised when the optimization request is invalid for the current backend."""


class OptimizationExecutionError(RuntimeError):
    """Raised when optimization cannot produce a valid result."""


class RemoteEvaluationTransportError(RuntimeError):
    """Raised when the execution service cannot be reached reliably."""


class RemoteEvaluationContractError(TypeError):
    """Raised when the execution service response breaks the expected contract."""


def _get_optimization_history(
    scenario: Any | None, algo: Any | None = None
) -> list[dict[str, dict[str, Any]]]:
    """Returns explicit Ax trial history when available."""
    trial_history = getattr(algo, "trial_history", None)
    if trial_history is not None:
        return trial_history
    return []


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_parameter_value(parameter: dict[str, Any], raw_value: Any) -> ScalarValue:
    """Decode a GEMSEO design-space value to the user-facing parameter value."""
    value = _coerce_scalar(raw_value)
    if parameter["type"] == "choice":
        choices = parameter.get("values", [])
        if not choices:
            raise OptimizationConfigurationError(
                f"Choice parameter {parameter['name']} requires at least one value."
            )
        if isinstance(value, (str, bool)) and value in choices:
            return value
        try:
            index = int(round(float(value)))
        except (TypeError, ValueError) as exc:
            raise OptimizationExecutionError(
                f"Cannot decode choice parameter {parameter['name']} from value {value!r}."
            ) from exc
        if not 0 <= index < len(choices):
            raise OptimizationExecutionError(
                f"Choice index {index} is out of bounds for parameter {parameter['name']}."
            )
        return choices[index]

    value_type = parameter.get("value_type", "float")
    if value_type == "int" and not isinstance(value, bool):
        return int(round(float(value)))
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        return float(value)
    return value


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
            raise RemoteEvaluationContractError(
                f"Execution service rejected the evaluation request with HTTP {exc.response.status_code}."
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
            self.parameter_definitions = {
                name: {"name": name, "type": "range", "value_type": "float"}
                for name in self.input_names
            }
        else:
            self.parameter_definitions = {
                parameter["name"]: parameter for parameter in inputs
            }
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
        self.evaluator = evaluator
        self.parameters = parameters
        self.objectives = objectives
        self.constraints = constraints or []
        self.fidelity_parameter = fidelity_parameter
        self.use_bonsai = use_bonsai
        self.parameter_constraints = parameter_constraints

    def explore(self, n_samples: int = 10, n_processes: int = 1) -> dict[str, Any]:
        """Runs a Design of Experiments (DOE) exploration using GEMSEO DOEScenario.

        Args:
            n_samples: Number of samples to evaluate.
            n_processes: Number of concurrent processes.

        Returns:
            A dictionary containing the exploration history.
        """
        if hasattr(self.evaluator, "problem"):
            discipline = self.evaluator.problem
        else:
            outputs = [o["name"] for o in self.objectives] + [
                c["name"] for c in self.constraints
            ]
            discipline = RemoteDiscipline(self.evaluator, self.parameters, outputs)

        design_space = _build_design_space(self.parameters)

        # We pass multiple objective names to GEMSEO
        objective_names = [o["name"] for o in self.objectives]

        scenario = create_scenario(
            [discipline],
            formulation_name="MDF",
            objective_name=objective_names,
            design_space=design_space,
            scenario_type="DOE",
        )

        _add_constraints_to_scenario(scenario, self.constraints)

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
        except Exception as e:
            logger.error(f"Exploration failed: {e}")
            return {"history": []}

    def optimize(self, n_steps: int = 5, n_init: int = 5) -> dict[str, Any]:
        """Runs the optimization loop using GEMSEO MDOScenario."""
        if self.fidelity_parameter is not None:
            warnings.warn("fidelity_parameter is ignored.")
        if hasattr(self.evaluator, "problem"):
            discipline = self.evaluator.problem
        else:
            outputs = [o["name"] for o in self.objectives] + [
                c["name"] for c in self.constraints
            ]
            discipline = RemoteDiscipline(self.evaluator, self.parameters, outputs)

        design_space = _build_design_space(self.parameters)

        # Build Scenario
        objective_names = [o["name"] for o in self.objectives]

        # Explicitly configure maximize_objective per user request.
        # GEMSEO maximize_objective expects a single boolean or a list of booleans
        maximize_objective = [not o.get("minimize", True) for o in self.objectives]
        if len(maximize_objective) == 1:
            maximize_objective = maximize_objective[0]

        scenario = create_scenario(
            [discipline],
            formulation_name="MDF",
            objective_name=objective_names,
            maximize_objective=maximize_objective,
            design_space=design_space,
            name="MDOScenario_Ax",
        )

        _add_constraints_to_scenario(scenario, self.constraints)

        algo = None
        try:
            from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary

            problem = scenario.formulation.optimization_problem

            algo = AxOptimizationLibrary()
            algo.execute(
                problem,
                max_iter=n_steps,
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
