"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import logging
from collections.abc import Sequence
from typing import Any, Protocol

import httpx
import numpy as np
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import Discipline
from gemseo.core.discipline.base_discipline import CacheType
from gemseo.typing import StrKeyMapping

from mdo_framework.core.errors import EvaluationError, evaluation_error_from_payload
from mdo_framework.core.topology import to_parameter_definition
from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary
from mdo_framework.optimization.bo_library import BaseBOLibrary, add_constraints
from mdo_framework.optimization.bo_types import BORunResult
from mdo_framework.optimization.errors import (
    OptimizationConfigurationError,
    OptimizationExecutionError,
)
from mdo_framework.optimization.parameter_codec import (
    ParameterDefinitionError,
    ParameterValueError,
)
from mdo_framework.optimization.parameter_codec import (
    decode_parameter_value as _shared_decode_parameter_value,
)
from mdo_framework.optimization.random_search import RandomSearchLibrary
from mdo_framework.schema import (
    ChoiceVar,
    ConstraintSpec,
    DesignVariable,
    ObjectiveSpec,
    Scalar,
)

logger = logging.getLogger(__name__)

ALGORITHMS: dict[str, type[BaseBOLibrary]] = {
    "Ax_Bayesian": AxOptimizationLibrary,
    "BO_RandomSearch": RandomSearchLibrary,
}


class RemoteEvaluationTransportError(RuntimeError):
    """Raised when the execution service cannot be reached reliably."""


class RemoteEvaluationContractError(TypeError):
    """Raised when the execution service response breaks the expected contract."""


def _decode_parameter_value(parameter: dict[str, Any], raw_value: Any) -> Scalar:
    """Decode a GEMSEO design-space value to the user-facing parameter value."""
    try:
        return _shared_decode_parameter_value(parameter, raw_value)
    except ParameterDefinitionError as exc:
        raise OptimizationConfigurationError(str(exc)) from exc
    except ParameterValueError as exc:
        raise OptimizationExecutionError(str(exc)) from exc


def _build_design_space(design_variables: Sequence[DesignVariable]) -> DesignSpace:
    """Build a GEMSEO design space with explicit integer encoding for choices.

    The current value is left to the optimization library, which sets the
    start point.
    """
    design_space = DesignSpace()
    for variable in design_variables:
        if isinstance(variable, ChoiceVar):
            design_space.add_variable(
                variable.name,
                lower_bound=0,
                upper_bound=len(variable.choices) - 1,
                type_="integer",
            )
        else:
            extra_args = {"type_": "integer"} if variable.value_type == "int" else {}
            design_space.add_variable(
                variable.name,
                lower_bound=variable.lower,
                upper_bound=variable.upper,
                **extra_args,
            )
    return design_space


def _result_contract(result: BORunResult) -> dict[str, Any]:
    """Return the result of a run in the contract of ``BayesianOptimizer``."""
    best = result.best
    objectives = [b for b in result.bindings if b.role == "objective"]
    constraints = [b for b in result.bindings if b.role == "constraint"]
    contract: dict[str, Any] = {
        "best_parameters": None,
        "best_objectives": None,
        "feasible": result.feasible,
        "constraints": {},
        "pareto_front": [
            {
                "parameters": dict(record.parameters),
                "objectives": {
                    b.name: record.outcome.metrics[b.name] for b in objectives
                },
            }
            for record in result.pareto_front
        ],
        "history": [
            record.to_history_entry(result.bindings) for record in result.records
        ],
        "stop_reason": result.stop_reason,
        "evaluations": result.evaluations,
    }
    if best is not None:
        metrics = best.outcome.metrics
        contract["best_parameters"] = dict(best.parameters)
        contract["best_objectives"] = {b.name: metrics[b.name] for b in objectives}
        contract["constraints"] = {
            b.name: {
                "value": metrics[b.name],
                "margin": b.margin(metrics[b.name]),
                "satisfied": b.satisfied(metrics[b.name]),
                "tolerance": b.tolerance,
            }
            for b in constraints
        }
    return contract


def _no_completed_trial_message(result: BORunResult | None) -> str:
    if result is None:
        return "Optimization did not run."
    message = f"No trial completed (stop reason: {result.stop_reason})."
    reasons = [r.outcome.reason for r in result.records if r.outcome.status == "failed"]
    if reasons:
        message += f" Last failure: {reasons[-1]}"
    return message


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
    """GEMSEO discipline evaluating the design variables through an evaluator.

    Args:
        evaluator: Local or remote implementation of the Evaluator protocol.
        design_variables: Inputs of the discipline, in design space order.
        outputs: Names of the outputs the evaluator computes.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        design_variables: Sequence[DesignVariable],
        outputs: Sequence[str],
    ):
        super().__init__(name="RemoteExecution")
        self.evaluator = evaluator
        self.parameter_definitions = {
            variable.name: to_parameter_definition(variable)
            for variable in design_variables
        }
        self.input_names = list(self.parameter_definitions)
        self.output_names = list(outputs)
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
    only logs it, so the typed error would otherwise be lost. The recorder
    never caches: caching is the wrapped discipline's own policy.
    """

    def __init__(self, discipline: Discipline) -> None:
        super().__init__(name=discipline.name)
        self.set_cache(CacheType.NONE)
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
    """Bayesian optimizer on the backend-neutral driver of GEMSEO.

    Args:
        evaluator: Local or remote implementation of the Evaluator protocol.
        design_variables: Design variables, in the order of the design space.
        objectives: Objectives to optimize, at least one.
        constraints: Inequality constraints on the outputs.
        parameter_constraints: Linear constraints between design variables.
        use_bonsai: Whether to use the experimental BONSAI acquisition of Ax.
        algorithm: Backend, a key of ``ALGORITHMS``.

    Raises:
        OptimizationConfigurationError: If there is no design variable or
            objective, the algorithm is unknown, or ``use_bonsai`` is not
            supported by it.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        design_variables: Sequence[DesignVariable],
        objectives: Sequence[ObjectiveSpec],
        constraints: Sequence[ConstraintSpec] = (),
        parameter_constraints: Sequence[str] = (),
        use_bonsai: bool = False,
        algorithm: str = "Ax_Bayesian",
    ) -> None:
        if not design_variables:
            raise OptimizationConfigurationError(
                "At least one design variable is required."
            )
        if not objectives:
            raise OptimizationConfigurationError("At least one objective is required.")
        if algorithm not in ALGORITHMS:
            raise OptimizationConfigurationError(
                f"Unknown algorithm {algorithm!r}. Available: {sorted(ALGORITHMS)}."
            )
        settings_class = ALGORITHMS[algorithm].ALGORITHM_INFOS[algorithm].Settings
        if use_bonsai and "use_bonsai" not in settings_class.model_fields:
            raise OptimizationConfigurationError(
                f"use_bonsai is not supported by the algorithm {algorithm!r}."
            )
        self.evaluator = evaluator
        self.design_variables = tuple(design_variables)
        self.objectives = tuple(objectives)
        self.constraints = tuple(constraints)
        self.parameter_constraints = tuple(parameter_constraints)
        self.use_bonsai = use_bonsai
        self.algorithm = algorithm
        self._settings_class = settings_class

    def _build_output_names(self) -> list[str]:
        names = [objective.name for objective in self.objectives]
        names += [constraint.name for constraint in self.constraints]
        return list(dict.fromkeys(names))

    def _build_discipline(self) -> Discipline:
        if hasattr(self.evaluator, "problem"):
            return self.evaluator.problem
        return RemoteDiscipline(
            self.evaluator,
            self.design_variables,
            self._build_output_names(),
        )

    def _prepare_scenario_context(self) -> tuple[Discipline, DesignSpace, list[str]]:
        return (
            self._build_discipline(),
            _build_design_space(self.design_variables),
            [objective.name for objective in self.objectives],
        )

    def _create_scenario(
        self,
        *,
        discipline: Discipline,
        design_space: DesignSpace,
        objective_names: list[str],
        scenario_type: str | None = None,
        maximize_objective: bool | None = None,
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
        add_constraints(scenario, self.constraints)
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

    def optimize(
        self,
        n_steps: int = 10,
        n_init: int = 5,
        evaluate_x0: bool | None = None,
        max_consecutive_failures: int = 5,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Runs Bayesian optimization using a GEMSEO MDOScenario.

        The tools are called ``n_init + n_steps`` times, plus once for the
        start point x0 when it is evaluated. Fewer calls happen only when the
        run stops early: the search space is exhausted, ``max_consecutive_failures``
        trials fail in a row or the time limit is reached.

        Args:
            n_steps: Model-driven trials, at least 1.
            n_init: Initial trials, at least 1.
            evaluate_x0: Whether to evaluate the start point first. ``None``
                evaluates it when every design variable declares ``initial``.
            max_consecutive_failures: Failed trials in a row that end the run,
                at least 1.
            seed: Seed of the backend, ``None`` for a random one.

        Returns:
            The result: ``best_parameters``, ``best_objectives`` (by objective
            name), ``feasible``, ``constraints`` (value, margin, satisfied and
            tolerance at the best point), ``pareto_front`` (several objectives
            only), ``history`` (one entry per trial), ``stop_reason`` and
            ``evaluations`` (per phase).

        Raises:
            OptimizationConfigurationError: If a budget is < 1, or the design
                space or the backend settings are invalid.
            OptimizationExecutionError: If no trial completed or the run was
                aborted. Its ``partial_result`` keeps the trials made so far.
        """
        for budget_name, budget in (
            ("n_steps", n_steps),
            ("n_init", n_init),
            ("max_consecutive_failures", max_consecutive_failures),
        ):
            if budget < 1:
                raise OptimizationConfigurationError(
                    f"{budget_name} must be >= 1, got {budget}."
                )
        discipline, design_space, objective_names = self._prepare_scenario_context()
        scenario = self._create_scenario(
            discipline=discipline,
            design_space=design_space,
            objective_names=objective_names,
            maximize_objective=len(self.objectives) == 1
            and not self.objectives[0].minimize,
            name=f"MDOScenario_{self.algorithm}",
        )
        settings_kwargs: dict[str, Any] = {
            "design_variables": self.design_variables,
            "objectives": self.objectives,
            "constraints": self.constraints,
            "parameter_constraints": self.parameter_constraints,
            "n_init": n_init,
            "n_steps": n_steps,
            "evaluate_x0": evaluate_x0,
            "seed": seed,
            "max_consecutive_failures": max_consecutive_failures,
        }
        if self.use_bonsai:
            settings_kwargs["use_bonsai"] = True
        library = ALGORITHMS[self.algorithm](self.algorithm)

        try:
            library.execute(
                scenario.formulation.optimization_problem,
                settings_model=self._settings_class(**settings_kwargs),
            )
            if library.result is not None and library.result.best is not None:
                self._write_reports(scenario)
        except (
            OptimizationConfigurationError,
            OptimizationExecutionError,
            RemoteEvaluationContractError,
            RemoteEvaluationTransportError,
        ) as error:
            if getattr(error, "partial_result", None) is None:
                error.partial_result = self._partial_result(library)
            raise
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            raise OptimizationExecutionError(
                f"Optimization failed: {e}",
                partial_result=self._partial_result(library),
            ) from e

        result = library.result
        if result is None or result.best is None:
            raise OptimizationExecutionError(
                _no_completed_trial_message(result),
                partial_result=self._partial_result(library),
            )
        return _result_contract(result)

    @staticmethod
    def _partial_result(library: BaseBOLibrary) -> dict[str, Any] | None:
        """Result contract of the trials a library made, if it got to run."""
        if library.result is None:
            return None
        return _result_contract(library.result)

    @staticmethod
    def _write_reports(scenario: Any) -> None:
        scenario.xdsmize(show_html=False)
        try:
            from gemseo.settings.post import OptHistoryView_Settings

            scenario.post_process(
                settings_model=OptHistoryView_Settings(save=True, show=False)
            )
        except Exception as pp_err:
            logger.warning(f"Failed to post-process: {pp_err}")
