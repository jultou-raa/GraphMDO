import json
import warnings
import logging
from typing import Any, Protocol

import httpx
import openmdao.api as om


from ax.api.client import Client
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
from ax.generation_strategy.generation_strategy import (
    GenerationStrategy,
    GenerationStep,
)
from ax.adapter.registry import Generators as Models
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

logger = logging.getLogger(__name__)


class Evaluator(Protocol):
    def evaluate(
        self, parameters: dict[str, Any], objectives: list[str]
    ) -> dict[str, float]:
        """
        Evaluates the requested objectives given the design parameters.
        """
        ...


class RemoteEvaluator:
    """
    Evaluates the design parameters remotely by communicating with the Execution microservice.

    Args:
        service_url: The URL of the execution service.
    """

    def __init__(self, service_url: str):
        self.service_url = service_url

    def evaluate(
        self, parameters: dict[str, Any], objectives: list[str]
    ) -> dict[str, float]:

        payload = {
            "inputs": parameters,
            "objectives": objectives,
        }
        response = httpx.post(f"{self.service_url}/evaluate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["results"]


class BayesianOptimizer:
    """
    Bayesian Optimizer using Ax Platform.

    Args:
        evaluator: Local or Remote implementation of Evaluator protocol.
        parameters: Dict defining the variables bounds, choices, and types.
        objectives: Dict defining the targeted metrics and their directions.
        constraints: Dict defining boundaries mapped out of OpenMDAO runs.
        fidelity_parameter: Name of variable designating multi-fidelity.
        use_bonsai: Toggle for experimental algorithmic execution.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        parameters: list[dict[str, Any]],
        objectives: list[dict[str, Any]],
        constraints: list[dict[str, Any]] | None = None,
        fidelity_parameter: str | None = None,
        use_bonsai: bool = False,
    ) -> None:
        self.evaluator = evaluator
        self.parameters = parameters
        self.objectives = objectives
        self.constraints = constraints or []
        self.fidelity_parameter = fidelity_parameter
        self.use_bonsai = use_bonsai

    def optimize(self, n_steps: int = 5, n_init: int = 5) -> dict[str, Any]:
        """Runs the optimization loop using AxClient.

        Iteratively generates candidate parameters, evaluates them using the
        provided evaluator, and updates the underlying Gaussian Process model.

        Args:
            n_steps: Number of Bayesian optimization steps to perform. Default is 5.
            n_init: Number of initial Sobol (quasi-random) exploration steps. Default is 5.

        Returns:
            A dictionary containing:
            - 'best_parameters': The optimal parameters found (or None if unresolved).
            - 'best_objectives': The metrics associated with the optimal parameters.
            - 'history': List of dicts representing all evaluated trials.
            - 'serialized_client': JSON string representation of the Ax client state.

        Example:
            ```python
            optimizer = BayesianOptimizer(evaluator, parameters, objectives)
            result = optimizer.optimize(n_steps=10, n_init=10)
            print(result["best_parameters"])
            ```
        """

        # Fidelity parameters are not yet supported in the modern Ax `Client` API.
        # `ax.api.configs.RangeParameterConfig` does not expose `is_fidelity` or
        # `target_value`. The legacy `AxClient` workaround is not viable either,
        # as it is deprecated and scheduled for removal in Ax 1.4.0.
        # See: https://ax.dev for updates on the new API roadmap.
        if self.fidelity_parameter is not None:
            warnings.warn(
                f"The `fidelity_parameter` argument ('{self.fidelity_parameter}') is "
                "currently not supported by the modern Ax `Client` API. "
                "`RangeParameterConfig` does not yet expose `is_fidelity` or "
                "`target_value`. The parameter will be treated as a regular range "
                "parameter until this is addressed upstream in Ax. "
                "Track: https://github.com/facebook/ax",
                UserWarning,
                stacklevel=2,
            )

        # Determine client setup

        if self.use_bonsai:
            logger.warning("Experimental feature BONSAI algorithm is activated.")
            gs = GenerationStrategy(
                name="bonsai",
                nodes=[
                    GenerationStep(
                        generator=Models.SOBOL,
                        num_trials=n_init,
                        min_trials_observed=n_init,
                        generator_name="SOBOL",
                    ),
                    GenerationStep(
                        generator=Models.BOTORCH_MODULAR,
                        num_trials=-1,
                        generator_name="BONSAI",
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
                        generator_name="SOBOL",
                    ),
                    GenerationStep(
                        generator=Models.BOTORCH_MODULAR,
                        num_trials=-1,
                        generator_kwargs={
                            "botorch_acqf_class": qLogNoisyExpectedImprovement,
                        },
                        generator_name="qLogNEI",
                    ),
                ],
            )

        # Configure parameter space
        client = Client()

        ax_params = []
        for p in self.parameters:
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
                    )
                )

        client.configure_experiment(
            name="mdo_optimization",
            parameters=ax_params,
        )

        objective_str = ", ".join(
            [
                f"{'-' if obj.get('minimize', True) else ''}{obj['name']}"
                for obj in self.objectives
            ]
        )

        client.configure_optimization(
            objective=objective_str,
            outcome_constraints=[
                f"{c['name']} {c['op']} {c['bound']}" for c in self.constraints
            ],
        )

        client.set_generation_strategy(gs)

        history = []

        total_trials = n_init + n_steps
        objective_names = [o["name"] for o in self.objectives] + [
            c["name"] for c in self.constraints
        ]

        for _ in range(total_trials):
            trials = client.get_next_trials(max_trials=1)
            for trial_index, parameters in trials.items():
                # Evaluate using the evaluator
                results = self.evaluator.evaluate(parameters, objective_names)

                # Record history
                history.append({"parameters": parameters, "objectives": results})

                # Complete the trial
                client.complete_trial(trial_index=trial_index, raw_data=results)

        try:
            # Handle possible pareto frontier for multi-objective
            if len(self.objectives) > 1:
                frontier = client.get_pareto_frontier()
                # For simplicity, returning the frontier as best
                best_params = []
                best_objs = []
                if frontier:
                    for params, metrics, trial_idx, arm_name in frontier:
                        best_params.append(params)
                        # Extract mean metric values
                        best_objs.append(
                            {
                                k: v[0] if isinstance(v, tuple) else v
                                for k, v in metrics.items()
                            }
                        )
                else:
                    best_params = None
                    best_objs = None

                return {
                    "best_parameters": best_params,
                    "best_objectives": best_objs,
                    "history": history,
                    "serialized_client": json.dumps(client._to_json_snapshot()),
                }
            else:
                best_parameters, best_obj, trial_idx, arm_name = (
                    client.get_best_parameterization()
                )
                return {
                    "best_parameters": best_parameters,
                    "best_objectives": best_obj,
                    "history": history,
                    "serialized_client": json.dumps(client._to_json_snapshot()),
                }
        except Exception as e:
            logger.warning(f"Could not retrieve best parameters: {e}")
            return {
                "best_parameters": None,
                "best_objectives": None,
                "history": history,
                "serialized_client": client.to_json_snapshot(),
            }
