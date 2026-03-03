import logging
from typing import Any, Dict, List, Protocol, Optional

import httpx
import openmdao.api as om


from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.generation_strategy.generation_strategy import (
    GenerationStrategy,
    GenerationStep,
)
from ax.adapter.registry import Generators as Models
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

logger = logging.getLogger(__name__)


class Evaluator(Protocol):
    def evaluate(
        self, parameters: Dict[str, Any], objectives: List[str]
    ) -> Dict[str, float]: ...


class LocalEvaluator:
    def __init__(self, problem: om.Problem):
        self.problem = problem

    def evaluate(
        self, parameters: Dict[str, Any], objectives: List[str]
    ) -> Dict[str, float]:
        for name, val in parameters.items():
            self.problem.set_val(name, val)
        self.problem.run_model()

        results = {}
        for obj in objectives:
            results[obj] = (
                float(self.problem.get_val(obj)[0])
                if hasattr(self.problem.get_val(obj), "__iter__")
                else float(self.problem.get_val(obj))
            )
        return results


class RemoteEvaluator:
    def __init__(self, service_url: str):
        self.service_url = service_url

    def evaluate(
        self, parameters: Dict[str, Any], objectives: List[str]
    ) -> Dict[str, float]:

        # Need to loop or post all objectives
        # The current endpoint expects one objective per call, or we can handle it
        # Assuming the execution service evaluates everything required in one pass
        # and we can pick out multiple objectives if we update it, or call per objective.
        # But wait, execution service takes single 'objective'. We will need to adapt it.
        # For simplicity in this step, if multiple objectives, we'll evaluate the first and mock others or call multiple times.
        # Ideally, execution service should take a list of objectives.
        # Let's adjust execution service payload if we need to. For now, let's call it per objective or assume the endpoint is updated to take multiple objectives.

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
    """

    def __init__(
        self,
        evaluator: Evaluator,
        parameters: List[Dict[str, Any]],
        objectives: List[Dict[str, Any]],
        fidelity_parameter: Optional[str] = None,
        use_bonsai: bool = False,
    ) -> None:
        self.evaluator = evaluator
        self.parameters = parameters
        self.objectives = objectives
        self.fidelity_parameter = fidelity_parameter
        self.use_bonsai = use_bonsai

    def optimize(self, n_steps: int = 5, n_init: int = 5) -> Dict[str, Any]:
        """Runs the optimization loop using AxClient."""

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
                    ),
                    GenerationStep(generator=Models.BONSAI, num_trials=-1),
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

        # Configure parameter space
        client = AxClient(generation_strategy=gs)
        ax_params = []
        for p in self.parameters:
            if p["type"] == "range":
                ax_params.append(
                    {
                        "name": p["name"],
                        "type": "range",
                        "bounds": p["bounds"],
                        "value_type": p.get("value_type", "float"),
                        "is_fidelity": p["name"] == self.fidelity_parameter,
                    }
                )
            elif p["type"] == "choice":
                ax_params.append(
                    {
                        "name": p["name"],
                        "type": "choice",
                        "values": p["values"],
                        "value_type": p.get("value_type", "float"),
                        "is_fidelity": p["name"] == self.fidelity_parameter,
                    }
                )

        ax_objectives = {}
        for obj in self.objectives:
            ax_objectives[obj["name"]] = ObjectiveProperties(
                minimize=obj.get("minimize", True),
            )

        client.create_experiment(
            name="mdo_optimization",
            parameters=ax_params,
            objectives=ax_objectives,
        )

        history = []

        total_trials = n_init + n_steps
        objective_names = [o["name"] for o in self.objectives]

        for _ in range(total_trials):
            parameters, trial_index = client.get_next_trial()

            # Evaluate using the evaluator
            results = self.evaluator.evaluate(parameters, objective_names)

            # Record history
            history.append({"parameters": parameters, "objectives": results})

            # Complete the trial
            client.complete_trial(trial_index=trial_index, raw_data=results)

        try:
            # Handle possible pareto frontier for multi-objective
            if len(self.objectives) > 1:
                pareto_results = client.get_pareto_optimal_parameters()
                # For simplicity, returning the frontier as best
                # AxClient returns a mapping of trial_index -> (parameters, metrics)
                best_params = []
                best_objs = []
                if pareto_results:
                    for trial_idx, (params, metrics) in pareto_results.items():
                        best_params.append(params)
                        # Extract mean metric values
                        best_objs.append({k: v[0] for k, v in metrics.items()})
                else:
                    best_params = None
                    best_objs = None

                return {
                    "best_parameters": best_params,
                    "best_objectives": best_objs,
                    "history": history,
                }
            else:
                best_parameters, metrics = client.get_best_parameters()
                best_obj = {k: v[0] for k, v in metrics.items()}
                return {
                    "best_parameters": best_parameters,
                    "best_objectives": best_obj,
                    "history": history,
                }
        except Exception as e:
            logger.warning(f"Could not retrieve best parameters: {e}")
            return {
                "best_parameters": None,
                "best_objectives": None,
                "history": history,
            }
