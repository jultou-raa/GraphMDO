"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import logging
import warnings
from typing import Any, Protocol

import httpx
import numpy as np
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import Discipline

import mdo_framework.optimization.ax_algo_lib  # noqa: F401

logger = logging.getLogger(__name__)


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

    def __init__(self, service_url: str):
        self.service_url = service_url

    def evaluate(
        self,
        parameters: dict[str, Any],
        objectives: list[str],
    ) -> dict[str, float]:

        payload = {
            "inputs": parameters,
            "objectives": objectives,
        }
        response = httpx.post(f"{self.service_url}/evaluate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["results"]


class RemoteDiscipline(Discipline):
    def __init__(
        self, evaluator: "RemoteEvaluator", inputs: list[str], outputs: list[str]
    ):
        super().__init__(name="RemoteExecution")
        self.evaluator = evaluator
        self.input_names = inputs
        self.output_names = outputs
        self.input_grammar.update_from_names(self.input_names)
        self.output_grammar.update_from_names(self.output_names)
        for in_name in self.input_names:
            self.default_input_data[in_name] = np.array([0.0])

    def _run(self, input_data: dict[str, np.ndarray]) -> None:
        params = {
            k: v.tolist()[0] if v.size == 1 else v.tolist()
            for k, v in input_data.items()
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
            inputs = [p["name"] for p in self.parameters]
            outputs = [o["name"] for o in self.objectives] + [
                c["name"] for c in self.constraints
            ]
            discipline = RemoteDiscipline(self.evaluator, inputs, outputs)

        design_space = DesignSpace()
        for p in self.parameters:
            if p["type"] == "range":
                design_space.add_variable(
                    p["name"], lower_bound=p["bounds"][0], upper_bound=p["bounds"][1]
                )
            elif p["type"] == "choice":
                vals = p["values"]
                if len(vals) == 1:
                    design_space.add_variable(
                        p["name"], value=0 if isinstance(vals[0], str) else vals[0]
                    )
                else:
                    if isinstance(vals[0], str):
                        design_space.add_variable(
                            p["name"], lower_bound=0, upper_bound=len(vals) - 1
                        )
                    else:
                        design_space.add_variable(
                            p["name"], lower_bound=min(vals), upper_bound=max(vals)
                        )

        objective_name = self.objectives[0]["name"]

        scenario = create_scenario(
            [discipline],
            formulation_name="MDF",
            objective_name=objective_name,
            design_space=design_space,
            scenario_type="DOE",
        )

        for c in self.constraints:
            ctype = "ineq" if c["op"] == "<=" else "eq"
            scenario.add_constraint(c["name"], constraint_type=ctype, value=c["bound"])

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
                "history": scenario.formulation.optimization_problem.database.to_dict(),
            }
        except Exception as e:
            logger.error(f"Exploration failed: {e}")
            return {"history": {}}

    def optimize(self, n_steps: int = 5, n_init: int = 5) -> dict[str, Any]:
        """Runs the optimization loop using GEMSEO MDOScenario."""
        if self.fidelity_parameter is not None:
            warnings.warn("fidelity_parameter is ignored.")

        # Get discipline
        if hasattr(self.evaluator, "problem"):
            discipline = self.evaluator.problem
        else:
            inputs = [p["name"] for p in self.parameters]
            outputs = [o["name"] for o in self.objectives] + [
                c["name"] for c in self.constraints
            ]
            discipline = RemoteDiscipline(self.evaluator, inputs, outputs)

        # Build DesignSpace
        design_space = DesignSpace()
        for p in self.parameters:
            if p["type"] == "range":
                design_space.add_variable(
                    p["name"], lower_bound=p["bounds"][0], upper_bound=p["bounds"][1]
                )
            elif p["type"] == "choice":
                vals = p["values"]
                if len(vals) == 1:
                    design_space.add_variable(
                        p["name"], value=0 if isinstance(vals[0], str) else vals[0]
                    )
                else:
                    if isinstance(vals[0], str):
                        design_space.add_variable(
                            p["name"], lower_bound=0, upper_bound=len(vals) - 1
                        )
                    else:
                        design_space.add_variable(
                            p["name"], lower_bound=min(vals), upper_bound=max(vals)
                        )

        # Build Scenario
        objective_name = self.objectives[0]["name"]
        minimize = self.objectives[0].get("minimize", True)

        scenario = create_scenario(
            [discipline],
            formulation_name="MDF",
            objective_name=objective_name,
            maximize_objective=not minimize,
            design_space=design_space,
            name="MDOScenario_Ax",
        )

        for c in self.constraints:
            ctype = "ineq" if c["op"] == "<=" else "eq"
            scenario.add_constraint(c["name"], constraint_type=ctype, value=c["bound"])

        try:
            from mdo_framework.optimization.ax_algo_lib import AxOptimizationLibrary

            problem = scenario.formulation.optimization_problem

            algo = AxOptimizationLibrary()
            algo.execute(
                problem,
                max_iter=n_steps + n_init,
                n_init=n_init,
                use_bonsai=self.use_bonsai,
                ax_parameters=self.parameters,
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

            # problem.optimum is GEMSEO's single source of truth: it searches
            # all evaluated points (including every Ax trial) for the best
            # feasible result, consistent with what the GEMSEO logger reports.
            optimum = problem.optimum
            best_params = {}
            offset = 0
            for v in design_space.variable_names:
                s = design_space.variable_sizes[v]
                val = optimum.design[offset : offset + s]
                best_params[v] = val[0] if s == 1 else val.tolist()
                offset += s

            best_obj = float(optimum.objective)

            return {
                "best_parameters": best_params,
                "best_objectives": {objective_name: best_obj},
                "history": [],
                "serialized_client": "{}",
            }
        except Exception as e:
            import traceback

            logger.error(f"Optimization failed: {e}\n{traceback.format_exc()}")
            return {
                "best_parameters": None,
                "best_objectives": None,
                "history": [],
                "serialized_client": "{}",
            }
