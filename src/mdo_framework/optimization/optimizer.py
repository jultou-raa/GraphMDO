import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from typing import List, Dict, Any, Protocol
import openmdao.api as om
import httpx


class Evaluator(Protocol):
    def evaluate(
        self, x: torch.Tensor, design_vars: List[str], objective: str
    ) -> torch.Tensor:
        ...


class LocalEvaluator:
    def __init__(self, problem: om.Problem):
        self.problem = problem

    def evaluate(
        self, x: torch.Tensor, design_vars: List[str], objective: str
    ) -> torch.Tensor:
        x_np = x.detach().numpy().flatten()
        for i, name in enumerate(design_vars):
            self.problem.set_val(name, x_np[i])
        self.problem.run_model()
        obj_val = self.problem.get_val(objective)
        return torch.tensor([obj_val], dtype=torch.double).reshape(1, 1)


class RemoteEvaluator:
    def __init__(self, service_url: str):
        self.service_url = service_url

    def evaluate(
        self, x: torch.Tensor, design_vars: List[str], objective: str
    ) -> torch.Tensor:
        payload = {
            "inputs": {
                name: float(val)
                for name, val in zip(design_vars, x.detach().numpy().flatten())
            },
            "objective": objective,
        }
        response = httpx.post(f"{self.service_url}/evaluate", json=payload)
        response.raise_for_status()
        data = response.json()
        return torch.tensor([data["result"]], dtype=torch.double).reshape(1, 1)


class BayesianOptimizer:
    """
    Bayesian Optimizer using BoTorch to drive execution via an Evaluator.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        design_vars: List[str],
        objective: str,
        bounds: torch.Tensor,
    ):
        self.evaluator = evaluator
        self.design_vars = design_vars
        self.objective = objective
        self.bounds = bounds

    def optimize(self, n_steps: int = 5, n_init: int = 5) -> Dict[str, Any]:
        """Runs the optimization loop."""
        n_vars = len(self.design_vars)

        # Initial Design of Experiments (Random)
        train_x = (
            torch.rand(n_init, n_vars, dtype=torch.double)
            * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )
        train_y = torch.cat(
            [
                self.evaluator.evaluate(
                    x.unsqueeze(0), self.design_vars, self.objective
                )
                for x in train_x
            ]
        )

        for i in range(n_steps):
            # Normalize data
            train_y_std = standardize(train_y)

            # Fit GP model
            gp = SingleTaskGP(train_x, train_y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Define Acquisition Function (Expected Improvement)
            EI = ExpectedImprovement(gp, best_f=train_y_std.max())

            # Optimize Acquisition Function
            candidate, _ = optimize_acqf(
                EI,
                bounds=self.bounds,
                q=1,
                num_restarts=5,
                raw_samples=20,
            )

            # Evaluate new candidate
            new_y = self.evaluator.evaluate(candidate, self.design_vars, self.objective)

            # Update training data
            train_x = torch.cat([train_x, candidate])
            train_y = torch.cat([train_y, new_y])

        # Find best result
        best_idx = train_y.argmax()
        best_x = train_x[best_idx]
        best_y = train_y[best_idx]

        return {
            "best_x": best_x.numpy(),
            "best_y": best_y.item(),
            "history_x": train_x.numpy(),
            "history_y": train_y.numpy(),
        }
