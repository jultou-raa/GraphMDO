import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from typing import List, Dict, Any
import openmdao.api as om


class BayesianOptimizer:
    """
    Bayesian Optimizer using BoTorch to drive OpenMDAO execution.
    """

    def __init__(
        self,
        problem: om.Problem,
        design_vars: List[str],
        objective: str,
        constraints: List[str] = None,
    ):
        self.problem = problem
        self.design_vars = design_vars
        self.objective = objective
        self.constraints = constraints or []
        self.bounds = self._get_bounds()

    def _get_bounds(self) -> torch.Tensor:
        """Extracts bounds from OpenMDAO problem."""
        # This assumes design variables are scalar and have bounds set in the problem
        # or we need to query the metadata.
        # For simplicity, we assume generic [0, 1] bounds if not specified,
        # but robust implementation should read from self.problem.model.get_io_metadata()

        # Placeholder: assume 1D design variables for now, or get from problem
        # We need to introspect the problem.
        # For this skeleton, we'll assume bounds are passed or defaulted.
        # Let's assume user sets them in OpenMDAO and we retrieve them.

        # We'll use a fixed range [0, 10] for demo purposes if not found.
        n_vars = len(self.design_vars)
        return torch.tensor([[0.0] * n_vars, [10.0] * n_vars], dtype=torch.double)

    def _evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the objective function using OpenMDAO."""
        # x is (q, d) tensor. We handle q=1 for now.
        x_np = x.detach().numpy().flatten()

        # Set design variables
        for i, name in enumerate(self.design_vars):
            self.problem.set_val(name, x_np[i])

        # Run model
        self.problem.run_model()

        # Get objective
        obj_val = self.problem.get_val(self.objective)

        # Return as tensor
        return torch.tensor([obj_val], dtype=torch.double).reshape(1, 1)

    def optimize(self, n_steps: int = 5, n_init: int = 5) -> Dict[str, Any]:
        """Runs the optimization loop."""
        n_vars = len(self.design_vars)

        # Initial Design of Experiments (Random)
        train_x = (
            torch.rand(n_init, n_vars, dtype=torch.double)
            * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )
        train_y = torch.cat([self._evaluate(x.unsqueeze(0)) for x in train_x])

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
            new_y = self._evaluate(candidate)

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
