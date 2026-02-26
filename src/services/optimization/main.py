from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import torch
from mdo_framework.optimization.optimizer import BayesianOptimizer, RemoteEvaluator

app = FastAPI(title="Optimization Service")

EXECUTION_SERVICE_URL = os.getenv("EXECUTION_SERVICE_URL", "http://localhost:8002")


class OptimizeRequest(BaseModel):
    design_vars: List[str]
    objective: str
    bounds: List[List[float]] = [[0.0, 1.0], [0.0, 1.0]]
    n_steps: int = 5
    n_init: int = 5


@app.post("/optimize")
async def optimize(req: OptimizeRequest):
    # 1. Setup Evaluator
    evaluator = RemoteEvaluator(EXECUTION_SERVICE_URL)

    # 2. Setup Bounds
    bounds_tensor = torch.tensor(req.bounds, dtype=torch.double)

    # 3. Setup Optimizer
    optimizer = BayesianOptimizer(
        evaluator=evaluator,
        design_vars=req.design_vars,
        objective=req.objective,
        bounds=bounds_tensor,
    )

    try:
        # 4. Run Optimization
        result = optimizer.optimize(n_steps=req.n_steps, n_init=req.n_init)

        # Convert numpy results to list for JSON serialization
        return {
            "best_x": result["best_x"].tolist(),
            "best_y": float(result["best_y"]),
            "history_x": result["history_x"].tolist(),
            "history_y": result["history_y"].tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}
