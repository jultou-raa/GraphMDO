import os
from typing import List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mdo_framework.optimization.optimizer import BayesianOptimizer, RemoteEvaluator

app = FastAPI(title="Optimization Service")

EXECUTION_SERVICE_URL = os.getenv("EXECUTION_SERVICE_URL", "http://localhost:8002")


class ParameterConfig(BaseModel):
    name: str
    type: str  # "range" or "choice"
    value_type: str = "float"  # "float", "int", "str"
    bounds: Optional[List[float]] = None
    values: Optional[List[Union[float, int, str]]] = None


class ObjectiveConfig(BaseModel):
    name: str
    minimize: bool = True
    fidelity: Optional[str] = (
        None  # indicates this objective depends on a fidelity parameter
    )


class OptimizeRequest(BaseModel):
    parameters: List[ParameterConfig]
    objectives: List[ObjectiveConfig]
    fidelity_parameter: Optional[str] = (
        None  # Name of the parameter determining fidelity
    )
    n_steps: int = 5
    n_init: int = 5
    use_bonsai: bool = False


@app.post("/optimize")
async def optimize(req: OptimizeRequest):
    # 1. Setup Evaluator
    evaluator = RemoteEvaluator(EXECUTION_SERVICE_URL)

    # 2. Setup Optimizer
    try:
        optimizer = BayesianOptimizer(
            evaluator=evaluator,
            parameters=[p.model_dump() for p in req.parameters],
            objectives=[o.model_dump() for o in req.objectives],
            fidelity_parameter=req.fidelity_parameter,
            use_bonsai=req.use_bonsai,
        )

        # 3. Run Optimization
        result = optimizer.optimize(n_steps=req.n_steps, n_init=req.n_init)

        # Convert tensor/numpy to lists for JSON
        def to_list(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return obj

        return {
            "best_parameters": result.get("best_parameters"),
            "best_objectives": result.get("best_objectives"),
            "history": [
                {
                    "parameters": trial["parameters"],
                    "objectives": trial["objectives"],
                }
                for trial in result.get("history", [])
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}
