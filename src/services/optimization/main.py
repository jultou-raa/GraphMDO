import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from mdo_framework.optimization.optimizer import BayesianOptimizer, RemoteEvaluator


def to_jsonable(obj: Any) -> Any:
    """Recursively converts objects to JSON-serializable types (handling NumPy and Tensors)."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "tolist") and callable(obj.tolist):
        # Handle PyTorch tensors and other objects with .tolist()
        return obj.tolist()
    if hasattr(obj, "item") and callable(obj.item):
        # Handle scalars from Tensors/NumPy
        return obj.item()
    return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the shared HTTP client
    async with httpx.AsyncClient() as client:
        app.state.client = client
        yield


app = FastAPI(title="Optimization Service", lifespan=lifespan)

EXECUTION_SERVICE_URL = os.getenv("EXECUTION_SERVICE_URL", "http://localhost:8002")
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8001")


class ObjectiveConfig(BaseModel):
    name: str
    minimize: bool = True
    fidelity: str | None = (
        None  # indicates this objective depends on a fidelity parameter
    )


class ConstraintConfig(BaseModel):
    name: str
    bound: float
    op: str = "<="  # "<=" or ">="


class OptimizeRequest(BaseModel):
    objectives: list[ObjectiveConfig]
    constraints: list[ConstraintConfig] | None = None
    fidelity_parameter: str | None = None  # Name of the parameter determining fidelity
    n_steps: int = 5
    n_init: int = 5
    use_bonsai: bool = False


@app.post("/optimize")
async def optimize(req: OptimizeRequest, request: Request):
    # 1. Fetch schema from Graph Service
    try:
        client: httpx.AsyncClient = request.app.state.client
        resp = await client.get(f"{GRAPH_SERVICE_URL}/schema")
        resp.raise_for_status()
        schema = resp.json()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch graph schema: {e}",
        )

    from mdo_framework.core.topology import TopologicalAnalyzer

    # 2. Identify Design Variables recursively from requested objectives and constraints
    analyzer = TopologicalAnalyzer(schema)

    target_outputs = [obj.name for obj in req.objectives]
    if req.constraints:
        target_outputs.extend([c.name for c in req.constraints])

    try:
        design_vars, _ = analyzer.resolve_dependencies(target_outputs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not design_vars:
        raise HTTPException(
            status_code=400,
            detail="No independent design variables found in the graph for the requested targets.",
        )

    # 3. Extract parameter definitions
    parameters = analyzer.extract_parameters(design_vars)

    if not parameters:
        raise HTTPException(
            status_code=400,
            detail="Failed to extract parameter definitions from schema.",
        )

    # 4. Setup Evaluator
    evaluator = RemoteEvaluator(EXECUTION_SERVICE_URL)

    # 5. Setup Optimizer
    try:
        constraints = (
            [c.model_dump() for c in req.constraints] if req.constraints else None
        )

        optimizer = BayesianOptimizer(
            evaluator=evaluator,
            parameters=parameters,
            objectives=[o.model_dump() for o in req.objectives],
            constraints=constraints,
            fidelity_parameter=req.fidelity_parameter,
            use_bonsai=req.use_bonsai,
        )

        # 6. Run Optimization (offload to thread to avoid blocking the event loop)
        result = await asyncio.to_thread(
            optimizer.optimize,
            n_steps=req.n_steps,
            n_init=req.n_init,
        )

        # Convert tensor/numpy to lists for JSON
        return to_jsonable(
            {
                "best_parameters": result.get("best_parameters"),
                "best_objectives": result.get("best_objectives"),
                "history": [
                    {
                        "parameters": trial["parameters"],
                        "objectives": trial["objectives"],
                    }
                    for trial in result.get("history", [])
                ],
                "serialized_client": result.get("serialized_client"),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}
