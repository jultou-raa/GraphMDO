from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import httpx
import os
import asyncio
import time
from mdo_framework.core.translator import GraphProblemBuilder

app = FastAPI(title="Execution Service")

GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8001")

# --- Cache Configuration ---
SCHEMA_CACHE: Optional[Dict] = None
SCHEMA_CACHE_EXPIRY: float = 0
CACHE_TTL = 60.0  # seconds
cache_lock = asyncio.Lock()


# --- Tool Registry ---
# In a scalable system, this might be dynamic or call other services.
def paraboloid_func(x, y):
    """
    f(x, y) = (x-3)**2 + xy + (y+4)**2 - 3
    """
    return (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0


TOOL_REGISTRY = {"Paraboloid": paraboloid_func}


class EvaluateRequest(BaseModel):
    inputs: Dict[str, float]
    objective: str


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    # 1. Fetch Schema from Graph Service (with caching)
    global SCHEMA_CACHE, SCHEMA_CACHE_EXPIRY
    async with cache_lock:
        if SCHEMA_CACHE is None or time.time() > SCHEMA_CACHE_EXPIRY:
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.get(f"{GRAPH_SERVICE_URL}/schema")
                    resp.raise_for_status()
                    SCHEMA_CACHE = resp.json()
                    SCHEMA_CACHE_EXPIRY = time.time() + CACHE_TTL
                except httpx.RequestError as e:
                    raise HTTPException(
                        status_code=503, detail=f"Could not reach Graph Service: {e}"
                    )
    schema = SCHEMA_CACHE

    # 2. Build Problem
    builder = GraphProblemBuilder(schema)
    try:
        prob = builder.build_problem(TOOL_REGISTRY)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build problem: {e}")

    # 3. Evaluate
    # evaluator = LocalEvaluator(prob)

    # Convert inputs to tensor format expected by Evaluator (Wait, LocalEvaluator expects tensor but calls set_val)
    # Actually, LocalEvaluator needs a tensor. Let's adapt here.
    # But wait, LocalEvaluator is designed to be called by BayesianOptimizer which passes a tensor.
    # Here we are the endpoint. We receive JSON.

    # We can just use the problem directly or use LocalEvaluator if we want consistency.
    # Let's use problem directly for simplicity in this endpoint logic.

    for name, val in req.inputs.items():
        try:
            prob.set_val(name, val)
        except Exception:
            # Variable might not exist or be controllable
            pass

    try:
        prob.run_model()
        result = prob.get_val(req.objective)
        # result is likely a numpy array
        if hasattr(result, "item"):
            result = result.item()
        elif isinstance(result, (list, tuple)):
            result = result[0]

        return {"result": float(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}
