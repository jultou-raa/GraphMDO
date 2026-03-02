import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, TypeAlias

import httpx
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from mdo_framework.core.translator import GraphProblemBuilder

# Configure logging
logger = logging.getLogger("uvicorn.error")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# --- Constants ---
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8001")
POOL_SIZE = int(os.getenv("PROBLEM_POOL_SIZE", "5"))

try:
    CACHE_TTL = float(os.getenv("CACHE_TTL", "60.0"))
    CACHE_BACKOFF = float(os.getenv("CACHE_BACKOFF", "15.0"))
except ValueError as e:
    logger.error("Failed to parse cache configuration.", exc_info=True)
    raise ValueError("CACHE_TTL and CACHE_BACKOFF must be numeric values in seconds.") from e


# --- Helper Functions ---
def paraboloid_func(x: float, y: float) -> float:
    """f(x, y) = (x-3)**2 + xy + (y+4)**2 - 3"""
    return (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0


def build_and_init(schema: dict[str, Any], registry: dict[str, Callable[..., Any]]) -> Any:
    """Instantiates builder and creates problem in a worker thread."""
    return GraphProblemBuilder(schema).build_problem(registry)


def execute_problem(prob: Any, inputs: dict[str, float], objective: str) -> Any:
    """Sets values and executes the model in a worker thread."""
    for name, val in inputs.items():
        prob.set_val(name, val)
    prob.run_model()
    return prob.get_val(objective)


def to_float(val: Any) -> float:
    """Efficiently converts result to a Python float."""
    if isinstance(val, (float, int)):
        return float(val)
    return float(np.asarray(val).flat[0])


# --- Tool Registry ---
ToolRegistry: TypeAlias = dict[str, Callable[..., Any]]
TOOL_REGISTRY: ToolRegistry = {"Paraboloid": paraboloid_func}


# --- Domain Models ---
class SchemaEnvelope:
    """Wraps raw schema data with pre-parsed metadata and hashing."""

    def __init__(self, raw_data: dict[str, Any]):
        self.data = raw_data
        try:
            variables = raw_data.get("variables", [])
            self.known_vars = {v["name"] for v in variables}

            self.known_objectives: set[str] = set()
            for tool in raw_data.get("tools", []):
                for out in tool.get("outputs", []):
                    if isinstance(out, dict):
                        self.known_objectives.add(out["name"])
                    else:
                        self.known_objectives.add(str(out))
        except (KeyError, TypeError) as e:
            logger.error("Failed to parse schema structure.", exc_info=True)
            raise ValueError("Schema format is invalid.") from e

        # Stable hash for memoization
        # Note: In production, consider using a faster/more stable serializer like orjson
        serialized = json.dumps(raw_data, sort_keys=True)
        self.hash = hashlib.sha256(serialized.encode()).hexdigest()


# --- Providers ---
class SchemaProvider:
    """Manages schema fetching and caching from the Graph Service."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.envelope: SchemaEnvelope | None = None
        self.expiry: float = 0.0
        self.lock = asyncio.Lock()

    async def get_schema(self) -> SchemaEnvelope:
        current_time = time.time()
        if self.envelope is not None and current_time <= self.expiry:
            return self.envelope

        try:
            # Idiomatic lock acquisition with timeout
            async with asyncio.timeout(1.5):
                async with self.lock:
                    # Re-check inside lock
                    current_time = time.time()
                    if self.envelope is not None and current_time <= self.expiry:
                        return self.envelope

                    try:
                        resp = await self.client.get(f"{GRAPH_SERVICE_URL}/schema")
                        resp.raise_for_status()
                        self.envelope = SchemaEnvelope(resp.json())
                        self.expiry = current_time + CACHE_TTL
                    except (httpx.RequestError, httpx.HTTPStatusError):
                        if self.envelope is not None:
                            self.expiry = current_time + CACHE_BACKOFF
                            logger.warning(
                                "Schema refresh failed, serving stale cache.",
                                exc_info=True,
                            )
                        else:
                            logger.error("Graph Service unavailable.", exc_info=True)
                            raise HTTPException(
                                status_code=503, detail="Graph Service unavailable."
                            )
                    except ValueError:
                        # Malformed schema data
                        logger.error("Graph Service returned an invalid schema.", exc_info=True)
                        if self.envelope is not None:
                            self.expiry = current_time + CACHE_BACKOFF
                            logger.warning("Serving stale cache due to malformed schema response.")
                        else:
                            raise HTTPException(
                                status_code=502, detail="Graph Service returned an invalid schema."
                            )
        except TimeoutError:
            logger.warning("Schema lock timeout, serving stale cache.", exc_info=True)
            if self.envelope is None:
                raise HTTPException(status_code=503, detail="Schema unavailable.")

        return self.envelope


class ProblemPool:
    """Manages a pool of OpenMDAO Problem instances for a given schema."""

    def __init__(self, registry: ToolRegistry, size: int = POOL_SIZE):
        self.registry = registry
        self.size = size
        self.pool = asyncio.Queue()
        self.current_hash: str | None = None
        self.lock = asyncio.Lock()

    async def get_instance(self, envelope: SchemaEnvelope) -> tuple[Any, str]:
        """Retrieves an instance and the hash it was built for."""
        async with self.lock:
            if self.current_hash != envelope.hash:
                logger.info("Schema hash changed (%s), rebuilding Problem pool.", envelope.hash)
                # Clear existing pool
                while not self.pool.empty():
                    try:
                        self.pool.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                # Populate new pool
                tasks = [
                    asyncio.to_thread(build_and_init, envelope.data, self.registry)
                    for _ in range(self.size)
                ]
                instances = await asyncio.gather(*tasks)
                for inst in instances:
                    await self.pool.put(inst)
                self.current_hash = envelope.hash

        inst = await self.pool.get()
        return inst, envelope.hash

    async def release_instance(self, instance: Any, instance_hash: str):
        """Returns an instance to the pool only if it matches the current schema."""
        async with self.lock:
            if instance_hash == self.current_hash:
                await self.pool.put(instance)
            else:
                logger.info("Discarding stale Problem instance (hash mismatch).")


# --- Dependencies ---
async def get_schema_provider(request: Request) -> SchemaProvider:
    return request.app.state.schema_provider


async def get_problem_pool(request: Request) -> ProblemPool:
    return request.app.state.problem_pool


# --- Request Models ---
class EvaluateRequest(BaseModel):
    inputs: dict[str, float]
    objective: str = Field(..., min_length=1, max_length=100)

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("At least one input is required.")
        if len(v) > 100:
            raise ValueError("Too many inputs (max 100 allowed).")
        for key in v:
            if len(key) > 50:
                preview = key[:20]
                raise ValueError(f"Input key '{preview}...' exceeds maximum length of 50.")
        return v


# --- App Setup ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # Startup validation
    if GRAPH_SERVICE_URL == "http://localhost:8001":
        logger.warning("GRAPH_SERVICE_URL is using the default localhost value.")

    client = httpx.AsyncClient(timeout=5.0)
    app_instance.state.schema_provider = SchemaProvider(client)
    app_instance.state.problem_pool = ProblemPool(TOOL_REGISTRY)
    yield
    await client.aclose()


app = FastAPI(title="Execution Service", lifespan=lifespan)


# --- Endpoints ---
@app.post("/evaluate")
async def evaluate(
    req: EvaluateRequest,
    schema_p: SchemaProvider = Depends(get_schema_provider),
    problem_pool: ProblemPool = Depends(get_problem_pool),
):
    """
    Evaluate an objective function using the graph-defined problem structure.
    """
    envelope = await schema_p.get_schema()

    # 1. Validation against Schema
    if req.objective not in envelope.known_objectives:
        raise HTTPException(status_code=422, detail=f"Unknown objective: {req.objective}")

    unknown = set(req.inputs.keys()) - envelope.known_vars
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown inputs: {unknown}")

    # 2. Execution from Pool
    instance, instance_hash = await problem_pool.get_instance(envelope)
    try:
        try:
            # Offload synchronous math to thread
            raw_result = await asyncio.to_thread(
                execute_problem, instance, req.inputs, req.objective
            )

            # 3. Safe result transformation
            try:
                result = to_float(raw_result)
                return {"result": result}
            except (IndexError, TypeError, ValueError) as e:
                logger.error("Result transformation failed.", exc_info=True)
                raise HTTPException(status_code=500, detail="Invalid result shape.") from e

        except HTTPException:
            # Re-raise explicit HTTPExceptions before the generic catch
            raise
        except (ValueError, KeyError) as e:
            # Input-related errors
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            # General execution failures
            logger.error("Execution failed.", exc_info=True)
            raise HTTPException(status_code=500, detail="An internal execution error occurred.")
    finally:
        # Always return instance to pool (if still valid)
        await problem_pool.release_instance(instance, instance_hash)


@app.get("/health")
async def health(request: Request):
    """Check connectivity to Graph Service."""
    try:
        resp = await request.app.state.schema_provider.client.get(f"{GRAPH_SERVICE_URL}/health")
        resp.raise_for_status()
        return {"status": "ok", "graph_service": "ok"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "graph_service": str(e)},
        )
