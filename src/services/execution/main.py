"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, TypeAlias

import httpx
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from mdo_framework.core.topology import build_variable_specs
from mdo_framework.core.translator import GraphProblemBuilder, encode_tool_inputs
from mdo_framework.schema import MAX_NAME_LENGTH, StudySchema

# Configure logging
logger = logging.getLogger("uvicorn.error")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# --- Constants ---
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8001")

try:
    CACHE_TTL = float(os.getenv("CACHE_TTL", "60.0"))
    CACHE_BACKOFF = float(os.getenv("CACHE_BACKOFF", "15.0"))
    POOL_SIZE = int(os.getenv("PROBLEM_POOL_SIZE", "5"))
    POOL_ACQUIRE_TIMEOUT = float(os.getenv("POOL_ACQUIRE_TIMEOUT", "5.0"))
except ValueError as e:
    logger.error("Failed to parse configuration.", exc_info=True)
    raise ValueError(
        "CACHE_TTL, CACHE_BACKOFF, PROBLEM_POOL_SIZE, and POOL_ACQUIRE_TIMEOUT must be numeric.",
    ) from e

if POOL_SIZE <= 0:
    raise ValueError("PROBLEM_POOL_SIZE must be a positive integer.")
if CACHE_TTL <= 0 or CACHE_BACKOFF <= 0 or POOL_ACQUIRE_TIMEOUT <= 0:
    raise ValueError("Timeout and TTL values must be positive.")


# --- Helper Functions ---
def paraboloid_func(x: float, y: float) -> float:
    """f(x, y) = (x-3)**2 + xy + (y+4)**2 - 3"""
    return (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0


def build_and_init(
    schema: StudySchema,
    registry: dict[str, Callable[..., Any]],
) -> Any:
    """Instantiates builder and creates problem in a worker thread."""
    return GraphProblemBuilder(schema).build_problem(registry)


def execute_problem(
    prob,
    inputs: dict[str, bool | int | float | str],
    objectives: list[str],
    variable_specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Executes the problem from user-facing inputs (choices given by value)."""
    input_data = encode_tool_inputs(inputs, variable_specs or {})
    out_data = prob.execute(input_data)
    return {
        obj: 0.0 if (val := out_data.get(obj)) is None else val for obj in objectives
    }


def to_float(val: Any) -> float:
    """Convert result to float. Raises IndexError if val is empty, TypeError if unconvertible."""
    if isinstance(val, (float, int)):
        return float(val)
    return float(np.asarray(val).flat[0])


# --- Tool Registry ---
ToolRegistry: TypeAlias = dict[str, Callable[..., Any]]
TOOL_REGISTRY: ToolRegistry = {"Paraboloid": paraboloid_func}
InputScalar: TypeAlias = bool | int | float | str


# --- Domain Models ---
class SchemaEnvelope:
    """Wraps a study schema with pre-computed metadata and its content hash."""

    def __init__(self, schema: StudySchema):
        self.schema = schema
        self.known_vars = {variable.name for variable in schema.variables}
        self.variable_specs = build_variable_specs(schema)
        self.known_objectives = set(schema.producers())
        self.hash = schema.content_hash()


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
        if self.envelope and current_time <= self.expiry:
            return self.envelope

        # IMPORTANT: The timeout MUST scope to lock acquisition only, NOT the
        # HTTP call. The HTTP client has its own 5.0s timeout. Wrapping both
        # in a single 1.5s ceiling makes the client timeout unreachable and
        # conflates slow-but-valid responses with lock starvation.
        try:
            async with asyncio.timeout(1.5):
                await self.lock.acquire()
        except TimeoutError:
            if self.envelope:
                logger.warning("Schema lock timeout, serving stale cache.")
                return self.envelope
            raise HTTPException(status_code=503, detail="Schema service lock timeout.")

        try:
            # Re-check inside lock (double-checked locking pattern).
            # The finally block guarantees release on all code paths.
            current_time = time.time()
            if self.envelope and current_time <= self.expiry:
                return self.envelope

            try:
                resp = await self.client.get(f"{GRAPH_SERVICE_URL}/schema")
                resp.raise_for_status()
                self.envelope = SchemaEnvelope(StudySchema.model_validate(resp.json()))
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
                        status_code=503,
                        detail="Graph Service unavailable.",
                    )
            except ValueError:
                # Malformed JSON or a payload that is not a valid StudySchema
                logger.error("Graph Service returned an invalid schema.", exc_info=True)
                if self.envelope is not None:
                    self.expiry = current_time + CACHE_BACKOFF
                    logger.warning(
                        "Serving stale cache due to malformed schema response.",
                    )
                else:
                    raise HTTPException(
                        status_code=502,
                        detail="Graph Service returned an invalid schema.",
                    )
        finally:
            self.lock.release()

        return self.envelope


class ProblemPool:
    """Manages a pool of GEMSEO OptimizationProblem instances for a given schema."""

    def __init__(self, registry: ToolRegistry, size: int = POOL_SIZE):
        self.registry = registry
        self.size = size
        self.pool = asyncio.Queue()
        self.current_hash: str | None = None
        self.lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task] = set()

    async def teardown(self):
        """Drain and clean up all pool instances.
        Note: This method is not independently thread-safe and is intended to be called
        either under self.lock (e.g., in get_instance) or after all requests have completed (e.g., in lifespan).
        """
        tasks = list(self._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        while not self.pool.empty():
            try:
                inst = self.pool.get_nowait()
                if hasattr(inst, "cleanup"):
                    await asyncio.to_thread(inst.cleanup)
            except asyncio.QueueEmpty:
                break

    async def _replenish_one(self, schema_hash: str, schema: StudySchema):
        """Builds one replacement instance if the schema hasn't changed."""
        try:
            inst = await asyncio.to_thread(build_and_init, schema, self.registry)
            async with self.lock:
                if self.current_hash == schema_hash:
                    await self.pool.put(inst)
        except Exception:
            logger.warning("Failed to replenish pool instance.", exc_info=True)

    async def discard_instance(self, instance: Any, envelope: SchemaEnvelope) -> None:
        """Explicitly drops an instance without returning it to the pool and spawns a replacement."""
        logger.warning("Discarding instance (not returned to pool).")
        if hasattr(instance, "cleanup"):
            await asyncio.to_thread(instance.cleanup)

        task = asyncio.create_task(self._replenish_one(envelope.hash, envelope.schema))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def get_instance(self, envelope: SchemaEnvelope) -> tuple[Any, str]:
        """Retrieves an instance and the hash it was built for."""
        async with self.lock:
            if self.current_hash != envelope.hash:
                logger.info("Schema change detected. Rebuilding pool.")
                # Clear pool
                await self.teardown()

                # Populate new pool
                # Note: All incoming requests queue on self.lock while the pool rebuilds.
                # This could be slow if build_and_init is expensive, but it's a deliberate
                # serialization point to prevent concurrent rebuilds.
                tasks = [
                    asyncio.to_thread(build_and_init, envelope.schema, self.registry)
                    for _ in range(self.size)
                ]

                # Ensure we don't leak successful instances if some fail
                results = await asyncio.gather(*tasks, return_exceptions=True)

                valid_instances = [r for r in results if not isinstance(r, Exception)]
                failed = [r for r in results if isinstance(r, Exception)]
                for err in failed:
                    logger.warning("Pool instance failed to build.", exc_info=err)

                for inst in valid_instances:
                    await self.pool.put(inst)

                if not valid_instances:
                    self.current_hash = None  # Reset so we try again next time
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to build any execution instances.",
                    )

                self.current_hash = envelope.hash

        try:
            async with asyncio.timeout(POOL_ACQUIRE_TIMEOUT):
                inst = await self.pool.get()
        except TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="No execution instances available.",
            )

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
    inputs: dict[str, InputScalar]
    objectives: list[str] = Field(..., min_length=1)

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, v: dict[str, InputScalar]) -> dict[str, InputScalar]:
        if not v:
            raise ValueError("At least one input is required.")
        if len(v) > 100:
            raise ValueError("Too many inputs (max 100 allowed).")
        for key in v:
            if len(key) > MAX_NAME_LENGTH:
                preview = key[:20]
                raise ValueError(
                    f"Input key '{preview}...' exceeds maximum length of "
                    f"{MAX_NAME_LENGTH}.",
                )
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
    await app_instance.state.problem_pool.teardown()
    await client.aclose()


app = FastAPI(title="Execution Service", lifespan=lifespan)


# --- Endpoints ---
@app.post("/evaluate")
async def evaluate(
    req: EvaluateRequest,
    schema_p: SchemaProvider = Depends(get_schema_provider),
    problem_pool: ProblemPool = Depends(get_problem_pool),
):
    """Evaluate an objective function using the graph-defined problem structure."""
    envelope = await schema_p.get_schema()

    # 1. Validation against Schema
    for obj in req.objectives:
        if obj not in envelope.known_objectives:
            raise HTTPException(status_code=422, detail=f"Unknown objective: {obj}")

    unknown = set(req.inputs.keys()) - envelope.known_vars
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown inputs: {unknown}")

    # 2. Execution from Pool
    instance, instance_hash = await problem_pool.get_instance(envelope)
    execution_succeeded = False
    try:
        try:
            # Offload synchronous math to thread
            raw_results = await asyncio.to_thread(
                execute_problem,
                instance,
                req.inputs,
                req.objectives,
                envelope.variable_specs,
            )
            execution_succeeded = True

            # 3. Safe result transformation
            try:
                results = {obj: to_float(val) for obj, val in raw_results.items()}
                return {"results": results}
            except (IndexError, TypeError, ValueError) as e:
                logger.error("Result transformation failed.", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail="Invalid result shape.",
                ) from e

        except HTTPException:
            # Re-raise explicit HTTPExceptions before the generic catch
            raise
        except (ValueError, KeyError) as e:
            # Input-related errors
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            # General execution failures
            logger.error("Execution failed.", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="An internal execution error occurred.",
            )
    finally:
        if execution_succeeded:
            await problem_pool.release_instance(instance, instance_hash)
        else:
            await problem_pool.discard_instance(instance, envelope)


@app.get("/health")
async def health(request: Request):
    """Check connectivity to Graph Service."""
    try:
        resp = await request.app.state.schema_provider.client.get(
            f"{GRAPH_SERVICE_URL}/health",
        )
        resp.raise_for_status()
        return {"status": "ok", "graph_service": "ok"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "graph_service": str(e)},
        )
