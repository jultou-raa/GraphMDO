"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import asyncio
import math
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mdo_framework.optimization.optimizer import (
    BayesianOptimizer,
    OptimizationConfigurationError,
    OptimizationExecutionError,
    RemoteEvaluationContractError,
    RemoteEvaluationTransportError,
    RemoteEvaluator,
)
from mdo_framework.schema import (
    ConstraintSpec,
    ObjectiveSpec,
    StudySchema,
    StudyValidationError,
    ValidationReport,
    report_from_validation_error,
)
from mdo_framework.validation import validate_study
from services.errors import register_validation_handler


def to_jsonable(obj: Any) -> Any:
    """Recursively converts objects to JSON-serializable types.

    NumPy values and tensors become Python values; non-finite floats become
    ``None`` because JSON cannot carry them.
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if hasattr(obj, "tolist") and callable(obj.tolist):
        # Handle PyTorch tensors and other objects with .tolist()
        return to_jsonable(obj.tolist())
    if hasattr(obj, "item") and callable(obj.item):
        # Handle scalars from Tensors/NumPy
        return to_jsonable(obj.item())
    return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the shared HTTP client
    async with httpx.AsyncClient() as client:
        app.state.client = client
        yield


app = FastAPI(title="Optimization Service", lifespan=lifespan)
register_validation_handler(app)

EXECUTION_SERVICE_URL = os.getenv("EXECUTION_SERVICE_URL", "http://localhost:8002")
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8001")


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objectives: list[ObjectiveSpec] = Field(min_length=1)
    constraints: list[ConstraintSpec] | None = None
    fidelity_parameter: str | None = Field(
        default=None,
        description="Reserved for multi-fidelity optimization; not supported yet.",
    )
    parameter_constraints: list[str] | None = None
    n_steps: int = Field(
        default=10,
        ge=1,
        description="Bayesian (BoTorch) iterations after the initial design.",
    )
    n_init: int = Field(
        default=5,
        ge=1,
        description=(
            "Initial Sobol trials. Tools are called at most "
            "n_init + n_steps times, plus one when the start point x0 is evaluated."
        ),
    )
    evaluate_x0: bool | None = Field(
        default=None,
        description=(
            "Evaluate the start point x0 as a baseline trial. By default x0 is "
            "evaluated only when every design variable declares an initial value."
        ),
    )
    max_consecutive_failures: int = Field(
        default=5,
        ge=1,
        description="Stop the run after this many failed trials in a row.",
    )
    use_bonsai: bool = False

    @field_validator("fidelity_parameter")
    @classmethod
    def reject_fidelity_parameter(cls, value: str | None) -> str | None:
        if value is not None:
            raise ValueError("Multi-fidelity optimization is not supported yet.")
        return value


async def _fetch_schema(request: Request) -> Any:
    """Fetches the raw study schema from the Graph Service.

    Every upstream failure is a bad gateway: transport errors, non-2xx
    statuses and a 2xx body that is not JSON. A JSON body that is not a valid
    study is left to the preflight, which reports it.
    """
    client: httpx.AsyncClient = request.app.state.client
    try:
        resp = await client.get(f"{GRAPH_SERVICE_URL}/schema")
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch graph schema: {e}",
        )
    try:
        return resp.json()
    except ValueError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Graph service returned a response that is not JSON: {e}",
        )


def _preflight(
    payload: Any, req: OptimizeRequest
) -> tuple[StudySchema | None, ValidationReport]:
    """Parses and validates the study; the schema is ``None`` if it cannot parse."""
    try:
        schema = StudySchema.model_validate(payload)
    except ValidationError as exc:
        return None, report_from_validation_error(exc)
    # The Execution Service owns the tool registry, so it is not checked here.
    report = validate_study(
        schema,
        objectives=req.objectives,
        constraints=req.constraints or (),
        parameter_constraints=req.parameter_constraints or (),
        registry=None,
    )
    return schema, report


@app.post("/validate", response_model=ValidationReport)
async def validate(req: OptimizeRequest, request: Request):
    """Reports whether the study would run, without running any tool."""
    payload = await _fetch_schema(request)
    _, report = await asyncio.to_thread(_preflight, payload, req)
    return report


def _error_detail(error: Exception) -> Any:
    """The error message, with what the run produced before it failed if any."""
    partial_result = getattr(error, "partial_result", None)
    if partial_result is None:
        return str(error)
    return {"message": str(error), "partial_result": to_jsonable(partial_result)}


@app.post("/optimize")
async def optimize(req: OptimizeRequest, request: Request):
    # 1. Fetch the schema from the Graph Service and reject an invalid study
    payload = await _fetch_schema(request)
    schema, report = await asyncio.to_thread(_preflight, payload, req)
    if schema is None or not report.valid:
        raise HTTPException(status_code=422, detail=report.model_dump(mode="json"))

    from mdo_framework.core.topology import TopologicalAnalyzer

    # 2. Identify Design Variables recursively from requested objectives and constraints
    analyzer = TopologicalAnalyzer(schema)

    target_outputs = [obj.name for obj in req.objectives]
    if req.constraints:
        target_outputs.extend([c.name for c in req.constraints])

    try:
        resolved = analyzer.resolve_dependencies(target_outputs)
    except StudyValidationError as e:
        raise HTTPException(status_code=422, detail=e.report.model_dump(mode="json"))

    # 3. Setup Evaluator
    evaluator = RemoteEvaluator(EXECUTION_SERVICE_URL)

    # 4. Setup and run the optimizer
    try:
        try:
            optimizer = BayesianOptimizer(
                evaluator=evaluator,
                design_variables=resolved.design_variables,
                objectives=req.objectives,
                constraints=req.constraints or (),
                parameter_constraints=req.parameter_constraints or (),
                use_bonsai=req.use_bonsai,
            )

            # Offload to a thread to avoid blocking the event loop
            result = await asyncio.to_thread(
                optimizer.optimize,
                n_steps=req.n_steps,
                n_init=req.n_init,
                evaluate_x0=req.evaluate_x0,
                max_consecutive_failures=req.max_consecutive_failures,
            )
        except OptimizationConfigurationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except (RemoteEvaluationTransportError, RemoteEvaluationContractError) as e:
            raise HTTPException(status_code=502, detail=_error_detail(e))
        except OptimizationExecutionError as e:
            raise HTTPException(status_code=500, detail=_error_detail(e))

        return to_jsonable(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")
    finally:
        evaluator.close()


@app.get("/health")
def health():
    return {"status": "ok"}
