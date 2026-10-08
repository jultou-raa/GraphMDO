"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mdo_framework.optimization.ax_algo_lib import AxObjectiveDict
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
register_validation_handler(app)

EXECUTION_SERVICE_URL = os.getenv("EXECUTION_SERVICE_URL", "http://localhost:8002")
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8001")


def objective_to_ax(objective: ObjectiveSpec) -> AxObjectiveDict:
    """Projects an objective onto the Ax objective contract."""
    ax_objective: AxObjectiveDict = {
        "name": objective.name,
        "minimize": objective.minimize,
    }
    if objective.threshold is not None:
        ax_objective["threshold"] = objective.threshold
    return ax_objective


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
            "Initial Sobol trials. The start point x0 is evaluated in addition, "
            "so tools are called at most 1 + n_init + n_steps times."
        ),
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

    # 3. Extract parameter definitions
    parameters = analyzer.extract_parameters(resolved.design_variables)

    # 4. Setup Evaluator
    evaluator = RemoteEvaluator(EXECUTION_SERVICE_URL)

    # 5. Setup and run the optimizer
    try:
        constraints = (
            [c.model_dump() for c in req.constraints] if req.constraints else None
        )
        try:
            optimizer = BayesianOptimizer(
                evaluator=evaluator,
                parameters=parameters,
                objectives=[objective_to_ax(o) for o in req.objectives],
                constraints=constraints,
                use_bonsai=req.use_bonsai,
                parameter_constraints=req.parameter_constraints,
            )

            # Offload to a thread to avoid blocking the event loop
            result = await asyncio.to_thread(
                optimizer.optimize,
                n_steps=req.n_steps,
                n_init=req.n_init,
            )
        except OptimizationConfigurationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except (RemoteEvaluationTransportError, RemoteEvaluationContractError) as e:
            raise HTTPException(status_code=502, detail=str(e))
        except OptimizationExecutionError as e:
            raise HTTPException(status_code=500, detail=str(e))

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")
    finally:
        evaluator.close()


@app.get("/health")
def health():
    return {"status": "ok"}
