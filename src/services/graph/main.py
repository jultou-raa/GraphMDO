"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from mdo_framework.db.client import FalkorDBClient
from mdo_framework.db.graph_manager import (
    DuplicateProducerError,
    GraphManager,
    NodeExistsError,
    NodeNotFoundError,
    RoleConflictError,
)
from mdo_framework.schema import (
    StudySchema,
    StudyValidationError,
    ToolNode,
    Variable,
)

from services.errors import register_validation_handler

app = FastAPI(title="Graph Service")
register_validation_handler(app)


def get_graph_manager() -> GraphManager:
    return GraphManager()


def _error(status_code: int, detail: object) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.exception_handler(NodeNotFoundError)
def node_not_found_handler(_: Request, exc: NodeNotFoundError) -> JSONResponse:
    missing = [{"label": label, "name": name} for label, name in exc.missing]
    return _error(404, {"missing": missing, "hint": exc.hint})


@app.exception_handler(NodeExistsError)
def node_exists_handler(_: Request, exc: NodeExistsError) -> JSONResponse:
    return _error(409, {"error": "exists", "label": exc.label, "name": exc.name})


@app.exception_handler(DuplicateProducerError)
def duplicate_producer_handler(_: Request, exc: DuplicateProducerError) -> JSONResponse:
    return _error(
        409,
        {
            "error": "duplicate_producer",
            "variable": exc.variable,
            "producers": list(exc.producers),
        },
    )


@app.exception_handler(RoleConflictError)
def role_conflict_handler(_: Request, exc: RoleConflictError) -> JSONResponse:
    return _error(
        409,
        {"error": "role_conflict", "variable": exc.variable, "message": exc.message},
    )


@app.exception_handler(StudyValidationError)
def study_validation_handler(_: Request, exc: StudyValidationError) -> JSONResponse:
    return _error(409, exc.report.model_dump(mode="json"))


class ConnectionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str


class StatusResponse(BaseModel):
    status: str


class VariableResponse(StatusResponse):
    variable: str


class ToolResponse(StatusResponse):
    tool: str


class ConnectionResponse(StatusResponse):
    type: str


def _check_path_name(path_name: str, body_name: str) -> None:
    if path_name != body_name:
        raise HTTPException(
            status_code=422,
            detail=(f"Path name '{path_name}' does not match body name '{body_name}'"),
        )


@app.post("/clear", response_model=StatusResponse)
def clear_graph(gm: GraphManager = Depends(get_graph_manager)):
    gm.clear_graph()
    return StatusResponse(status="cleared")


@app.post("/variables", response_model=VariableResponse, status_code=201)
def create_variable(var: Variable, gm: GraphManager = Depends(get_graph_manager)):
    gm.add_variable(var)
    return VariableResponse(status="created", variable=var.name)


@app.put("/variables/{name}", response_model=VariableResponse)
def put_variable(
    name: str,
    var: Variable,
    response: Response,
    gm: GraphManager = Depends(get_graph_manager),
):
    _check_path_name(name, var.name)
    created = gm.put_variable(var)
    response.status_code = 201 if created else 200
    return VariableResponse(
        status="created" if created else "replaced", variable=var.name
    )


@app.delete("/variables/{name}", response_model=VariableResponse)
def delete_variable(name: str, gm: GraphManager = Depends(get_graph_manager)):
    gm.delete_variable(name)
    return VariableResponse(status="deleted", variable=name)


@app.post("/tools", response_model=ToolResponse, status_code=201)
def create_tool(tool: ToolNode, gm: GraphManager = Depends(get_graph_manager)):
    gm.add_tool(tool)
    return ToolResponse(status="created", tool=tool.name)


@app.put("/tools/{name}", response_model=ToolResponse)
def put_tool(
    name: str,
    tool: ToolNode,
    response: Response,
    gm: GraphManager = Depends(get_graph_manager),
):
    _check_path_name(name, tool.name)
    created = gm.put_tool(tool)
    response.status_code = 201 if created else 200
    return ToolResponse(status="created" if created else "replaced", tool=tool.name)


@app.delete("/tools/{name}", response_model=ToolResponse)
def delete_tool(name: str, gm: GraphManager = Depends(get_graph_manager)):
    gm.delete_tool(name)
    return ToolResponse(status="deleted", tool=name)


@app.post("/connections/input", response_model=ConnectionResponse)
def connect_input(
    conn: ConnectionCreate,
    gm: GraphManager = Depends(get_graph_manager),
):
    # Variable -> Tool
    gm.connect_input_to_tool(conn.source, conn.target)
    return ConnectionResponse(status="connected", type="input")


@app.post("/connections/output", response_model=ConnectionResponse)
def connect_output(
    conn: ConnectionCreate,
    gm: GraphManager = Depends(get_graph_manager),
):
    # Tool -> Variable
    gm.connect_tool_to_output(conn.source, conn.target)
    return ConnectionResponse(status="connected", type="output")


@app.get("/schema", response_model=StudySchema)
def get_schema(gm: GraphManager = Depends(get_graph_manager)):
    return gm.get_study_schema()


def ping_database() -> None:
    """Raise if FalkorDB does not answer a PING."""
    FalkorDBClient().client.connection.ping()


@app.get("/health")
def health():
    """Report 200 when FalkorDB answers, 503 otherwise."""
    try:
        ping_database()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "falkordb": str(e)},
        )
    return {"status": "ok", "falkordb": "ok"}
