from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from mdo_framework.db.graph_manager import GraphManager

app = FastAPI(title="Graph Service")

# Initialize GraphManager
# In a real app, this might be a dependency injection
gm = GraphManager()


class VariableCreate(BaseModel):
    name: str
    value: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None


class ToolCreate(BaseModel):
    name: str
    fidelity: str = "high"


class ConnectionCreate(BaseModel):
    source: str
    target: str


@app.post("/clear")
def clear_graph():
    gm.clear_graph()
    return {"status": "cleared"}


@app.post("/variables")
def create_variable(var: VariableCreate):
    gm.add_variable(var.name, var.value, var.lower, var.upper)
    return {"status": "created", "variable": var.name}


@app.post("/tools")
def create_tool(tool: ToolCreate):
    gm.add_tool(tool.name, tool.fidelity)
    return {"status": "created", "tool": tool.name}


@app.post("/connections/input")
def connect_input(conn: ConnectionCreate):
    # Variable -> Tool
    gm.connect_input_to_tool(conn.source, conn.target)
    return {"status": "connected", "type": "input"}


@app.post("/connections/output")
def connect_output(conn: ConnectionCreate):
    # Tool -> Variable
    gm.connect_tool_to_output(conn.source, conn.target)
    return {"status": "connected", "type": "output"}


@app.get("/schema")
def get_schema():
    return gm.get_graph_schema()
