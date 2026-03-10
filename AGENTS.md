# AGENTS.md

## Strict Coding Standards

1.  **PEP 20 (The Zen of Python)**
    -   Explicit is better than implicit.
    -   Simple is better than complex.
    -   Readability counts.

2.  **PEP 8 (Style Guide for Python Code)**
    -   Indentation: 4 spaces.
    -   Line Length: 88 characters.
    -   Naming Conventions:
        -   Functions/Variables: `lowercase_with_underscores`
        -   Classes: `CapitalizedWords`
        -   Constants: `ALL_CAPS_WITH_UNDERSCORES`
    -   Imports: Standard library, third-party, local application.
    -   Type Hinting: Use explicit Python 3 type hints.

3.  **Typing and Interfaces**
    -   Target Python 3.12+ syntax (`X | None`, `list[str]`, `dict[str, Any]`).
    -   Keep FastAPI request/response contracts explicit with Pydantic models.
    -   Prefer small, composable functions over hidden side effects.

## Current Codebase Overview

-   `main.py` is the local paraboloid demo wiring `GraphManager` -> `GraphProblemBuilder` -> `LocalEvaluator` -> `BayesianOptimizer`.
-   `src/mdo_framework/db/` contains the FalkorDB integration (`client.py`, `graph_manager.py`).
-   `src/mdo_framework/core/` contains graph-to-GEMSEO translation and execution helpers (`components.py`, `evaluators.py`, `surrogates.py`, `topology.py`, `translator.py`).
-   `src/mdo_framework/optimization/` contains the optimizer orchestration (`optimizer.py`) and the Ax-backed algorithm library (`ax_algo_lib.py`).
-   `src/services/graph/main.py` exposes the Graph Service API: `/clear`, `/variables`, `/tools`, `/connections/input`, `/connections/output`, `/schema`.
-   `src/services/execution/main.py` exposes the Execution Service API: `/evaluate`, `/health`, plus schema caching and pooled problem instances.
-   `src/services/optimization/main.py` exposes the Optimization Service API: `/optimize`, `/health`.
-   `tests/` covers the core modules, services, database layer, optimizer, topology, translator, and the top-level demo entry point.

## Runtime Architecture

1.  **Graph Layer**
    -   FalkorDB stores variables, tools, and directed data-flow edges.
    -   `GraphManager.get_graph_schema()` is the canonical boundary exported to the rest of the system.

2.  **Translation Layer**
    -   `GraphProblemBuilder` builds GEMSEO problems from the graph schema.
    -   `TopologicalAnalyzer` resolves dependencies and extracts optimization parameters from requested outputs.

3.  **Evaluation Layer**
    -   Local execution uses `LocalEvaluator`.
    -   Remote execution uses the Execution Service, which maintains a `SchemaProvider` cache and a `ProblemPool` of initialized GEMSEO problems.

4.  **Optimization Layer**
    -   `BayesianOptimizer` orchestrates Ax/GEMSEO optimization.
    -   `ax_algo_lib.py` maintains explicit `trial_history` records and integrates constrained optimization behavior.

5.  **Service Deployment**
    -   `docker-compose.yml` runs FalkorDB plus three FastAPI services.
    -   Default ports are 8001 (graph), 8002 (execution), and 8003 (optimization).

## Implementation Directives

-   **Graph Schema Is the Source of Truth**: Flow data from FalkorDB through `get_graph_schema()` into topology analysis, translation, and services.
-   **Preserve Design Variable Order**: Keep FalkorDB insertion order for design variables; do not sort parameter names alphabetically before execution or optimization.
-   **Use Keyword-Based Tool Invocation**: Wrapped tool functions must receive named inputs, not positional fallbacks that can scramble graph-defined ordering.
-   **Keep Optimization State Explicit**: Use `problem.optimum` and `trial_history` as the authoritative optimization outputs; avoid hidden cross-object attributes.
-   **Respect Constraint Semantics**: Current optimization code uses GEMSEO/Ax convention `g(x) <= 0`; the paraboloid example encodes `c_xy = x - y`.
-   **Extend Service Infrastructure, Do Not Bypass It**: Schema refresh/backoff belongs in `SchemaProvider`; reusable GEMSEO instances belong in `ProblemPool`.
-   **Preserve Service Boundaries**: Cross-service calls should flow through `GRAPH_SERVICE_URL` and `EXECUTION_SERVICE_URL`, matching local and Docker Compose deployment.

## Dependency Management

-   This project uses `uv` for dependency management.
-   Core runtime stack includes FalkorDB, FastAPI, GEMSEO, SMT, Ax Platform, BoTorch, pymoo, httpx, and NumPy/SciPy.
-   Install project dependencies with `uv sync`.
-   Install development dependencies with `uv sync --all-extras --dev`.
-   Add a dependency with `uv add <package_name>`.
-   Run commands in the environment with `uv run <command>`.
-   **Do not use pip install manually.**

## Development Commands

-   Run the local demo: `uv run python main.py`
-   Start the Graph Service: `uv run uvicorn services.graph.main:app --host 0.0.0.0 --port 8001`
-   Start the Execution Service: `uv run uvicorn services.execution.main:app --host 0.0.0.0 --port 8002`
-   Start the Optimization Service: `uv run uvicorn services.optimization.main:app --host 0.0.0.0 --port 8003`
-   Start the full stack with containers: `docker compose up --build`

## Validation and Docs

-   Run tests with `uv run pytest tests/`.
-   Run lint checks with `uv run ruff check .`.
-   Format code with `uv run ruff format .`.
-   Serve documentation locally with `uv run mkdocs serve`.
-   Documentation lives under `docs/` and is built with MkDocs Material.
