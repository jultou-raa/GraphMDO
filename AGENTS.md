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
-   `src/mdo_framework/schema.py` defines the typed `StudySchema` contract (`RangeVar`, `ChoiceVar`, `FixedParam`, `StateVar`, `ToolNode`, `Finding`, `ValidationReport`, `StudyValidationError`).
-   `src/mdo_framework/validation.py` holds `validate_study()` (preflight against a request) and `validate_registry()` (tool registry and signature checks); `core/dependencies.py` is the shared dependency walk.
-   `src/mdo_framework/db/` contains the FalkorDB integration (`client.py`, `graph_manager.py`).
-   `src/mdo_framework/core/` contains schema-to-GEMSEO translation and execution helpers (`components.py`, `dependencies.py`, `errors.py`, `evaluators.py`, `mda.py`, `surrogates.py`, `topology.py`, `translator.py`).
-   `src/mdo_framework/optimization/` contains the optimizer orchestration (`optimizer.py`) and the Ax-backed algorithm library (`ax_algo_lib.py`).
-   `src/services/graph/main.py` exposes the typed Graph Service API: `POST/PUT/DELETE` on `/variables` and `/tools`, `/connections/input`, `/connections/output`, `/schema`, `/clear`, `/health`. Conflicts return `409`, unknown nodes `404`, invalid bodies `422`.
-   `src/services/execution/main.py` exposes the Execution Service API: `/evaluate`, `/health`, plus schema caching, a registry check on each loaded schema (`422` `SCHEMA_INVALID`), and pooled problem instances.
-   `src/services/optimization/main.py` exposes the Optimization Service API: `/optimize` (with a `422` validation preflight), `/validate`, `/health`.
-   `src/services/errors.py` registers the shared request-validation handler (`422` without echoing input) on all three services.
-   `tests/` covers the core modules, services, database layer, optimizer, topology, translator, and the top-level demo entry point.
-   `tests/e2e/` holds the seeded, non-mocked Ax + GEMSEO regression suite (marker `e2e`); open bugs are pinned there as strict xfails.

## Runtime Architecture

1.  **Graph Layer**
    -   FalkorDB stores variables, tools, and directed data-flow edges.
    -   `GraphManager.get_study_schema()` returns the typed `StudySchema`, the canonical boundary exported to the rest of the system.

2.  **Translation Layer**
    -   `GraphProblemBuilder` builds GEMSEO problems from the `StudySchema`: one `ToolComponent` per tool (strict output contract, finite-difference Jacobians) chained in a `StrictMDAChain` (`core/mda.py`).
    -   Coupled tools run sequentially (Gauss-Seidel) by default; `MDASettings` selects the algorithm, and a coupled group that does not converge raises `MDANotConvergedError`.
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

-   **StudySchema Is the Source of Truth**: Flow data from FalkorDB through `get_study_schema()` into validation, topology analysis, translation, and services. Do not reintroduce untyped dict schemas.
-   **Validate Before Running**: Study-level checks belong in `validate_study()` and tool-registry checks in `validate_registry()`; add new findings there with a stable error code and document them in `docs/technical-reference/study-schema.md`.
-   **Keep the Published JSON Schema in Sync**: After changing `schema.py`, regenerate `docs/technical-reference/study-schema.json` with the command printed by `tests/test_docs_schema.py`.
-   **Preserve Design Variable Order**: Keep FalkorDB insertion order for design variables; do not sort parameter names alphabetically before execution or optimization.
-   **Use Keyword-Based Tool Invocation**: Wrapped tool functions must receive named inputs, not positional fallbacks that can scramble graph-defined ordering.
-   **Raise Typed Evaluation Errors**: A point that cannot be evaluated raises an `EvaluationError` subclass from `core/errors.py` (`ToolExecutionError`, `ToolOutputError`, `MDANotConvergedError`), never a bare exception or a silently returned invalid value; services serialize them with `to_payload()`.
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

-   Run all tests with `uv run pytest tests/`.
-   Run fast unit tests with `uv run pytest -m "not e2e" tests/`.
-   Run the real Ax + GEMSEO suite with `OMP_NUM_THREADS=1 uv run pytest -m e2e tests/`.
-   Tests that pin an open bug are `xfail(strict=True)` with the issue number in the
    reason; the PR that fixes the bug removes the marker.
-   Run lint checks with `uv run ruff check .`.
-   Format code with `uv run ruff format .`.
-   Serve documentation locally with `uv run mkdocs serve`.
-   Documentation lives under `docs/` and is built with MkDocs Material.
