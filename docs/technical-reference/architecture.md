# System Architecture

GraphMDO is designed as a modular, service-oriented framework for Multidisciplinary Design Optimization.

## Core Components

The architecture consists of three primary layers:

1.  **Graph Layer (FalkorDB)**
    *   Stores the "Fundamental Problem Graph" (FPG).
    *   Nodes represent Variables (design, fixed, state) and Tools (Functions, Codes).
    *   Edges represent data flow (Inputs To, Outputs From).
    *   The graph is exported as a typed `StudySchema`, the contract between the layers (see [Study Schema](study-schema.md)).

2.  **Execution Layer (GEMSEO)**
    *   Translates the study schema into an executable GEMSEO Problem.
    *   Wraps Python functions or external codes into `ToolComponent`.
    *   Handles variable promotion and data passing between components.

3.  **Optimization Layer (Ax/SMT)**
    *   Drives the execution layer to minimize/maximize objectives.
    *   Uses Constrained Bayesian Optimization via Ax Platform (handling continuous, discrete, choices, and multi-objective definitions).
    *   Supports multi-fidelity surrogates (Co-Kriging) via SMT integration.

## The Study Schema Contract

`GraphManager.get_study_schema()` is the only way the graph leaves the Graph Layer. It returns a frozen Pydantic `StudySchema` (`mdo_framework.schema`) of typed variables (`RangeVar`, `ChoiceVar`, `FixedParam`, `StateVar`) and `ToolNode` tools plus the data-flow edges. Topology analysis, translation and all three services consume this one type; there is no untyped dictionary form.

Validation runs at three points, each reporting stable error codes:

| Where | Check |
| --- | --- |
| `StudySchema` construction | Structural rules: duplicate names, undeclared references, duplicate producers, produced design variables and fixed parameters. |
| Optimization Service (`POST /validate`, preflight of `POST /optimize`) | `validate_study(schema, request)`: objectives and constraints against the graph, dependency reachability, design space and parameter constraints. Invalid studies are rejected with `422` before any tool runs. |
| Execution Service (`POST /evaluate`) and `GraphProblemBuilder.build_problem` | `validate_registry`: every tool exists in the tool registry and its signature matches the wiring. |

The full list of codes is in [Study Schema](study-schema.md#validating-a-study).

!!! warning "Breaking change"
    Variable and tool nodes written before the typed contract carry no `kind`. Reading them reports `LEGACY_NODE` (`GET /schema` answers `409`); delete and recreate them through the typed API. There is no compatibility layer.

## Decoupled Services

The framework exposes these layers as independent microservices:

*   **Graph Service**: Manages the FalkorDB connection and provides typed APIs for graph manipulation (create, replace and delete variables and tools, connect them) and the `StudySchema` export. Conflicts return `409` and unknown nodes `404`.
*   **Execution Service**: Consumes the study schema, builds and pools GEMSEO problem instances (`ProblemPool`), caches schema data (`SchemaProvider`), and exposes an evaluation endpoint (`/evaluate`). It abstracts the complexity of running the underlying engineering models while offloading synchronous execution to local threads. The default registry shipped with the service contains the demo `Paraboloid` tool and can be extended with additional callables.
*   **Optimization Service**: The "brain" of the operation. It validates the study (`/validate`), runs the optimization loop via Ax, deciding which design points to evaluate next while enforcing graph-derived inequality constraints through calls to the Execution Service.

## Data Flow

1.  **Problem Definition**: User defines the problem graph via the Graph Service API.
2.  **Schema Retrieval (Cached)**: Execution Service fetches the current `StudySchema` from Graph Service. The schema is robustly cached with TTL (`CACHE_TTL`) and self-heals by fetching fresh hashes upon expiry.
3.  **Optimization Request**: User sends an optimization request (objectives, optional inequality constraints, algorithm settings) to Optimization Service.
    *   Optimization Service validates the request against the current study schema and answers `422` with a report if it is invalid.
    *   Optimization Service derives the design variables and parameter definitions from the study schema.
4.  **Evaluation Loop**:
    *   Optimization Service selects a candidate point `x`.
    *   Sends `x` to Execution Service via HTTP.
    *   Execution Service acquires a pre-built GEMSEO problem from the `ProblemPool` (auto-rebuilt if schema hashing changes out-of-band).
    *   Execution Service runs the GEMSEO model on a worker thread and returns the requested outputs `y`.
    *   Optimization Service updates its internal model (GP) with `(x, y)`.
    *   Repeat until convergence or step limit.
5.  **Result**: Optimization Service returns the best design point found, the best objective values, and explicit trial-history records.
