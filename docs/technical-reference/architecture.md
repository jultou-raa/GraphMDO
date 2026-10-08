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
    *   Wraps Python functions or external codes into `ToolComponent`, a GEMSEO discipline that enforces the tool output contract and approximates Jacobians by finite differences (see [Study Schema](study-schema.md#tool-function-contract)).
    *   Solves coupled tools with a convergence-checked MDA (see [Coupled tools](#coupled-tools)).
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

## Coupled tools

`GraphProblemBuilder.build_problem(registry, mda_settings=None)` chains the tools in a `StrictMDAChain`. Tools that exchange variables in a cycle form a coupled group, solved by an inner MDA configured by `MDASettings` (`mdo_framework.core.mda`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `inner_mda_name` | `"MDAGaussSeidel"` | Algorithm of each coupled group: `MDAGaussSeidel`, `MDAJacobi` or `MDANewtonRaphson`. |
| `tolerance` | `1e-6` | Normalized residual at which a coupled group has converged. |
| `max_mda_iter` | `20` | Maximum number of iterations. |
| `max_consecutive_unsuccessful_iterations` | `8` | Iterations without residual decrease after which the algorithm stops. |
| `n_processes` | `1` | Coupled tools run at the same time. Above 1 only with `MDAJacobi`, and every tool producing a coupling must declare `thread_safe` (otherwise `TOOL_NOT_THREAD_SAFE`). |
| `accept_tolerance` | `None` | Largest residual still accepted when the algorithm stops; `None` accepts only `tolerance`. |

By default the coupled tools run one after the other, so a tool exception always propagates (GEMSEO's parallel execution drops every worker exception that is not a `ValueError`). When a coupled group stops above its accepted residual, the evaluation raises `MDANotConvergedError` instead of returning the unconverged values; this applies to `LocalEvaluator`, to `/evaluate` and to every optimizer evaluation. Couplings start from their `initial_guess`, or `0.0`. If any tool is declared `deterministic=False`, the chain and its algorithms do not cache.

Evaluation failures share one hierarchy (`mdo_framework.core.errors`), all `ValueError` subclasses with a stable `code`: `ToolExecutionError` (`TOOL_FAILED`, the tool raised), `ToolOutputError` (`OUTPUT_INVALID`, outputs break the contract), `InfeasiblePointError` (`POINT_INFEASIBLE`, raised by a tool for a point it cannot compute) and `MDANotConvergedError` (`MDA_NOT_CONVERGED`). The Execution Service reports them as structured `422` errors, which `RemoteEvaluator` raises again as the same classes (see [Microservices](microservices.md#execution-service-port-8002)). `BayesianOptimizer.explore()` skips the samples that fail (GEMSEO's DOE logs and continues); if no sample evaluates, it raises the first typed error.

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
