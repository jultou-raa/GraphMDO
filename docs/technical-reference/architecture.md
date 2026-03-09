# System Architecture

GraphMDO is designed as a modular, service-oriented framework for Multidisciplinary Design Optimization.

## Core Components

The architecture consists of three primary layers:

1.  **Graph Layer (FalkorDB)**
    *   Stores the "Fundamental Problem Graph" (FPG).
    *   Nodes represent Variables (Inputs, Outputs) and Tools (Functions, Codes).
    *   Edges represent data flow (Inputs To, Outputs From).
    *   The schema is dynamically queryable via OpenCypher.

2.  **Execution Layer (GEMSEO)**
    *   Translates the graph topology into an executable GEMSEO Problem.
    *   Wraps Python functions or external codes into `ToolComponent`.
    *   Handles variable promotion and data passing between components.

3.  **Optimization Layer (Ax/SMT)**
    *   Drives the execution layer to minimize/maximize objectives.
    *   Uses Constrained Bayesian Optimization via Ax Platform (handling continuous, discrete, choices, and multi-objective definitions).
    *   Supports multi-fidelity surrogates (Co-Kriging) via SMT integration.

## Decoupled Services

The framework exposes these layers as independent microservices:

*   **Graph Service**: Manages the FalkorDB connection and provides APIs for graph manipulation (CRUD operations on nodes/edges) and schema export.
*   **Execution Service**: Consumes the graph schema, builds and pools GEMSEO problem instances (`ProblemPool`), caches schema data (`SchemaProvider`), and exposes an evaluation endpoint (`/evaluate`). It abstracts the complexity of running the underlying engineering models while offloading synchronous execution to local threads. The default registry shipped with the service contains the demo `Paraboloid` tool and can be extended with additional callables.
*   **Optimization Service**: The "brain" of the operation. It runs the optimization loop via Ax, deciding which design points to evaluate next while enforcing graph-derived inequality constraints through calls to the Execution Service.

## Data Flow

1.  **Problem Definition**: User defines the problem graph via the Graph Service API.
2.  **Schema Retrieval (Cached)**: Execution Service fetches the current graph schema from Graph Service. The schema is robustly cached with TTL (`CACHE_TTL`) and self-heals by fetching fresh hashes upon expiry.
3.  **Optimization Request**: User sends an optimization request (objectives, optional inequality constraints, algorithm settings) to Optimization Service.
    *   Optimization Service derives the design variables and parameter definitions from the current graph schema.
4.  **Evaluation Loop**:
    *   Optimization Service selects a candidate point `x`.
    *   Sends `x` to Execution Service via HTTP.
    *   Execution Service acquires a pre-built GEMSEO problem from the `ProblemPool` (auto-rebuilt if schema hashing changes out-of-band).
    *   Execution Service runs the GEMSEO model on a worker thread and returns the requested outputs `y`.
    *   Optimization Service updates its internal model (GP) with `(x, y)`.
    *   Repeat until convergence or step limit.
5.  **Result**: Optimization Service returns the best design point found, the best objective values, and explicit trial-history records.
