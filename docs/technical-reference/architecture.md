# System Architecture

GraphMDO is designed as a modular, service-oriented framework for Multidisciplinary Design Optimization.

## Core Components

The architecture consists of three primary layers:

1.  **Graph Layer (FalkorDB)**
    *   Stores the "Fundamental Problem Graph" (FPG).
    *   Nodes represent Variables (Inputs, Outputs) and Tools (Functions, Codes).
    *   Edges represent data flow (Inputs To, Outputs From).
    *   The schema is dynamically queryable via OpenCypher.

2.  **Execution Layer (OpenMDAO)**
    *   Translates the graph topology into an executable OpenMDAO System.
    *   Wraps Python functions or external codes into `ToolComponent`.
    *   Handles variable promotion and data passing between components.

3.  **Optimization Layer (BoTorch/SMT)**
    *   Drives the execution layer to minimize/maximize objectives.
    *   Uses Bayesian Optimization (Gaussian Processes + Acquisition Functions).
    *   Supports multi-fidelity surrogates (Co-Kriging) via SMT integration.

## Decoupled Services

The framework exposes these layers as independent microservices:

*   **Graph Service**: Manages the FalkorDB connection and provides APIs for graph manipulation (CRUD operations on nodes/edges) and schema export.
*   **Execution Service**: Consumes the graph schema, builds the OpenMDAO problem instance, and exposes an evaluation endpoint (`/evaluate`). It abstracts the complexity of running the underlying engineering models.
*   **Optimization Service**: The "brain" of the operation. It runs the optimization loop (e.g., BoTorch), deciding which design points to evaluate next by calling the Execution Service.

## Data Flow

1.  **Problem Definition**: User defines the problem graph via the Graph Service API.
2.  **Schema Retrieval**: Execution Service fetches the current graph schema from Graph Service.
3.  **Optimization Request**: User sends an optimization request (design vars, objective, bounds) to Optimization Service.
4.  **Evaluation Loop**:
    *   Optimization Service selects a candidate point `x`.
    *   Sends `x` to Execution Service via HTTP.
    *   Execution Service runs the OpenMDAO model and returns objective `y`.
    *   Optimization Service updates its internal model (GP) with `(x, y)`.
    *   Repeat until convergence or step limit.
5.  **Result**: Optimization Service returns the best design point found.
