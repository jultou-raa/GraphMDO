# Microservices

## Graph Service (Port 8001)

Manages the FalkorDB property graph.

-   **POST /clear**: Resets the entire graph.
-   **POST /variables**: Creates a new variable node.
-   **POST /tools**: Creates a new tool node.
-   **POST /connections/input**: Connects a variable to a tool (input).
-   **POST /connections/output**: Connects a tool to a variable (output).
-   **GET /schema**: Returns the complete graph schema as a JSON object for translation.

## Execution Service (Port 8002)

Runs the OpenMDAO problem.

-   **POST /evaluate**: Accepts input values and objective/constraint names to evaluate. Retrieves the graph schema (utilizing robust caching with TTL and backoff strategies), handles asynchronous execution via a pre-built `ProblemPool` of OpenMDAO instances to avoid per-request rebuild overhead, offloads synchronous OpenMDAO execution to worker threads, and returns the objective value. The service automatically detects schema changes and self-heals the problem pool.

## Optimization Service (Port 8003)

Orchestrates the optimization process.

-   **POST /optimize**: Accepts optimization parameters (ranges, choices, single/multi objectives, outcome constraints, steps). Uses `BayesianOptimizer` wrapping Ax Platform to drive the `RemoteEvaluator` connected to the Execution Service.
