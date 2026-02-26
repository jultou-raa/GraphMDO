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

-   **POST /evaluate**: Accepts input values and an objective name. Retrieves the graph schema, builds the OpenMDAO problem, sets inputs, runs the model, and returns the objective value.

## Optimization Service (Port 8003)

Orchestrates the optimization process.

-   **POST /optimize**: Accepts optimization parameters (design variables, objective, bounds, steps). Uses `BayesianOptimizer` to drive the `RemoteEvaluator` connected to the Execution Service.
