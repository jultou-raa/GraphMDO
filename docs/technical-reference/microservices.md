# Microservices

## Graph Service (Port 8001)

Manages the FalkorDB property graph.

-   **POST /clear**: Resets the entire graph.
-   **POST /variables**: Creates a new variable node.
-   **POST /tools**: Creates a new tool node.
-   **POST /connections/input**: Connects a variable to a tool (input).
-   **POST /connections/output**: Connects a tool to a variable (output).
-   **GET /schema**: Returns the complete graph schema as a JSON object for translation.
-   **GET /health**: Pings FalkorDB; returns `200` with `{"status": "ok"}` or `503` with `{"status": "degraded"}` when the database is unreachable.

## Execution Service (Port 8002)

Runs the GEMSEO problem.

-   **POST /evaluate**: Accepts `inputs` and a list of requested output names in `objectives`. Retrieves the graph schema (utilizing robust caching with TTL and backoff strategies), handles asynchronous execution via a pre-built `ProblemPool` of GEMSEO instances to avoid per-request rebuild overhead, offloads synchronous GEMSEO execution to worker threads, and returns a `results` object keyed by the requested outputs. Unknown inputs or outputs are rejected before execution. The default demo registry currently exposes the `Paraboloid` tool returning the scalar output `f_xy`; additional constrained outputs require extending the registry.

## Optimization Service (Port 8003)

Orchestrates the optimization process.

-   **POST /optimize**: Accepts optimization objectives, optional constraints using `<=` or `>=`, and algorithm settings (`n_steps`, `n_init`, `use_bonsai`, `parameter_constraints`). `n_init` (default 5) is the number of initial Sobol trials and `n_steps` (default 10) the number of Bayesian (BoTorch) iterations; both must be at least 1. Together with the start point, the tools are called at most `1 + n_init + n_steps` times. The service derives design variables from the graph schema using the requested objectives and constraints, then uses `BayesianOptimizer` wrapping Ax Platform to drive the `RemoteEvaluator` connected to the Execution Service.
-   **Response Shape**: Returns `best_parameters`, `best_objectives`, and a `history` list of explicit trial records, each containing `parameters` and `objectives`. Some deployments may also expose optional metadata such as `serialized_client`.
-   **Error Mapping**: Returns `400` for invalid graph-derived optimization requests, `502` for graph/execution service communication failures or invalid execution responses, and `500` for optimization execution failures.
-   **Compose Wiring**: `docker-compose.yml` sets both `GRAPH_SERVICE_URL` and `EXECUTION_SERVICE_URL` for `optimization-service`, and every service declares a `/health` healthcheck so `docker compose up --wait` returns once the stack is ready.
