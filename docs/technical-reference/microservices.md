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

-   **POST /evaluate**: Accepts `inputs` and a list of requested output names in `objectives`. Retrieves the graph schema (utilizing robust caching with TTL and backoff strategies), handles asynchronous execution via a pre-built `ProblemPool` of GEMSEO instances to avoid per-request rebuild overhead, offloads synchronous GEMSEO execution to worker threads, and returns a `results` object keyed by the requested outputs. Unknown inputs or outputs are rejected before execution. Each time a schema version is loaded, every tool it references is checked against the tool registry (existence and callable signature); if the check fails, `/evaluate` answers `422` with `{"code": "SCHEMA_INVALID", "report": ...}` (the same report shape as `/validate`) before any problem is checked out or any tool runs. The default demo registry currently exposes the `Paraboloid` tool returning the scalar output `f_xy`; additional constrained outputs require extending the registry.

## Optimization Service (Port 8003)

Orchestrates the optimization process.

-   **POST /optimize**: Accepts optimization objectives (at least one), optional constraints using `<=` or `>=`, and algorithm settings (`n_steps`, `n_init`, `use_bonsai`, `parameter_constraints`). Objective and constraint names follow the schema name rule, and thresholds and bounds must be finite numbers. `n_init` (default 5) is the number of initial Sobol trials and `n_steps` (default 10) the number of Bayesian (BoTorch) iterations; both must be at least 1. Together with the start point, the tools are called at most `1 + n_init + n_steps` times. The study is validated first (see `/validate`); an invalid study is rejected with `422` before any tool runs. The service then derives design variables from the graph schema using the requested objectives and constraints, and uses `BayesianOptimizer` wrapping Ax Platform to drive the `RemoteEvaluator` connected to the Execution Service.
-   **POST /validate**: Takes the same body as `/optimize` and runs the same preflight against the graph schema without calling any tool. It always answers `200` with a report `{"errors": [...], "warnings": [...], "valid": bool}`, where each finding has a stable `code`, a `message` and the `names` involved. A schema that does not parse is reported as findings, not as an error status. The tool registry belongs to the Execution Service, so registry findings are not part of this report.
-   **Response Shape**: Returns `best_parameters`, `best_objectives`, and a `history` list of explicit trial records, each containing `parameters` and `objectives`. Some deployments may also expose optional metadata such as `serialized_client`.
-   **Error Mapping**: Returns `422` for a malformed request or an invalid study (the body of an invalid study is the validation report under `detail`), `400` for optimizer configuration errors, `502` for graph/execution service communication failures or invalid execution responses, and `500` for optimization execution failures.
-   **Compose Wiring**: `docker-compose.yml` sets both `GRAPH_SERVICE_URL` and `EXECUTION_SERVICE_URL` for `optimization-service`, and every service declares a `/health` healthcheck so `docker compose up --wait` returns once the stack is ready.
