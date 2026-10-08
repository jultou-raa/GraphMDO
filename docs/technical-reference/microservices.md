# Microservices

All three services return request-validation failures in the same shape: `422` with `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}`. The submitted values are never echoed, so a `NaN` or an oversized string does not leak back into the response or break JSON encoding.

## Graph Service (Port 8001)

Manages the FalkorDB property graph through the typed `GraphManager`. Variable bodies are `RangeVar`, `ChoiceVar`, `FixedParam` or `StateVar` models selected by `kind`; tool bodies are `ToolNode` (`name`, optional `fidelity`, `deterministic`, `thread_safe` and `arg_map`). See the [Study Schema](study-schema.md).

| Endpoint | Purpose | Success | Errors |
| --- | --- | --- | --- |
| `POST /clear` | Resets the entire graph. | `200` `{"status": "cleared"}` | |
| `POST /variables` | Creates a variable. | `201` `{"status": "created", "variable": name}` | `409` if the name exists, `422` for an invalid body |
| `PUT /variables/{name}` | Creates or replaces a variable; keeps its order and connections. The body name must equal the path name. | `201` `created` or `200` `replaced` | `409` if the variable is produced by a tool and the new kind is not `state`, `422` for an invalid body or a name mismatch |
| `DELETE /variables/{name}` | Deletes a variable. | `200` `{"status": "deleted", "variable": name}` | `404` |
| `POST /tools` | Creates a tool. | `201` `{"status": "created", "tool": name}` | `409`, `422` |
| `PUT /tools/{name}` | Creates or replaces a tool. | `201` or `200` | `422` |
| `DELETE /tools/{name}` | Deletes a tool. | `200` `{"status": "deleted", "tool": name}` | `404` |
| `POST /connections/input` | Connects a variable (`source`) to a tool (`target`) as an input. | `200` `{"status": "connected", "type": "input"}` | `404`, `409` |
| `POST /connections/output` | Connects a tool (`source`) to a variable (`target`) as an output. | `200` `{"status": "connected", "type": "output"}` | `404`, `409` |
| `GET /schema` | Returns the `StudySchema` as JSON. | `200` | `409` if the stored graph is not a valid study |
| `GET /health` | Pings FalkorDB. | `200` `{"status": "ok", "falkordb": "ok"}` | `503` `{"status": "degraded", ...}` when the database is unreachable |

Error bodies:

-   `404`: `{"detail": {"missing": [{"label": "Variable", "name": "nope"}], "hint": null}}`. The `hint` is set when the arguments look swapped, for example a tool passed where a variable is expected.
-   `409`: `{"detail": {"error": "exists", "label": "Variable", "name": "x"}}`, `{"detail": {"error": "duplicate_producer", "variable": "f_xy", "producers": ["Paraboloid", "Other"]}}` (a variable has one producer), or `{"detail": {"error": "role_conflict", "variable": ..., "message": ...}}` (a design variable or fixed parameter cannot be a tool output, and a variable cannot be both an input and an output of one tool). `GET /schema` answers `409` with a validation report in `detail` when the graph breaks a structural rule or holds a variable node without `kind` (`LEGACY_NODE`).

## Execution Service (Port 8002)

Runs the GEMSEO problem.

-   **POST /evaluate**: Accepts `inputs` and a list of requested output names in `objectives`. Retrieves the study schema (utilizing robust caching with TTL and backoff strategies), handles asynchronous execution via a pre-built `ProblemPool` of GEMSEO instances to avoid per-request rebuild overhead, offloads synchronous GEMSEO execution to worker threads, and returns a `results` object keyed by the requested outputs. Unknown inputs or outputs are rejected before execution with `422`.
-   **Evaluation errors**: A point that cannot be evaluated is answered `422` with `{"detail": {"code", "message", "tool", "retryable"}}`. `code` is `TOOL_FAILED` (the tool function raised), `OUTPUT_INVALID` (its outputs break the output contract: wrong keys, a tuple, non-numeric, NaN or infinite values), `POINT_INFEASIBLE` (the tool raised `InfeasiblePointError` for a point it cannot compute) or `MDA_NOT_CONVERGED` (the coupled tools did not converge); `tool` names the failing tool when there is one. The pooled problem instance is discarded. Other invalid request values, such as a value that is not one of a choice variable's choices, are answered `400`. `RemoteEvaluator` turns these bodies back into the exceptions a local evaluation raises (`ToolExecutionError`, `ToolOutputError`, `InfeasiblePointError`, `MDANotConvergedError`), so a remote failure is classified like a local one; any other rejection raises `RemoteEvaluationContractError` with the server's `detail` in its message.
-   **Registry check**: Each time a schema version is loaded, every tool it references is checked against the tool registry (existence and callable signature, the registry part of `validate_study`). If the check fails, `/evaluate` answers `422` with `{"detail": {"code": "SCHEMA_INVALID", "report": {...}}}` before any problem is checked out or any tool runs. Here `detail.code` labels the response itself ("the loaded study cannot run on this service") and `report` has the same shape as the `/validate` report. Its findings are registry findings (`UNREGISTERED_TOOL`, `SIGNATURE_MISMATCH`, `ARG_MAP_UNKNOWN_INPUT`, `ARG_MAP_COLLISION`), not a `SCHEMA_INVALID` finding. A body from the Graph Service that does not parse as a `StudySchema` is a different case, answered with `502` (see below).
-   **Graph Service failures**: `503` when the Graph Service is unavailable and no cached schema exists (this includes its `409` on a graph that is not a valid study), `502` when it returns a body that is not a `StudySchema`. A cached schema is served stale while the Graph Service is down.
-   **Registry**: The default demo registry exposes the `Paraboloid` tool, which returns `f_xy` and the constraint output `c_xy = x - y` as a dictionary; a graph using it must declare both outputs. Additional tools require extending the registry.
-   **Configuration**: `CACHE_TTL`, `CACHE_BACKOFF`, `PROBLEM_POOL_SIZE` and `POOL_ACQUIRE_TIMEOUT`.

## Optimization Service (Port 8003)

Orchestrates the optimization process.

-   **POST /optimize**: Accepts optimization objectives (at least one), optional constraints using `<=` or `>=`, and algorithm settings (`n_steps`, `n_init`, `use_bonsai`, `parameter_constraints`). Objective and constraint names follow the schema name rule, and thresholds and bounds must be finite numbers. `n_init` (default 5) is the number of initial Sobol trials and `n_steps` (default 10) the number of Bayesian (BoTorch) iterations; both must be at least 1. Together with the start point, the tools are called at most `1 + n_init + n_steps` times. The study is validated first (see `/validate`); an invalid study is rejected with `422` before any tool runs. The service then derives design variables from the study schema using the requested objectives and constraints, and uses `BayesianOptimizer` wrapping Ax Platform to drive the `RemoteEvaluator` connected to the Execution Service.
-   **POST /validate**: Takes the same body as `/optimize` and runs the same preflight against the study schema without calling any tool. Once the schema has been fetched from the Graph Service, it answers `200` with a report `{"errors": [...], "warnings": [...], "valid": bool}`, whether the study is valid or not. Each finding has a stable `code`, a `message` and the `names` involved. A fetched JSON body that does not parse as a `StudySchema` is reported as findings (`SCHEMA_INVALID` or the structural codes), still with `200`. It answers `502` when fetching `GET /schema` fails: a transport error, a non-2xx status (for example the Graph Service's `409` for a stored legacy node) or a body that is not JSON. A request body that fails validation is answered with the generic `422` `detail` list (`loc`, `msg`, `type`). The tool registry belongs to the Execution Service, so registry findings are not part of this report.
-   **Response Shape**: Returns `best_parameters`, `best_objectives`, and a `history` list of explicit trial records, each containing `parameters` and `objectives`. Some deployments may also expose optional metadata such as `serialized_client`.
-   **Error Mapping**: Returns `422` for a malformed request on both endpoints (a `detail` list of `loc`, `msg`, `type`) and, on `/optimize` only, for an invalid study (the validation report is the `detail`; `/validate` answers `200` with that report instead). It returns `400` for optimizer configuration errors, `502` for graph/execution service communication failures, invalid execution responses, or a Graph Service error status (including its `409`) or non-JSON body, and `500` for optimization execution failures.
-   **Compose Wiring**: `docker-compose.yml` sets both `GRAPH_SERVICE_URL` and `EXECUTION_SERVICE_URL` for `optimization-service`, and every service declares a `/health` healthcheck so `docker compose up --wait` returns once the stack is ready.
