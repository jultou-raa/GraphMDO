# Running Optimization

GraphMDO allows you to run optimization tasks either locally using the Python API or remotely via microservices.

## Using Microservices

The microservices architecture decouples the graph management, execution, and optimization logic.

### 1. Start Services

Start the stack and wait until every service reports healthy:

```bash
docker compose up -d --build --wait
```

The Compose file already wires `GRAPH_SERVICE_URL` and `EXECUTION_SERVICE_URL`, and each service waits for its dependencies' `/health` endpoints before starting.

### 2. Define Problem (Graph Service)

Use the Graph Service API to build your problem graph. Every variable body carries a `kind` (`range`, `choice`, `fixed` or `state`) that selects its type; see the [Study Schema](../technical-reference/study-schema.md) for the fields of each kind.

```bash
# Clear Graph
curl -X POST http://localhost:8001/clear

# Add Variables (x and y are design variables, f_xy and c_xy are computed by the tool)
curl -X POST http://localhost:8001/variables -d '{"kind": "range", "name": "x", "lower": 0.0, "upper": 10.0}' -H "Content-Type: application/json"
curl -X POST http://localhost:8001/variables -d '{"kind": "range", "name": "y", "lower": 0.0, "upper": 10.0}' -H "Content-Type: application/json"
curl -X POST http://localhost:8001/variables -d '{"kind": "state", "name": "f_xy"}' -H "Content-Type: application/json"
curl -X POST http://localhost:8001/variables -d '{"kind": "state", "name": "c_xy"}' -H "Content-Type: application/json"

# Add Tool
curl -X POST http://localhost:8001/tools -d '{"name": "Paraboloid"}' -H "Content-Type: application/json"

# Connect
curl -X POST http://localhost:8001/connections/input -d '{"source": "x", "target": "Paraboloid"}' -H "Content-Type: application/json"
curl -X POST http://localhost:8001/connections/input -d '{"source": "y", "target": "Paraboloid"}' -H "Content-Type: application/json"
curl -X POST http://localhost:8001/connections/output -d '{"source": "Paraboloid", "target": "f_xy"}' -H "Content-Type: application/json"
curl -X POST http://localhost:8001/connections/output -d '{"source": "Paraboloid", "target": "c_xy"}' -H "Content-Type: application/json"

# Read the study back
curl http://localhost:8001/schema
```

`POST` creates and answers `201`; a name that already exists is rejected with `409`. `PUT /variables/{name}` and `PUT /tools/{name}` create or replace (`201` or `200`) and keep the node's order and connections, and `DELETE` removes a node (`404` if it does not exist). Connecting a name that does not exist is a `404`, and a second tool producing the same variable is a `409`.

### 3. Run Optimization (Optimization Service)

Send an optimization request. The Optimization Service will coordinate with the Execution Service (which runs the tool) and Graph Service (for schema).

The request does not include an explicit `parameters` section. Design variables are inferred from the study schema by traversing dependencies from the requested objectives and constraints.

The built-in demo `Paraboloid` tool returns both `f_xy` and `c_xy = x - y`, so the graph above must declare both outputs: a tool's outputs must match what the function returns exactly (see the [tool function contract](../technical-reference/study-schema.md#tool-function-contract)). To optimize with other outputs, extend the execution-service tool registry with a callable returning them.

```bash
curl -X POST http://localhost:8003/optimize \
     -H "Content-Type: application/json" \
     -d '{
           "objectives": [
               {"name": "f_xy", "minimize": true}
           ],
           "n_init": 5,
           "n_steps": 10
         }'
```

The evaluation budget is explicit:

- `n_init` (default `5`, at least `1`): initial Sobol trials that explore the design space.
- `n_steps` (default `10`, at least `1`): Bayesian (BoTorch) iterations after the initial design.

The start point (the centre of the design space) is evaluated first, so the tools are called at most `1 + n_init + n_steps` times. Fewer calls happen only when Ax stops proposing new designs, for example once a small discrete space is exhausted. Values below `1` are rejected with `422`.

You will receive a JSON response containing:

- `best_parameters`: the best graph-derived design point found.
- `best_objectives`: the best objective values associated with that point.
- `history`: an explicit list of trial records, each with `parameters` and `objectives`.

Some deployments may also include optional metadata such as `serialized_client`.

### 4. Validate Before Running

`/optimize` validates the study against the study schema before any tool runs. To get the same check without running anything, send the same body to `/validate`:

```bash
curl -X POST http://localhost:8003/validate \
     -H "Content-Type: application/json" \
     -d '{"objectives": [{"name": "f_xy", "minimize": true}]}'
```

Once the study has been fetched from the Graph Service, `/validate` answers `200` with a report, whether the study is valid or not. A valid study has no errors:

```json
{"errors": [], "warnings": [], "valid": true}
```

Each finding has a stable `code`, a `message` and the `names` involved. For example, with `"parameter_constraints": ["x + z <= 1"]` added to the body above:

```json
{
  "errors": [
    {
      "code": "PARAMETER_CONSTRAINT_INVALID",
      "message": "invalid parameter constraint 'x + z <= 1': 'z' is not a design variable",
      "names": ["x + z <= 1"]
    }
  ],
  "warnings": [],
  "valid": false
}
```

`/optimize` runs the same preflight. When the study is invalid it answers `422` with the report under `detail`, and nothing is evaluated:

```bash
curl -X POST http://localhost:8003/optimize \
     -H "Content-Type: application/json" \
     -d '{"objectives": [{"name": "missing", "minimize": true}]}'
```

```json
{
  "detail": {
    "errors": [
      {
        "code": "UNKNOWN_OUTPUT",
        "message": "'missing' is not a declared variable",
        "names": ["missing"]
      }
    ],
    "warnings": [],
    "valid": false
  }
}
```

Warnings (for example `UNUSED_VARIABLE`) do not block a run. All codes are listed in the [Study Schema](../technical-reference/study-schema.md#validating-a-study) reference. The Optimization Service has no tool registry, so tool-function problems are reported by the Execution Service instead (see [Microservices](../technical-reference/microservices.md)).

Other failures:

- Malformed requests, such as non-finite numbers, empty `objectives`, unknown fields or names that break the name rule, are rejected with `422` by both `/validate` and `/optimize`. In that case `detail` is a list of `{"loc", "msg", "type"}` entries that never echo the submitted values. This is not a validation report.
- If fetching the schema from the Graph Service fails, both endpoints answer `502`: the Graph Service cannot be reached, it answers with a non-2xx status (for example `409` because a variable node was stored without `kind`), or its body is not JSON. A JSON body that does not parse as a study is not a `502`: `/validate` reports it as findings with `200`, and `/optimize` answers `422` with that report. Transport and contract failures of the Execution Service during a run are `502` too.
