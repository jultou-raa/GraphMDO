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

Use the Graph Service API to build your problem graph.

```bash
# Clear Graph
curl -X POST http://localhost:8001/clear

# Add Variables
curl -X POST http://localhost:8001/variables -d '{"name": "x", "lower": 0.0, "upper": 10.0}' -H "Content-Type: application/json"
curl -X POST http://localhost:8001/variables -d '{"name": "y", "lower": 0.0, "upper": 10.0}' -H "Content-Type: application/json"
curl -X POST http://localhost:8001/variables -d '{"name": "f_xy"}' -H "Content-Type: application/json"

# Add Tool
curl -X POST http://localhost:8001/tools -d '{"name": "Paraboloid"}' -H "Content-Type: application/json"

# Connect
curl -X POST http://localhost:8001/connections/input -d '{"source": "x", "target": "Paraboloid"}' -H "Content-Type: application/json"
curl -X POST http://localhost:8001/connections/input -d '{"source": "y", "target": "Paraboloid"}' -H "Content-Type: application/json"
curl -X POST http://localhost:8001/connections/output -d '{"source": "Paraboloid", "target": "f_xy"}' -H "Content-Type: application/json"
```

### 3. Run Optimization (Optimization Service)

Send an optimization request. The Optimization Service will coordinate with the Execution Service (which runs the tool) and Graph Service (for schema).

The request does not include an explicit `parameters` section. Design variables are inferred from the graph schema by traversing dependencies from the requested objectives and constraints.

The built-in demo execution service only exposes the scalar `Paraboloid -> f_xy` output. If you want to optimize with explicit constraints over additional outputs, extend the execution-service tool registry with a callable returning those outputs.

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

If the request cannot be mapped to independent design variables from the graph, the service returns `400`. Upstream graph or execution failures are returned as `502`.
