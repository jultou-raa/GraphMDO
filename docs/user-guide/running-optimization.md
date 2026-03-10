# Running Optimization

GraphMDO allows you to run optimization tasks either locally using the Python API or remotely via microservices.

## Using Microservices

The microservices architecture decouples the graph management, execution, and optimization logic.

### 1. Start Services

Ensure all services are running:

```bash
docker compose up -d
```

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
           "n_steps": 10
         }'
```

You will receive a JSON response containing:

- `best_parameters`: the best graph-derived design point found.
- `best_objectives`: the best objective values associated with that point.
- `history`: an explicit list of trial records, each with `parameters` and `objectives`.

Some deployments may also include optional metadata such as `serialized_client`.

If the request cannot be mapped to independent design variables from the graph, the service returns `400`. Upstream graph or execution failures are returned as `502`.

If you run the full stack with the current `docker-compose.yml`, also ensure the optimization service can resolve the graph service through `GRAPH_SERVICE_URL=http://graph-service:8001`.
