# Running Optimization

GraphMDO allows you to run optimization tasks either locally using the Python API or remotely via microservices.

## Using Microservices

The microservices architecture decouples the graph management, execution, and optimization logic.

### 1. Start Services

Ensure all services are running:

```bash
docker-compose up -d
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

```bash
curl -X POST http://localhost:8003/optimize \
     -H "Content-Type: application/json" \
     -d '{
           "design_vars": ["x", "y"],
           "objective": "f_xy",
           "bounds": [[0.0, 10.0], [0.0, 10.0]],
           "n_steps": 10
         }'
```

You will receive a JSON response with the optimization history and best results.
