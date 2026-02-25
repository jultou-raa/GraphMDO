# GraphMDO: Dynamic Multi-Fidelity MDO Framework

GraphMDO is an advanced Multidisciplinary Design Optimization (MDO) framework that integrates graph databases with state-of-the-art optimization and surrogate modeling tools.

## Key Features

*   **Native Graph Formulation**: Uses [FalkorDB](https://falkordb.com/) to store problem definitions (variables, tools, dependencies) as a property graph.
*   **Dynamic Problem Construction**: Automatically translates the graph topology into an executable [OpenMDAO](https://openmdao.org/) problem.
*   **Multi-Fidelity Surrogates**: Integrates [SMT](https://smt.readthedocs.io/en/latest/) for Co-Kriging and other surrogate models.
*   **Bayesian Optimization**: Leverages [BoTorch](https://botorch.org/) for efficient, constrained optimization.

## Project Architecture

1.  **FalkorDB**: Stores the "Fundamental Problem Graph" (FPG).
2.  **Graph Manager**: Python API to manipulate the graph structure.
3.  **Translator**: Converts the graph into an OpenMDAO System.
4.  **Optimizer**: Drivers (BoTorch, Pymoo) that execute the OpenMDAO problem.

## Installation

This project uses `uv` for dependency management.

1.  **Install uv** (if not installed):
    See https://docs.astral.sh/uv/getting-started/installation/

2.  **Clone and Install**:
    ```bash
    git clone https://github.com/your-repo/graph-mdo.git
    cd graph-mdo
    uv sync
    ```

3.  **FalkorDB**:
    Ensure you have a running FalkorDB instance (e.g., via Docker):
    ```bash
    docker run -p 6379:6379 -it falkordb/falkordb
    ```

## Usage

### 1. Defining a Problem (Python API)

You can programmatically build your MDO problem graph:

```python
from mdo_framework.db.graph_manager import GraphManager

gm = GraphManager()
gm.clear_graph()

# Define Variables
gm.add_variable("x", value=1.0, lower=0.0, upper=10.0)
gm.add_variable("y", value=2.0, lower=0.0, upper=10.0)
gm.add_variable("z", value=0.0)

# Define Tool
gm.add_tool("MyTool")

# Define Connections
gm.connect_input_to_tool("x", "MyTool")
gm.connect_input_to_tool("y", "MyTool")
gm.connect_tool_to_output("MyTool", "z")
```

### 2. Running Optimization

Once the graph is populated, you can run the optimization workflow. You need to provide the actual Python functions corresponding to the tool names in the graph.

```python
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.optimization.optimizer import BayesianOptimizer

# Define tool implementation
def my_tool_func(x, y):
    return x + y  # Simple example

# Registry maps graph tool names to Python callables
tool_registry = {
    "MyTool": my_tool_func
}

# Build OpenMDAO Problem from Graph
builder = GraphProblemBuilder(gm)
prob = builder.build_problem(tool_registry)

# Run Optimization
optimizer = BayesianOptimizer(
    problem=prob,
    design_vars=["x", "y"],
    objective="z"
)

result = optimizer.optimize(n_steps=10)
print(f"Best Result: {result['best_y']} at {result['best_x']}")
```

### 3. Running Tests

```bash
uv run pytest tests/
```

## Contributing

1.  Follow PEP 8 guidelines.
2.  Ensure 100% test coverage for new features.
3.  Use `uv run pre-commit run --all-files` before committing.
