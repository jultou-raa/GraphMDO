# Quick Start

This guide will walk you through setting up a simple Multidisciplinary Design Optimization (MDO) problem using GraphMDO.

## Prerequisites

- Python 3.12+
-  installed (recommended)
- Docker (for FalkorDB and microservices)

## Define a Problem Graph

You can use the Python API to programmatically define your problem in the FalkorDB database.

```python
from mdo_framework.db.graph_manager import GraphManager

# Initialize Graph Manager
gm = GraphManager()
gm.clear_graph()  # Start fresh

# 1. Define Variables (Design Variables, Outputs, etc.)
gm.add_variable("x", value=1.0, lower=0.0, upper=10.0)
gm.add_variable("y", value=2.0, lower=0.0, upper=10.0)
gm.add_variable("z", value=0.0)

# 2. Define Tools
# Tools are functions or external codes that compute outputs from inputs.
gm.add_tool("MyTool")

# 3. Define Connections
# Connect variable nodes to tool nodes to define data flow.
gm.connect_input_to_tool("x", "MyTool")
gm.connect_input_to_tool("y", "MyTool")
gm.connect_tool_to_output("MyTool", "z")
```

## Run Optimization

Once the graph is populated, you can run the optimization.

```python
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.optimization.optimizer import BayesianOptimizer, LocalEvaluator
import torch

# 1. Define Tool Implementation
def my_tool_func(x, y):
    return x + y  # Example function: z = x + y

# Registry maps graph tool names to Python callables
tool_registry = {
    "MyTool": my_tool_func
}

# 2. Build OpenMDAO Problem from Graph Schema
schema = gm.get_graph_schema()
builder = GraphProblemBuilder(schema)
prob = builder.build_problem(tool_registry)

# 3. Setup Optimizer
evaluator = LocalEvaluator(prob)
optimizer = BayesianOptimizer(
    evaluator=evaluator,
    design_vars=["x", "y"],
    objective="z",
    bounds=torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.double)
)

# 4. Execute Optimization
result = optimizer.optimize(n_steps=10)
print(f"Best Result: {result['best_y']} at {result['best_x']}")
```

## Next Steps

- Explore [Installation](installation.md) for full setup instructions.
- Learn about [Running Optimization](running-optimization.md) with microservices.
