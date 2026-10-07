# Quick Start

This guide will walk you through setting up a simple Multidisciplinary Design Optimization (MDO) problem using GraphMDO.

## Prerequisites

- Python 3.12+
- `uv` installed (recommended)
- Docker (for FalkorDB and microservices)

## Define a Problem Graph

You can use the Python API to programmatically define your problem in the FalkorDB database. Variables are typed: a `RangeVar` is a design variable with bounds, a `StateVar` is a value computed by a tool. See the [Study Schema](../technical-reference/study-schema.md) for all variable kinds and rules.

```python
from mdo_framework.db.graph_manager import GraphManager
from mdo_framework.schema import RangeVar, StateVar, ToolNode

# Initialize Graph Manager
gm = GraphManager()
gm.clear_graph()  # Start fresh

# 1. Define Variables (Design Variables, Outputs, etc.)
gm.add_variable(RangeVar(name="x", lower=0.0, upper=10.0))
gm.add_variable(RangeVar(name="y", lower=0.0, upper=10.0))
gm.add_variable(StateVar(name="z"))
gm.add_variable(StateVar(name="c_xy"))

# 2. Define Tools
# Tools are functions or external codes that compute outputs from inputs.
gm.add_tool(ToolNode(name="MyTool"))

# 3. Define Connections
# Connect variable nodes to tool nodes to define data flow.
gm.connect_input_to_tool("x", "MyTool")
gm.connect_input_to_tool("y", "MyTool")
gm.connect_tool_to_output("MyTool", "z")
gm.connect_tool_to_output("MyTool", "c_xy")
```

`add_variable` and `add_tool` raise `NodeExistsError` when the name is taken; `put_variable` and `put_tool` create or replace. Connecting a name that does not exist raises `NodeNotFoundError`, and giving a variable a second producer raises `DuplicateProducerError`.

## Validate the Study

Read the graph back as a `StudySchema` and check it before running anything. `validate_study` calls no tool and reports every problem at once. Passing the tool registry also checks that each tool has a function that accepts the graph inputs.

```python
from mdo_framework.schema import ConstraintSpec, ObjectiveSpec
from mdo_framework.validation import validate_study


# 1. Define Tool Implementation
def my_tool_func(x, y):
    z = x + y
    c_xy = x - y
    return {"z": z, "c_xy": c_xy}  # Best practice: return a dictionary mapping outputs


# Registry maps graph tool names to Python callables
tool_registry = {"MyTool": my_tool_func}

schema = gm.get_study_schema()
report = validate_study(
    schema,
    objectives=[ObjectiveSpec(name="z", minimize=True)],
    constraints=[ConstraintSpec(name="c_xy", op="<=", bound=0.0)],
    registry=tool_registry,
)
print(report.valid)  # True
for finding in report.errors + report.warnings:
    print(finding.code, finding.message)
```

## Run Optimization

Once the graph is populated and valid, you can run the optimization.

```python
from mdo_framework.core.translator import GraphProblemBuilder
from mdo_framework.optimization.optimizer import BayesianOptimizer
from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.core.topology import TopologicalAnalyzer

# 2. Build GEMSEO Problem from the Study Schema
builder = GraphProblemBuilder(schema)
prob = builder.build_problem(tool_registry)

# 3. Resolve Topological Dependencies
analyzer = TopologicalAnalyzer(schema)
# The targets are our objective and constraint
resolved = analyzer.resolve_dependencies(["z", "c_xy"])
parameters = analyzer.extract_parameters(resolved.design_variables)

# 4. Setup Optimizer
evaluator = LocalEvaluator(prob, builder.variable_specs)
optimizer = BayesianOptimizer(
    evaluator=evaluator,
    parameters=parameters,
    objectives=[{"name": "z", "minimize": True}],
    constraints=[{"name": "c_xy", "op": "<=", "bound": 0.0}],
)

# 5. Execute Optimization
# x0 + 5 Sobol trials (n_init) + 10 Bayesian iterations (n_steps) = 16 evaluations
result = optimizer.optimize(n_steps=10, n_init=5)
print(f"Best Result: {result['best_objectives']} at {result['best_parameters']}")
```

`resolve_dependencies` returns a `ResolvedInputs` with the `design_variables`, `fixed_parameters` and `tools` that the targets depend on. It raises `StudyValidationError` (with the full report in `.report`) when the targets cannot be computed.

## Next Steps

- Explore [Installation](installation.md) for full setup instructions.
- Learn about [Running Optimization](running-optimization.md) with microservices.
- Read the [Study Schema](../technical-reference/study-schema.md) reference and the finding codes.
