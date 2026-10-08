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

# 4. Setup Optimizer
evaluator = LocalEvaluator(prob, builder.variable_specs)
optimizer = BayesianOptimizer(
    evaluator,
    resolved.design_variables,
    [ObjectiveSpec(name="z", minimize=True)],
    [ConstraintSpec(name="c_xy", op="<=", bound=0.0)],
)

# 5. Execute Optimization
# 5 Sobol trials (n_init) + 10 Bayesian iterations (n_steps) = 15 evaluations,
# plus the start point x0 when evaluate_x0=True
result = optimizer.optimize(n_steps=10, n_init=5)
print(f"Best Result: {result['best_objectives']} at {result['best_parameters']}")
print(f"Stopped because: {result['stop_reason']}, feasible: {result['feasible']}")
```

`resolve_dependencies` returns a `ResolvedInputs` with the `design_variables`, `fixed_parameters` and `tools` that the targets depend on. It raises `StudyValidationError` (with the full report in `.report`) when the targets cannot be computed.

`optimize()` returns a result with these keys:

- `best_parameters` and `best_objectives`: the best feasible completed trial (the compromise point closest to the ideal point on the Pareto front for several objectives), or the least-violating completed trial when none is feasible.
- `feasible` and `constraints`: whether that trial satisfies every constraint, and for each constraint its `value`, `margin`, `satisfied` and `tolerance`.
- `pareto_front`: the non-dominated trials (`parameters` and `objectives`) of a multi-objective run, `[]` otherwise.
- `history`: one record per trial with `index`, `phase` (`x0`, `init` or `bo`), `status` (`completed`, `failed` or `abandoned`), `reason`, `parameters`, `objectives`, `constraints` and `feasible`. A tool that raises makes a `failed` trial; the run continues.
- `stop_reason`: `budget`, `max_time`, `search_space_exhausted`, `consecutive_failures` or `aborted`.
- `evaluations`: the number of evaluated trials per phase (`x0`, `init`, `bo`) and the number of `failed` ones.

The start point x0 is evaluated only when `evaluate_x0=True` or every design variable declares an `initial` value. If no trial completes, `optimize()` raises `OptimizationExecutionError` whose `partial_result` holds the same structure.

## Next Steps

- Explore [Installation](installation.md) for full setup instructions.
- Learn about [Running Optimization](running-optimization.md) with microservices.
- Read the [Study Schema](../technical-reference/study-schema.md) reference and the finding codes.
