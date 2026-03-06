# GraphMDO: Dynamic Multi-Fidelity MDO Framework

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/graphmdo.svg)](https://pypi.org/project/graphmdo/)
[![Quality Checks](https://github.com/jultou-raa/GraphMDO/actions/workflows/quality.yml/badge.svg)](https://github.com/jultou-raa/GraphMDO/actions/workflows/quality.yml)
[![Security Scan](https://github.com/jultou-raa/GraphMDO/actions/workflows/security.yml/badge.svg)](https://github.com/jultou-raa/GraphMDO/actions/workflows/security.yml)
[![Deploy Documentation](https://github.com/jultou-raa/GraphMDO/actions/workflows/docs.yml/badge.svg)](https://github.com/jultou-raa/GraphMDO/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/jultou-raa/GraphMDO/graph/badge.svg?token=f2150ayDNv)](https://codecov.io/gh/jultou-raa/GraphMDO)

GraphMDO bridges data engineering and MDO. It extracts topological data (solvers, variables, fidelity levels) to form an oriented graph, specifically utilizing GEMSEO for semantic formulation and execution. Execution is handled natively by GEMSEO and the Surrogate Modeling Toolbox (SMT), driven by constrained Bayesian optimization (ax-platform) or evolutionary algorithms (pymoo). The primary operational goal is to isolate and maximize a single target performance metric while strictly holding all other performance metrics constant.

## Key Features

*   **Native Graph Formulation**: Uses [FalkorDB](https://falkordb.com/) to store problem definitions (variables, tools, dependencies) as a property graph.
*   **Dynamic Problem Construction**: Automatically translates the graph topology into an executable [GEMSEO](https://gemseo.readthedocs.io/) MDO formulation.
*   **Multi-Fidelity Surrogates**: Integrates [SMT](https://smt.readthedocs.io/en/latest/) for Co-Kriging and other surrogate models.
*   **Constrained Bayesian Optimization**: Leverages [Ax Platform](https://ax.dev/) for robust optimization, easily managing GEMSEO multi-objective targets, fidelity, and discrete/continuous parameters.

## Project Architecture

The framework is divided into four sequential phases, each corresponding to a layer of abstraction:

1. **Graph Layer** — FalkorDB stores the *Fundamental Problem Graph* (FPG): variables, tools, and directed connections.
2. **Core Layer** — Python modules translate the graph schema into executable GEMSEO constructs (disciplines, design space, topology).
3. **Execution Service** — A FastAPI microservice that manages a pool of GEMSEO `OptimizationProblem` instances and exposes an HTTP evaluation endpoint.
4. **Optimization Layer** — Bayesian (Ax Platform) or DOE (GEMSEO Sobol) drivers run over the GEMSEO MDO scenario.

```mermaid
flowchart TD
    subgraph USER["👤 User"]
        U1["Define Variables\n& Tools"]
        U2["Provide Tool\nCallables / Registry"]
        U3["Read Results"]
    end

    subgraph GRAPH["🗄️ Graph Layer — FalkorDB"]
        G1["GraphManager\n(graph_manager.py)"]
        G2[("FalkorDB\nProperty Graph\nFPG")]
        G3["FalkorDB Client\n(client.py)"]
        G1 -- "add_variable / add_tool\nconnect_input_to_tool" --> G2
        G3 -- "Cypher queries" --> G2
        G1 -- uses --> G3
    end

    subgraph CORE["⚙️ Core Layer — mdo_framework.core"]
        C1["TopologicalAnalyzer\n(topology.py)\nresolve_dependencies()"]
        C2["GraphProblemBuilder\n(translator.py)\nbuild_problem()"]
        C3["GemseoComponent\n(components.py)\nGEMSEO Discipline wrapper"]
        C4["SurrogateComponent\n(surrogates.py)\nSMT Co-Kriging"]
        C5["LocalEvaluator\n(evaluators.py)"]
        C2 --> C3
        C2 --> C4
        C3 --> C5
    end

    subgraph GEMSEO["📐 GEMSEO MDO Engine"]
        GS1["DesignSpace"]
        GS2["MDOScenario / DOEScenario"]
        GS3["OptimizationProblem"]
        GS2 --> GS3
        GS1 --> GS2
    end

    subgraph OPT["🔬 Optimization Layer — mdo_framework.optimization"]
        O1["BayesianOptimizer\n(optimizer.py)"]
        O2["AxOptimizationLibrary\n(ax_algo_lib.py)\nCustom GEMSEO Algorithm"]
        O3["Ax Client\n(ax-platform)"]
        O4["DOE Explore\nSobol sampler"]
        O1 --> O2
        O1 --> O4
        O2 --> O3
        O3 -- "suggest_next_trials()" --> O2
        O2 -- "complete_trial() / mark_failed()" --> O3
    end

    subgraph SVC["🌐 Execution Service — services.execution"]
        S1["FastAPI App\n(main.py)"]
        S2["SchemaProvider\n(schema cache + TTL)"]
        S3["ProblemPool\n(GEMSEO problem pool)"]
        S4["/evaluate endpoint"]
        S5["/health endpoint"]
        S1 --> S2
        S1 --> S3
        S1 --> S4
        S1 --> S5
        S4 --> S3
    end

    %% Cross-layer data flow
    U1 --> G1
    U2 --> C2
    G2 -- "get_graph_schema()" --> C1
    G2 -- "get_graph_schema()" --> C2
    C1 -- "design parameters\n& bounds" --> O1
    C5 --> GS3
    GS1 --> O1
    GS3 --> O2
    O2 -- "evaluate_functions(x)" --> GS3
    O3 -- "best_parameterization" --> O1
    O1 --> U3

    %% Remote path
    O1 -. "RemoteEvaluator\n(HTTP /evaluate)" .-> S4
    S3 -. "GEMSEO problem\ninstance" .-> S4

    style USER fill:#1e293b,stroke:#64748b,color:#f1f5f9
    style GRAPH fill:#0f172a,stroke:#3b82f6,color:#bfdbfe
    style CORE fill:#0f172a,stroke:#8b5cf6,color:#ddd6fe
    style GEMSEO fill:#0f172a,stroke:#06b6d4,color:#a5f3fc
    style OPT fill:#0f172a,stroke:#f59e0b,color:#fef3c7
    style SVC fill:#0f172a,stroke:#10b981,color:#a7f3d0
```

## Installation

This project uses `uv` for dependency management.

1.  **Install uv** (if not installed):
    See [astral.sh/uv](https://astral.sh/uv).

2.  **Clone and Install**:
    ```bash
    git clone https://github.com/jultou-raa/GraphMDO.git
    cd GraphMDO
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
from mdo_framework.core.evaluators import LocalEvaluator
from mdo_framework.core.topology import TopologicalAnalyzer

# Define tool implementation
def my_tool_func(x, y):
    return x + y  # Simple example

# Registry maps graph tool names to Python callables
tool_registry = {
    "MyTool": my_tool_func
}

# Build GEMSEO Problem from Graph
schema = gm.get_graph_schema()
builder = GraphProblemBuilder(schema)
prob = builder.build_problem(tool_registry)

# Resolve Topology mapping design_vars automatically from the graph schema
analyzer = TopologicalAnalyzer(schema)
design_vars, _ = analyzer.resolve_dependencies(["z"])
parameters = analyzer.extract_parameters(design_vars)

# Run Optimization
evaluator = LocalEvaluator(prob)
optimizer = BayesianOptimizer(
    evaluator=evaluator,
    parameters=parameters,
    objectives=[{"name": "z", "minimize": True}],
)

result = optimizer.optimize(n_steps=10)
print(f"Best Result: {result['best_objectives']} at {result['best_parameters']}")
```

### 3. Running Tests

```bash
uv run pytest tests/
```

## Contributing

1.  Follow PEP 8 guidelines.
2.  Ensure 100% test coverage for new features.
3.  Use `uv run pre-commit run --all-files` before committing.

## License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0). See the [LICENSE](LICENSE) file for details.
