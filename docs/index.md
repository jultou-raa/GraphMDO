# GraphMDO Documentation

Welcome to the official documentation for **GraphMDO**, a dynamic Multidisciplinary Design Optimization (MDO) framework.

GraphMDO bridges data engineering and MDO. It extracts topological data (solvers, variables, fidelity levels) to form an oriented graph, specifically utilizing KADMOS for semantic formulation and exporting to CMDOWS. The execution is handled by OpenMDAO and the Surrogate Modeling Toolbox (SMT), driven by constrained Bayesian optimization ([ax-platform](https://ax.dev/)) or evolutionary algorithms ([pymoo](https://pymoo.org/)). The primary operational goal is to isolate and maximize a single target performance metric while strictly holding all other performance metrics constant.

## Key Features

- **Graph-Native Formulation**: Define variables, tools, and dependencies as nodes and edges in a property graph.
- **Dynamic Translation**: Automatically generate OpenMDAO execution graphs from the database state.
- **Multi-Fidelity Support**: Built-in support for Co-Kriging and other multi-fidelity surrogate models.
- **Constrained Bayesian Optimization**: Efficient global optimization with Gaussian Processes utilizing Ax Platform, supporting choices, range definitions, and strict constraints to isolate target metrics.
- **Microservices Architecture**: Decoupled Graph, Execution, and Optimization services for scalability.

## Quick Links

- [User Guide](user-guide/quick-start.md): Get started with GraphMDO.
- [Technical Reference](technical-reference/architecture.md): Understand the system architecture.
- [API Reference](api/core/translator.md): Explore the Python API.
- [Developer Guide](dev-guide/contributing.md): Contribute to the project.
