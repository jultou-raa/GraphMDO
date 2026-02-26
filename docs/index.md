# GraphMDO Documentation

Welcome to the official documentation for **GraphMDO**, a dynamic Multidisciplinary Design Optimization (MDO) framework.

GraphMDO leverages modern graph databases ([FalkorDB](https://falkordb.com/)) to store engineering problem definitions natively as property graphs. It seamlessly translates these graph structures into executable optimization problems using [OpenMDAO](https://openmdao.org/), supports multi-fidelity surrogate modeling via [SMT](https://smt.readthedocs.io/), and performs advanced constrained optimization using [BoTorch](https://botorch.org/).

## Key Features

- **Graph-Native Formulation**: Define variables, tools, and dependencies as nodes and edges in a property graph.
- **Dynamic Translation**: Automatically generate OpenMDAO execution graphs from the database state.
- **Multi-Fidelity Support**: Built-in support for Co-Kriging and other multi-fidelity surrogate models.
- **Bayesian Optimization**: Efficient global optimization with Gaussian Processes and acquisition functions.
- **Microservices Architecture**: Decoupled Graph, Execution, and Optimization services for scalability.

## Quick Links

- [User Guide](user-guide/quick-start.md): Get started with GraphMDO.
- [Technical Reference](technical-reference/architecture.md): Understand the system architecture.
- [API Reference](api/core/translator.md): Explore the Python API.
- [Developer Guide](dev-guide/contributing.md): Contribute to the project.
