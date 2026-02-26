# AGENTS.md

## Strict Coding Standards

1.  **PEP 20 (The Zen of Python)**
    -   Explicit is better than implicit.
    -   Simple is better than complex.
    -   Readability counts.

2.  **PEP 8 (Style Guide for Python Code)**
    -   Indentation: 4 spaces.
    -   Line Length: 88 characters.
    -   Naming Conventions:
        -   Functions/Variables: `lowercase_with_underscores`
        -   Classes: `CapitalizedWords`
        -   Constants: `ALL_CAPS_WITH_UNDERSCORES`
    -   Imports: Standard library, third-party, local application.
    -   Type Hinting: Use explicit Python 3 type hints.

## Project Architecture

-   **Phase 1**: Graph Database (FalkorDB).
-   **Phase 2**: Graph Formulation (KADMOS/OpenCypher).
-   **Phase 3**: Execution (OpenMDAO, SMT).
-   **Phase 5**: Constrained Optimization (BoTorch).

## Implementation Directives

-   **Native Graph Construction**: Use FalkorDB property graph.
-   **MDO Component Wrapping**: Ensure `setup()` and `setup_partials()` are clearly defined.
-   **Optimization Constraints**: Use epsilon-constraint methods.

## Dependency Management

-   This project uses `uv` for dependency management.
-   To add a dependency: `uv add <package_name>`
-   To run commands in the environment: `uv run <command>`
-   **Do not use pip install manually.**
