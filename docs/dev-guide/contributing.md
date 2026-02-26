# Contributing

We welcome contributions!

## Development Environment

1.  **Install `uv`**:
    See [astral.sh/uv](https://astral.sh/uv).

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/graph-mdo.git
    cd graph-mdo
    ```

3.  **Install Dependencies (including dev)**:
    ```bash
    uv sync --all-extras --dev
    ```

## Code Style

We use `ruff` for linting and formatting.

-   **Check**: `uv run ruff check .`
-   **Format**: `uv run ruff format .`

## Running Tests

Tests are written using `pytest`.

```bash
uv run pytest tests/
```

Ensure you have 100% test coverage before submitting a PR.

## Documentation

To build and preview documentation locally:

```bash
uv run mkdocs serve
```

Documentation source files are located in `docs/`.
