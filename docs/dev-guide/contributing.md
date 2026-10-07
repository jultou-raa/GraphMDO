# Contributing

We welcome contributions!

## Development Environment

1.  **Install `uv`**:
    See [astral.sh/uv](https://astral.sh/uv).

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/jultou-raa/GraphMDO.git
    cd GraphMDO
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
uv run pytest tests/                           # everything
uv run pytest -m "not e2e" tests/              # fast unit tests
OMP_NUM_THREADS=1 uv run pytest -m e2e tests/  # real Ax + GEMSEO runs (tests/e2e/)
```

Tests that pin an open bug are marked `xfail(strict=True)` with the issue number in
the reason. The PR that fixes the bug removes the marker.

Ensure you have 100% test coverage before submitting a PR.

## Documentation

To build and preview documentation locally:

```bash
uv run mkdocs serve
```

Documentation source files are located in `docs/`.
