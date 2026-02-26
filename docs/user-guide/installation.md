# Installation

## Prerequisites

- Python 3.12+
- Docker (optional but recommended for FalkorDB and microservices)

## Install `uv`

We recommend using `uv` for dependency management. See [astral.sh/uv](https://astral.sh/uv) for installation instructions.

## Clone and Setup

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/your-repo/graph-mdo.git
    cd graph-mdo
    ```

2.  **Install Dependencies**

    ```bash
    uv sync
    ```

    This will create a virtual environment and install all dependencies specified in `pyproject.toml`.

## Setup FalkorDB

FalkorDB is required for graph storage. The easiest way to run it is via Docker.

```bash
docker run -p 6379:6379 -it falkordb/falkordb
```

Alternatively, you can use the provided `docker-compose.yml` to spin up the entire stack.

```bash
docker-compose up -d
```

This will launch:
- FalkorDB (Port 6379)
- Graph Service (Port 8001)
- Execution Service (Port 8002)
- Optimization Service (Port 8003)
