FROM python:3.12-slim

WORKDIR /app

# Install uv (via pip as curl|sh is blocked in some environments or just safer here)
RUN pip install uv

COPY pyproject.toml uv.lock ./
COPY README.md ./

# Copy source code BEFORE sync, as uv sync needs to build the package (editable install or source check)
# Since we moved to src/ layout, copy src/ to src/
COPY src/ src/

# Pass a dummy version to setuptools-scm during the docker build to prevent failures
# since the .git directory is explicitly not copied (bloat, layer caching, security).
# This image is primarily used for Security Scanning in the CI.
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"

# Install dependencies
RUN uv sync --frozen

# Default command (path adjusted to installed package location or src if editable)
# uv run automatically adds the environment to path.
# uvicorn needs to find the module. If installed as editable, it's in path.
CMD ["uv", "run", "uvicorn", "services.graph.main:app", "--host", "0.0.0.0", "--port", "8000"]
