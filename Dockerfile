FROM python:3.12-slim

WORKDIR /app

# Install uv (via pip as curl|sh is blocked in some environments or just safer here)
RUN pip install uv

# Copy dependency files FIRST
COPY pyproject.toml uv.lock README.md ./

# Pass a dummy version to setuptools-scm during the docker build to prevent failures
# since the .git directory is explicitly not copied (bloat, layer caching, security).
# This image is primarily used for Security Scanning in the CI.
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"

# Step 1: Install dependencies ONLY.
# Docker will CACHE this heavy step unless pyproject.toml or uv.lock changes.
RUN uv sync --frozen --no-install-project

# Step 2: Copy your source code (changes often)
COPY src/ src/

# Step 3: Install the project itself (fast)
RUN uv sync --frozen

# Default command (path adjusted to installed package location or src if editable)
# uv run automatically adds the environment to path.
# uvicorn needs to find the module. If installed as editable, it's in path.
CMD ["uv", "run", "uvicorn", "services.graph.main:app", "--host", "0.0.0.0", "--port", "8000"]
