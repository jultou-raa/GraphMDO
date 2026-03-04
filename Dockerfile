FROM python:3.12-slim

# Grab uv directly from astral's image (faster and smaller than pip install)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Docker-specific uv optimizations
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Pass a dummy version to setuptools-scm during the docker build to prevent failures
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"

# Security: Create and use a non-root user
RUN useradd -m appuser

WORKDIR /app
RUN chown appuser:appuser /app

USER appuser

# Copy dependency files FIRST
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./

# Step 1: Install dependencies ONLY.
# Docker will CACHE this heavy step unless pyproject.toml or uv.lock changes.
RUN uv sync --frozen --no-install-project

# Step 2: Copy your source code (changes often)
COPY --chown=appuser:appuser src/ src/

# Step 3: Install the project itself (fast)
RUN uv sync --frozen

# Default command (path adjusted to installed package location or src if editable)
CMD ["uv", "run", "uvicorn", "services.graph.main:app", "--host", "0.0.0.0", "--port", "8000"]
