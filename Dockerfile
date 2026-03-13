# Stage 1: Build stage
FROM python:3.12-slim AS builder

# Grab uv directly from astral's image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# Docker-specific uv optimizations
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Pass a dummy version to setuptools-scm during the docker build to prevent failures
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0"

WORKDIR /app

# Copy dependency files FIRST
COPY pyproject.toml uv.lock README.md ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-install-project

# Copy your source code
COPY src/ src/

# Install the project itself
RUN uv sync --frozen


# Stage 2: Final runtime stage
FROM python:3.12-slim

# Security: Ensure uv/uvx are NOT present in the final image
RUN rm -f /usr/bin/uv /usr/bin/uvx /bin/uv /bin/uvx

WORKDIR /app

# Security: Create and use a non-root user
RUN useradd -m appuser
RUN chown appuser:appuser /app
USER appuser

# Copy the virtual environment and the source code from the builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src

# Set environment variables to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["uvicorn", "services.graph.main:app", "--host", "0.0.0.0", "--port", "8000"]
