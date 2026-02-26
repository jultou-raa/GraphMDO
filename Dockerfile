FROM python:3.12-slim

WORKDIR /app

# Install uv (via pip as curl|sh is blocked in some environments or just safer here)
RUN pip install uv

COPY pyproject.toml uv.lock ./
COPY README.md ./

# Install dependencies
RUN uv sync --frozen

COPY mdo_framework/ mdo_framework/
COPY services/ services/

# Default command
CMD ["uv", "run", "uvicorn", "services.graph.main:app", "--host", "0.0.0.0", "--port", "8000"]
