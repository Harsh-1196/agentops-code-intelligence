# Multi-stage production container for AgentOps backend
FROM python:3.11-slim AS backend

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and install Python dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install hatchling && \
    pip install .

# Copy application source code
COPY agentops/ agentops/
COPY evaluation/ evaluation/
COPY docs/ docs/
COPY README.md LICENSE NOTICE ./

# Create data directory for persistent SQLite
RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "agentops.api:app", "--host", "0.0.0.0", "--port", "8000"]
