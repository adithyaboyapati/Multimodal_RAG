# ==============================================================================
# Multi-Stage Production Dockerfile for NovaCore Multimodal RAG Platform
# ==============================================================================

# Stage 1: Build stage
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final minimal runtime
FROM python:3.10-slim AS runtime

WORKDIR /app

# Install runtime dependencies for PyMuPDF, Pillow, and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Create unprivileged application user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/novacore_extracted_images && \
    chown -R appuser:appuser /app

# Copy application source and assets
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser docs/ ./docs/

USER appuser

EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Production ASGI server entrypoint
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
