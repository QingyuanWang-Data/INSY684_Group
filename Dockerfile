FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HC_ARTIFACT_PATH=/app/artifacts/model_bundle.joblib

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY artifacts/model_bundle.joblib ./artifacts/model_bundle.joblib
RUN uv sync --locked --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

# --no-sync prevents an unexpected dependency resolution attempt at container startup.
CMD ["uv", "run", "--no-sync", "uvicorn", "homecredit_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
