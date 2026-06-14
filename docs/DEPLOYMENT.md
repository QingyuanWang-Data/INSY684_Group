# Deployment Notes

## Runtime Shape

The production-facing component is a FastAPI service packaged as
`homecredit_service.main:app`. It exposes health, readiness, metadata, feature
importance, single-prediction, and batch-prediction endpoints.

The service separates process health from model readiness:

- `GET /health` confirms that the API process is running.
- `GET /ready` confirms that `artifacts/model_bundle.joblib` has been loaded.

This matters in production because the container can start before a model bundle
is available. A deployment platform should route traffic only after `/ready`
returns success.

## Local Run

```bash
make run
```

For Windows without `make`:

```powershell
uv run uvicorn homecredit_service.main:app --host 0.0.0.0 --port 8000
```

## Docker Run

```bash
docker build -t insy684-homecredit-service .
docker run --rm -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts insy684-homecredit-service
```

The container installs dependencies from `uv.lock` with `uv sync --locked` for
reproducibility. The model artifact is not baked into the image; it is mounted at
runtime through `/app/artifacts` so model releases can be separated from image
releases.

## Cloud-Native Target

A lightweight cloud deployment can use any container runtime that supports:

- Docker image build and registry push
- One HTTP service port, `8000`
- Environment variable `HC_ARTIFACT_PATH=/app/artifacts/model_bundle.joblib`
- Persistent storage or object-store download for the model bundle
- Health check on `/health`
- Readiness check on `/ready`

Suitable platforms include Azure Container Apps, Google Cloud Run, AWS ECS
Fargate, Render, or Railway. For this course project, the repository provides the
container, dependency lock, readiness contract, CI quality gate, and Docker build
gate. A live cloud
deployment can be added once credentials and a deployment target are assigned to
the team.

## Release Checklist

1. Run `.\scripts\quality.ps1` or `make quality`.
2. Train or select the approved model bundle.
3. Store `artifacts/model_bundle.joblib` in the release artifact location.
4. Build the Docker image.
5. Deploy the image with the artifact mounted or downloaded at startup.
6. Confirm `/health`, `/ready`, and a sample `/predict` request.
7. Record model version, data snapshot, metrics, and known limitations in the
   model card.
