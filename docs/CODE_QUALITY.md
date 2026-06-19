# Code Quality and Engineering Practices

## Production API Practices

- FastAPI application factory pattern via `create_app`.
- Separate liveness (`/health`) and readiness (`/ready`) checks.
- Typed request and response schemas with Pydantic.
- Batch prediction input bounded by `MAX_BATCH_RECORDS` to protect service resources.
- Feature-importance endpoint returns a typed response model.

## Reproducibility

- Dependencies are declared in `pyproject.toml`.
- `uv.lock` pins the resolved dependency graph.
- `scripts/quality.ps1` provides a Windows-friendly quality gate.
- GitHub Actions runs quality checks against Python 3.11 and 3.13.

## Testing

- API tests cover health, readiness, metadata, prediction, feature importance, validation
  errors, and missing-artifact behavior.
- Modeling tests cover training, test-evaluation guardrails, threshold handling,
  hyperparameter overrides, temporal validation, and inference transforms.
- Governance report tests cover fairness and monitoring report generation.

## MLOps Coverage

- MLflow logging hooks capture training metrics, parameters, reports, and plots.
- Optuna tuning command produces reproducible trial and best-parameter artifacts.
- Fairness and monitoring reports are generated from model artifacts rather than written
  manually.

## Local Quality Gate

Run:

```powershell
.\scripts\quality.ps1
```

This executes:

- `ruff check .`
- `ty check src tests`
- `pytest`
