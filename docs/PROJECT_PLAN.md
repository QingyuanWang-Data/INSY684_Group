# INSY684 Project Plan

## Migrated Baseline

The project starts from the previous Home Credit Default Risk codebase and
extends it for the INSY684 group assignment. The migrated baseline includes:

- Reproducible Python package under `src/homecredit_service`.
- Feature engineering for the main application table and auxiliary Home Credit
  tables.
- LightGBM model training with class-imbalance handling.
- Validation, final-evaluation guardrails, calibration, threshold optimization,
  cost-sensitivity analysis, temporal validation, subgroup checks, and drift
  diagnostics.
- FastAPI prediction endpoints and a local desktop inference app.
- Dockerfile, Makefile, pytest test suite, GitHub Actions CI, and Docker build
  gate.
- Business, data, hypothesis, validation, explainability, results, limitations,
  fairness, monitoring, model-card, and deployment documentation.

## Assignment Requirements Mapping

| Requirement area | Current status | Evidence / final action |
| --- | --- | --- |
| Same use case / dataset | Covered | Continued Home Credit default risk use case from course 1. |
| GitHub repository | Covered | Repository and branch listed in `docs/SUBMISSION_INFO.md`. |
| Unit tests and build | Covered | `tests/`, ruff, ty, pytest, and GitHub Actions CI. |
| CI/CD | Covered for CI and build packaging | CI quality gate and Docker build gate exist; live deployment workflow can be added after cloud target is assigned. |
| Docker containers | Covered | `Dockerfile`, `.dockerignore`, locked `uv` install, and runtime artifact mount. |
| Cloud native application | Documented | `docs/DEPLOYMENT.md` describes local, Docker, readiness, and cloud runtime target. |
| Machine learning in production | Covered | FastAPI service, readiness checks, model artifact contract, and monitoring docs. |
| MLflow | Implemented | Tracking hooks and `make train-mlflow`; fresh run requires local Kaggle data. |
| Hyperparameter optimization / AutoML | Implemented | Optuna command and output contract documented in `docs/EXPERIMENT_TRACKING.md`. |
| Explainability | Covered | Global feature importance and governance explanation in docs. |
| Fairness and ethical AI | Covered | `docs/FAIRNESS.md` and `docs/MODEL_CARD.md`. |
| Model tracking / drift | Covered | `docs/MONITORING.md`, PSI drift diagnostics, and generated reports. |
| Final presentation | Covered | Reviewed PPTX available in `presentations/` with final metrics and team context. |

## Final Release Procedure

1. If raw Kaggle data is available locally, rerun MLflow training and Optuna
   tuning:

   ```bash
   make train-mlflow
   make tune
   make reports
   ```

2. Run the local quality gate:

   ```powershell
   .\scripts\quality.ps1
   ```

3. Confirm the final `code` branch is synchronized with GitHub.

## Team Information

- Team name: Home Credit Risk MLOps Team
- GitHub repository: `https://github.com/QingyuanWang-Data/INSY684_Group`
- Working branch: `code`
- Team members, roles and GitHub IDs: see `docs/SUBMISSION_INFO.md`
