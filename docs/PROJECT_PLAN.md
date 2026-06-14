# INSY684 Project Plan

## Migrated Baseline

The project now starts from the previous Home Credit Default Risk codebase. The
migrated baseline includes:

- Reproducible Python package under `src/homecredit_service`.
- Feature engineering for the main application table and auxiliary Home Credit
  tables.
- LightGBM model training with class-imbalance handling.
- Validation, final-evaluation guardrails, calibration, threshold optimization,
  cost-sensitivity analysis, temporal validation, subgroup checks, and drift
  diagnostics.
- FastAPI prediction endpoints and a local desktop inference app.
- Dockerfile, Makefile, pytest test suite, and GitHub Actions CI.
- Business, data, hypothesis, validation, explainability, results, and
  limitations documentation.

## Assignment Requirements Mapping

| Requirement area | Current status | Next action |
| --- | --- | --- |
| Same use case / dataset | Covered | Continue with Home Credit default risk. |
| GitHub repository | Covered | Push migrated structure to GitHub after review. |
| Unit tests and build | Covered | Keep expanding tests as new modules are added. |
| CI/CD | Partially covered | CI exists; add deployment workflow only if needed. |
| Docker containers | Covered | Verify image build after dependency changes. |
| Cloud native application | Partially covered | Add deployment documentation or lightweight cloud target. |
| Machine learning in production | Partially covered | Strengthen monitoring, model registry, and release workflow. |
| MLflow | Partially covered | Tracking hooks added; run experiments and document results. |
| Hyperparameter optimization / AutoML | Partially covered | Optuna command added; run tuning and compare with baseline. |
| Explainability | Covered | Keep global and local explanations; document governance use. |
| Fairness and ethical AI | Covered | Generated report exists; review and update after retraining. |
| Model tracking / drift | Covered | Generated monitoring report exists; review after retraining. |
| Final presentation | Not covered | Build final deck after code additions stabilize. |

## Proposed Work Order

1. Confirm the migrated baseline runs locally.
2. Run MLflow-tracked training and save experiment screenshots or exported run IDs.
3. Run Optuna tuning and compare best trial with baseline.
4. Review generated fairness and monitoring reports.
5. Add cloud deployment documentation.
6. Update docs and presentation slides.

## Team Information To Fill In

- Team name:
- GitHub repository:
- Team members:
- GitHub IDs:
