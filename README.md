# INSY684 Group Project

Production-oriented machine learning project for Home Credit default risk scoring.

This repository continues the team's course 1 project and reorganizes it for the
INSY684 group assignment. The current baseline is a FastAPI + LightGBM service
with training, validation, explainability, drift diagnostics, Docker packaging,
and automated tests.

## Project Scope

The project predicts whether a loan applicant is likely to experience payment
difficulty using the Kaggle Home Credit Default Risk dataset.

Main goals:

- Build a reproducible credit-risk ML pipeline.
- Serve predictions through an API and local desktop app.
- Track validation, calibration, threshold, subgroup, and drift diagnostics.
- Extend the project with INSY684 topics such as MLflow tracking, hyperparameter
  optimization, CI/CD, fairness review, and deployment documentation.

## Repository Structure

```text
.
├── src/homecredit_service/      # Training, feature engineering, API, inference service
├── tests/                       # Unit and API tests
├── docs/                        # Business, data, validation, explainability, and planning docs
├── presentations/               # Future weekly/final presentation materials
├── Dockerfile                   # Containerized API runtime
├── Makefile                     # Common project commands
├── pyproject.toml               # Python package and tool configuration
├── requirements.txt             # pip-compatible dependency list
└── .github/workflows/ci.yml     # GitHub Actions quality gate
```

## Data

Raw Kaggle data is not committed to GitHub. Download the Home Credit Default Risk
files from Kaggle and place them in:

```text
homecreditdefaultriskdata/
```

Required files:

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `POS_CASH_balance.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`

## Setup

Recommended with `uv`:

```bash
make install
```

Alternative with pip:

```bash
pip install -r requirements.txt
```

## Quality Checks

```bash
make lint
make typecheck
make test
```

Run the full local gate:

```bash
make quality
```

On Windows without `make`, run the PowerShell equivalent:

```powershell
.\scripts\quality.ps1
```

## Training

Development training:

```bash
make train
```

Final holdout evaluation:

```bash
make final-eval
```

Quick sampled run:

```bash
uv run train-homecredit --data-dir homecreditdefaultriskdata --artifact-dir artifacts --sample-size 5000
```

MLflow-tracked training:

```bash
make train-mlflow
```

Open the local MLflow UI:

```bash
make mlflow-ui
```

## Hyperparameter Tuning

Run Optuna tuning on a sampled training set:

```bash
make tune
```

The tuning command writes:

- `artifacts/tuning_report.json`
- `artifacts/tuning_trials.csv`

## Governance Reports

Generate fairness and monitoring reports from saved artifacts:

```bash
make reports
```

On Windows without `make`:

```powershell
uv run generate-governance-reports --artifact-dir artifacts --docs-dir docs
```

The command writes:

- `docs/FAIRNESS.md`
- `docs/MONITORING.md`
- `docs/MODEL_CARD.md`

## Code Quality

See `docs/CODE_QUALITY.md` for the engineering practices used in this repository,
including typed APIs, readiness checks, CI matrix testing, bounded batch inference,
MLflow tracking, Optuna tuning, and generated governance reports.

## Serving

Start the FastAPI service:

```bash
make run
```

Useful endpoints:

- `GET /health`
- `GET /metadata`
- `GET /feature-importance?limit=20`
- `POST /predict`
- `POST /predict/batch`

## Docker

```bash
docker build -t insy684-homecredit-service .
docker run --rm -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts insy684-homecredit-service
```

## Current Baseline Coverage

Already migrated from the course 1 project:

- Feature engineering and multi-table aggregation
- Advanced missing-value handling and encoding
- LightGBM classifier with imbalance handling
- Train/validation/test split and gated final evaluation
- Cross-validation and temporal holdout diagnostics
- Calibration, threshold optimization, and cost-sensitivity analysis
- Subgroup robustness and PSI-based drift diagnostics
- FastAPI prediction service
- Local desktop inference app
- Dockerfile, Makefile, pytest tests, and GitHub Actions CI
- MLflow logging hooks for training runs
- Optuna tuning command for LightGBM hyperparameters

## Next INSY684 Additions

Planned next implementation work:

- Run and document MLflow experiment results
- Run and document Optuna tuning results
- Review generated fairness and monitoring reports
- Cloud deployment notes or deployment workflow
- Final presentation deck

See `docs/PROJECT_PLAN.md` for the working checklist.
