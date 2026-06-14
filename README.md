# INSY684 Group Project

Production-oriented machine learning project for Home Credit default risk
scoring.

This repository continues the team's course 1 project and reorganizes it for the
INSY684 group assignment. The current baseline is a FastAPI + LightGBM service
with training, validation, explainability, drift diagnostics, Docker packaging,
MLflow hooks, Optuna tuning, governance documentation, CI, and automated tests.

## Project Scope

The project predicts whether a loan applicant is likely to experience payment
difficulty using the Kaggle Home Credit Default Risk dataset.

Main goals:

- Build a reproducible credit-risk ML pipeline.
- Serve predictions through a tested API and local desktop app.
- Track validation, calibration, threshold, subgroup, and drift diagnostics.
- Extend the course 1 project with INSY684 topics such as MLflow tracking,
  hyperparameter optimization, CI/CD, fairness review, model monitoring, and
  deployment documentation.

## Repository Structure

```text
.
|-- src/homecredit_service/      # Training, feature engineering, API, inference
|-- tests/                       # Unit and API tests
|-- docs/                        # Business, data, validation, governance docs
|-- presentations/               # Final-deck technical section
|-- artifacts/                   # Baseline reports and plots
|-- Dockerfile                   # Containerized API runtime
|-- Makefile                     # Common project commands
|-- pyproject.toml               # Python package and tool configuration
|-- uv.lock                      # Locked dependency resolution
|-- requirements.txt             # pip-compatible dependency list
`-- .github/workflows/           # Quality and Docker build gates
```

## Submission Pointers

- Repository: `https://github.com/QingyuanWang-Data/INSY684_Group`
- Working branch: `code`
- Submission checklist: `docs/SUBMISSION_INFO.md`
- Project plan and requirement mapping: `docs/PROJECT_PLAN.md`
- Final technical deck section:
  `presentations/insy684-mlops-technical-section.pptx`

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

See `docs/EXPERIMENT_TRACKING.md` for the run protocol and evidence checklist.

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

## Serving

Start the FastAPI service:

```bash
make run
```

Useful endpoints:

- `GET /health`
- `GET /ready`
- `GET /metadata`
- `GET /feature-importance?limit=20`
- `POST /predict`
- `POST /predict/batch`

`/health` verifies that the API process is alive. `/ready` verifies that the
model bundle has been loaded; it will return unavailable until
`artifacts/model_bundle.joblib` exists locally or is mounted into the container.

## Docker

```bash
docker build -t insy684-homecredit-service .
docker run --rm -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts insy684-homecredit-service
```

The Docker image installs dependencies from `uv.lock` and expects the model
bundle to be provided through the mounted `artifacts` directory. See
`docs/DEPLOYMENT.md` for local, container, and cloud-native deployment notes.

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
- Docker image build gate in GitHub Actions
- MLflow logging hooks for training runs
- Optuna tuning command for LightGBM hyperparameters
- Generated fairness, monitoring, model-card, and code-quality reports

## Remaining Before Final Submission

- Fill in the final team name, member names, and GitHub IDs in
  `docs/SUBMISSION_INFO.md`.
- If the raw Kaggle data is available, rerun `make train-mlflow` and `make tune`
  so the latest experiment outputs can be shown in the final slides.
- Merge the technical section PPT into the group presentation deck.
