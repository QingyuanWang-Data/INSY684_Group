# Experiment Tracking and Tuning

## MLflow Tracking

The training pipeline supports MLflow logging so model runs can be compared by
metrics, parameters, artifacts, and code version.

```bash
make train-mlflow
```

The command logs core training metrics, selected model parameters, and generated
artifacts. To inspect runs locally:

```bash
make mlflow-ui
```

Recommended evidence to capture for the final presentation:

- MLflow run ID
- validation AUC / PR-AUC
- calibration and threshold metrics
- artifact list
- screenshot or exported summary of the best run

## Hyperparameter Optimization

Optuna tuning is exposed through:

```bash
make tune
```

Expected outputs:

- `artifacts/tuning_report.json`
- `artifacts/tuning_trials.csv`

The tuning workflow optimizes LightGBM hyperparameters on a sampled development
split. The final model should still be evaluated through the existing holdout
guardrails before release.

## Current Evidence Status

The repository includes the implementation, tests, and baseline artifact reports.
Raw Kaggle files are intentionally not committed, so a fresh MLflow or Optuna run
requires adding the dataset locally under `homecreditdefaultriskdata/`.

If the team has the raw data available before submission, rerun:

```bash
make train-mlflow
make tune
make reports
```

Then include the new MLflow run summary and tuning comparison in the final deck.
This gives the presentation stronger evidence for the MLflow, model tracking,
and hyperparameter optimization parts of the assignment.
