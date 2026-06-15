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

The repository includes the implementation, tests, baseline artifact reports,
and a fresh sampled final-evaluation run completed on June 15, 2026.

Sampled MLflow training run:

- Run name: `sampled-50000-final-test`
- Portable evidence: committed `artifacts/training_report.json`, plots, and
  governance reports; local MLflow run IDs remain machine-specific.
- Sample size: `50,000` application rows
- Feature count: `450`
- Train / validation / test split rows: `35,000 / 10,000 / 5,000`
- Validation ROC-AUC: `0.7619`
- Test ROC-AUC: `0.7644`
- Test PR-AUC: `0.2309`
- Train ROC-AUC: `0.9679`
- Selected decision threshold: `0.549`
- Output evidence: `artifacts/training_report.json` and `artifacts/plots/`

Optuna tuning run:

- Sample size: `20,000`
- Trials: `10`
- Timeout: `1,800` seconds
- Best validation ROC-AUC: `0.7679`
- Best trial: `9`
- Best parameters:
  `n_estimators=387`, `learning_rate=0.0199`, `num_leaves=28`,
  `min_child_samples=64`, `subsample=0.7638`,
  `colsample_bytree=0.8918`, `reg_alpha=0.3825`,
  `reg_lambda=1.7857`, `max_depth=9`
- Output evidence: `artifacts/tuning_report.json` and
  `artifacts/tuning_trials.csv`

The local `mlruns/` directory and `mlflow.db` are intentionally ignored because
they are machine-specific runtime tracking stores. The committed JSON, CSV,
plots, and governance reports provide the portable evidence for review.

To refresh the evidence before submission, rerun:

```bash
make train-mlflow
make tune
make reports
```

Then include the MLflow run summary and tuning comparison in the final deck. This
gives the presentation stronger evidence for the MLflow, model tracking, and
hyperparameter optimization parts of the assignment.
