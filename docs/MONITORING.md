# Model Monitoring Report

## Current Model Snapshot

- Train ROC-AUC: `0.9679`
- Validation ROC-AUC: `0.7619`
- Test ROC-AUC: `0.7644`
- Validation PR-AUC: `0.2545`
- Test PR-AUC: `0.2309`
- Threshold used: `0.549`
- Validation Brier score: `0.1708`
- Validation expected calibration error: `0.3173`
- Test Brier score: `0.1718`
- Test expected calibration error: `0.3188`

## Prediction Drift

- Score PSI train vs validation: `0.013916` (stable)
- Score PSI train vs test: `0.012282` (stable)

## Feature Drift

| Feature | Train vs validation PSI | Severity | Train vs test PSI | Severity |
| --- | --- | --- | --- | --- |
| AMT_CREDIT | 0.001828 | stable | 0.003718 | stable |
| CREDIT_INCOME_RATIO | 0.001472 | stable | 0.001170 | stable |
| ANNUITY_INCOME_RATIO | 0.001391 | stable | 0.004749 | stable |
| DAYS_EMPLOYED | 0.001322 | stable | 0.001101 | stable |
| AMT_ANNUITY | 0.001258 | stable | 0.001672 | stable |
| DAYS_BIRTH | 0.001066 | stable | 0.001731 | stable |
| AMT_INCOME_TOTAL | 0.000844 | stable | 0.002015 | stable |

## Policy Monitoring

- Validation approval rate: `0.8949`
- Validation default rate within approved: `0.0574`
- Expected validation cost per applicant: `0.3330`

## Alert Rules

- PSI below 0.10: stable.
- PSI from 0.10 to 0.25: investigate moderate distribution shift.
- PSI above 0.25: block automatic promotion and perform model review.
- Recheck subgroup fairness metrics whenever threshold or training data changes.
- Recalibrate or retrain if calibration error or business cost worsens materially.
