# Model Monitoring Report

## Current Model Snapshot

- Train ROC-AUC: `0.9160`
- Validation ROC-AUC: `0.7666`
- Test ROC-AUC: `0.7647`
- Validation PR-AUC: `0.2588`
- Threshold used: `0.549`
- Validation Brier score: `0.1883`
- Validation expected calibration error: `0.3448`

## Prediction Drift

- Score PSI train vs validation: `0.002662` (stable)
- Score PSI train vs test: `0.003003` (stable)

## Feature Drift

| Feature | Train vs validation PSI | Severity | Train vs test PSI | Severity |
| --- | --- | --- | --- | --- |
| ANNUITY_INCOME_RATIO | 0.001391 | stable | 0.004749 | stable |
| AMT_CREDIT | 0.001828 | stable | 0.003718 | stable |
| AMT_INCOME_TOTAL | 0.000844 | stable | 0.002015 | stable |
| DAYS_BIRTH | 0.001066 | stable | 0.001731 | stable |
| AMT_ANNUITY | 0.001258 | stable | 0.001672 | stable |
| CREDIT_INCOME_RATIO | 0.001472 | stable | 0.001170 | stable |
| DAYS_EMPLOYED | 0.001322 | stable | 0.001101 | stable |

## Policy Monitoring

- Validation approval rate: `0.8662`
- Validation default rate within approved: `0.0537`
- Expected validation cost per applicant: `0.3323`

## Alert Rules

- PSI below 0.10: stable.
- PSI from 0.10 to 0.25: investigate moderate distribution shift.
- PSI above 0.25: block automatic promotion and perform model review.
- Recheck subgroup fairness metrics whenever threshold or training data changes.
- Recalibrate or retrain if calibration error or business cost worsens materially.
