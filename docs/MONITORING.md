# Model Monitoring Report

## Current Model Snapshot

- Train ROC-AUC: `0.9095`
- Validation ROC-AUC: `0.7844`
- Test ROC-AUC: `0.7858`
- Validation PR-AUC: `0.2793`
- Threshold used: `0.549`
- Validation Brier score: `0.1853`
- Validation expected calibration error: `0.3411`

## Prediction Drift

- Score PSI train vs validation: `0.001304` (stable)
- Score PSI train vs test: `0.000685` (stable)

## Feature Drift

| Feature | Train vs validation PSI | Severity | Train vs test PSI | Severity |
| --- | --- | --- | --- | --- |
| DAYS_BIRTH | 0.000223 | stable | 0.000682 | stable |
| DAYS_EMPLOYED | 0.000131 | stable | 0.000474 | stable |
| AMT_ANNUITY | 0.000208 | stable | 0.000438 | stable |
| CREDIT_INCOME_RATIO | 0.000051 | stable | 0.000324 | stable |
| AMT_INCOME_TOTAL | 0.000275 | stable | 0.000250 | stable |
| ANNUITY_INCOME_RATIO | 0.000262 | stable | 0.000205 | stable |
| AMT_CREDIT | 0.000208 | stable | 0.000251 | stable |

## Policy Monitoring

- Validation approval rate: `0.8590`
- Validation default rate within approved: `0.0507`
- Expected validation cost per applicant: `0.3216`

## Alert Rules

- PSI below 0.10: stable.
- PSI from 0.10 to 0.25: investigate moderate distribution shift.
- PSI above 0.25: block automatic promotion and perform model review.
- Recheck subgroup fairness metrics whenever threshold or training data changes.
- Recalibrate or retrain if calibration error or business cost worsens materially.
