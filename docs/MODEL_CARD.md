# Model Card

## Intended Use

This model estimates payment-difficulty risk for Home Credit loan applicants. It is intended for credit-risk decision support, portfolio monitoring, and policy simulation, not as a fully automated denial system.

## Data

- Rows: `307511`
- Columns: `452`
- Positive class rate: `0.0807`
- Negative class rate: `0.9193`

## Performance

- Train ROC-AUC: `0.9095`
- Validation ROC-AUC: `0.7844`
- Test ROC-AUC: `0.7858`
- Threshold used: `0.549`

## Key Parameters

| Parameter | Value |
| --- | --- |
| Objective | custom_focal_loss |
| Learning rate | 0.025 |
| Num leaves | 48 |
| Max depth | 12 |
| Imbalance strategy | focal_loss |

## Limitations

- Historical data may encode past lending and socioeconomic patterns.
- Subgroup gaps require monitoring before threshold or policy changes.
- Calibration should be reviewed before using scores as absolute probabilities.
- Production performance must be monitored after deployment for drift.

## Governance Controls

- Use `/ready` to confirm model artifact availability before routing traffic.
- Log training runs with MLflow for reproducibility.
- Recompute fairness and monitoring reports after retraining.
- Keep final credit decisions subject to policy and human review.
