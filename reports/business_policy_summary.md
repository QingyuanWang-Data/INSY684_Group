# Business Policy Summary

## Purpose

This report translates the current model evidence from the technical branch into business-facing policy language. It is intended for the business lead, final presentation, and project documentation. It should be refreshed when model artifacts or threshold policy change.

## Model Evidence Snapshot

| Item | Current evidence |
| --- | ---: |
| Course 1 reference test ROC-AUC | `0.7858` |
| Course 2 final validation ROC-AUC | `0.7666` |
| Course 2 final test ROC-AUC | `0.7647` |
| Course 2 Optuna best validation ROC-AUC | `0.7679` |
| Course 2 test PR-AUC | `0.2303` |
| Selected threshold | `0.549` |
| Approval rate at selected threshold | `86.62%` |
| Approved default rate | `5.37%` |
| Expected cost per applicant | `0.3323` |
| Recall for payment-difficulty applicants | `42.24%` |
| Specificity for normal-repayment applicants | `89.15%` |

The final test ROC-AUC of `0.7647` confirms that the current Course 2 model ranks applicants meaningfully better than random on the held-out test split. The Course 2 result should still not be interpreted only as a score competition against the Course 1 baseline. The main Course 2 value is the production-oriented workflow: MLflow hooks, Optuna tuning, Docker packaging, CI checks, monitoring reports, fairness review, and deployment documentation.

## Threshold and Business Policy Interpretation

At threshold `0.549`, the current policy approves about `86.62%` of applicants in the validation sample, with an approved default rate of about `5.37%`. This supports a growth-oriented policy, but it still misses a meaningful share of risky applicants because recall is about `42.24%`.

The cost-sensitivity table shows the expected policy tradeoff:

| False negative cost | Selected threshold | Approval rate | Approved default rate | Recall |
| ---: | ---: | ---: | ---: | ---: |
| `2.5` | `0.6078` | `95.50%` | `6.68%` | `20.75%` |
| `5.0` | `0.5490` | `86.62%` | `5.37%` | `42.24%` |
| `7.5` | `0.5098` | `77.96%` | `4.59%` | `55.53%` |
| `10.0` | `0.4608` | `64.21%` | `3.38%` | `73.04%` |
| `15.0` | `0.4314` | `54.77%` | `2.72%` | `81.49%` |

When false negatives become more expensive, the selected threshold moves lower. Lower thresholds catch more risky applicants but reduce approval volume and increase manual review or stricter-treatment cases.

| Business objective | Likely threshold direction | Expected impact |
| --- | --- | --- |
| Protect approval volume and customer growth | Higher threshold | More approvals, but higher risk leakage |
| Reduce missed risky borrowers | Lower threshold | More risk capture, but more applicants delayed or rejected |
| Balance growth and risk | Three-band policy | Low risk auto-approve, medium risk review, high risk stricter treatment |

## Fairness and Governance Interpretation

The current subgroup report shows meaningful recall and approval-rate differences across applicant groups. Examples:

- Gender recall gap: female applicants have recall `0.3790`, while male applicants have recall `0.4822`, a gap of about `0.1032`.
- Income-type recall gap: pensioner applicants have recall `0.2400`, while working applicants have recall `0.4810`, a gap of about `0.2410`.
- Family-status recall gap: widow applicants have recall `0.1628`, while single / not married applicants have recall `0.4965`, a gap of about `0.3337`.

These gaps do not automatically prove unfairness, but they are strong reasons to avoid fully automated denial decisions. They should trigger review before any threshold is used in a real lending process.

Recommended controls:

- Keep human review for medium-risk and borderline cases.
- Recompute subgroup metrics after retraining, threshold changes, or data updates.
- Track recall, specificity, approval rate, and approved default rate by subgroup.
- Avoid presenting the model as a fully automated credit denial engine.

## Business Recommendation

Use the model as a credit-risk decision-support system rather than a fully automated lending decision engine. The recommended business framing is:

1. Low-risk band: consider auto-approval or standard terms.
2. Medium-risk band: route to manual review.
3. High-risk band: consider decline, lower credit limit, stricter pricing, or additional documentation.

## Branch Evidence Notes

- `code` is the current technical integration branch and contains the final-evaluation evidence used above.
- `data` contains the data processing and profiling work.
- `Extend` contains supplementary clustering and model-comparison experiments. Its best listed test ROC-AUC is `0.7692` for selected original features with LightGBM focal loss, while clustering-only features perform weakly. This is useful as extra experimentation evidence, but the main business report should continue to anchor on the integrated `code` branch unless the team explicitly merges the extension work.
- `business` contains the business lead documentation and policy interpretation.

Before final submission, this report should be checked against the final `main` branch results, final threshold policy, and any new fairness or monitoring outputs that the team generates.
