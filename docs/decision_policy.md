# Decision Policy

## Purpose

This document explains how the model's default risk score can support lending decisions. The
model should be treated as a decision-support tool, not as a fully automated replacement for
human judgment or credit policy.

## Model Output

The model estimates the probability that a loan applicant may experience payment difficulties.

The target definition is:

- `TARGET = 1`: applicant has payment difficulties.
- `TARGET = 0`: applicant repays normally.

The model output can be interpreted as a risk score. A higher score means higher predicted
default risk.

## Why a Policy Is Needed

A model score by itself does not tell the business what action to take. The business needs a
policy that connects the score to an operational decision.

For this project, the decision policy should balance:

- Reducing credit losses from risky borrowers.
- Avoiding unnecessary rejection of reliable borrowers.
- Maintaining a reasonable approval rate.
- Keeping manual review workload manageable.
- Supporting explainability and fairness review.

## Recommended Three-Band Policy

A three-band policy is more practical than a single approve-or-decline threshold.

| Risk band | Model score interpretation | Suggested action | Business reason |
| --- | --- | --- | --- |
| Low risk | Applicant appears unlikely to have payment difficulties | Auto-approve or offer standard terms | Supports growth and reduces manual review workload |
| Medium risk | Applicant has uncertain or moderate risk | Send to manual review | Uses human judgment where the model is less decisive |
| High risk | Applicant appears more likely to have payment difficulties | Decline, reduce credit limit, request more documents, or offer stricter terms | Reduces expected credit losses |

The exact score cutoffs should be selected after model validation and business cost analysis.

## Threshold Tradeoff

The decision threshold is a business choice. It should not be selected only because it gives the
highest technical metric.

If the threshold is lower:

- More applicants may be classified as risky.
- The model may catch more truly risky borrowers.
- More reliable borrowers may be delayed, reviewed, or rejected.

If the threshold is higher:

- Fewer applicants may be classified as risky.
- Approval volume may be protected.
- More risky borrowers may be approved.

This tradeoff should be evaluated using both machine learning metrics and business metrics.

## Current Threshold Evidence

The current technical branch uses threshold `0.549` for validation policy evidence. At this
threshold:

- Final test ROC-AUC: `0.7647`.
- Test PR-AUC: `0.2303`.
- Approval rate: `86.62%`.
- Approved default rate: `5.37%`.
- Recall for payment-difficulty applicants: `42.24%`.
- Specificity for normal-repayment applicants: `89.15%`.
- Expected validation cost per applicant: `0.3323`.

The cost-sensitivity evidence shows that when false negatives become more expensive, the
selected threshold moves lower. This catches more risky applicants but reduces approval volume.
For example, a false-negative cost of `15.0` selects threshold `0.4314`, with approval rate
`54.77%` and recall `81.49%`. This is much more risk-control oriented than the current
threshold `0.549`.

## Metrics for Policy Evaluation

The team should evaluate candidate thresholds using:

- Approval rate.
- Manual review rate.
- Default rate among approved applicants.
- Expected credit loss.
- Recall for `TARGET = 1`.
- Precision for `TARGET = 1`.
- False positive rate.
- False negative rate.
- Subgroup-level performance and approval rates.

These metrics help explain how a technical threshold affects business outcomes.

## Business Cost Interpretation

Two error types are important:

- False negative: a risky applicant is treated as low risk.
- False positive: a reliable applicant is treated as high risk.

False negatives can lead to defaults and financial losses. False positives can lead to lost
revenue, poor customer experience, and fairness concerns.

The final policy should acknowledge both types of errors. In lending, false negatives often have
a direct loss impact, but false positives can also be costly because they may reject good
customers.

## Governance Recommendation

The model should support credit decisions, but it should not make final decisions without
governance. Before real business use, the team would need:

- Clear documentation of model purpose and limitations.
- Review of important features and explanations.
- Fairness checks across available applicant groups.
- Monitoring for data drift and model performance changes.
- Human review for medium-risk or unclear cases.
- Periodic threshold review as economic conditions and business goals change.

## Current Project Position

At the current stage, the decision policy is supported by validation evidence and final held-out
test evidence from the technical branch. Before final submission, this document should be
checked against the final `main` branch version with:

- The selected model version.
- The selected threshold or risk band cutoffs.
- Final validation and test performance at the selected policy.
- Business impact estimates.
- Fairness and subgroup findings.
