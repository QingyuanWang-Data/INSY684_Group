# Business Problem and Success Metrics

## Business Problem

Home Credit provides loans to applicants who may have limited or non-traditional credit
history. This creates a business challenge: the lender needs to approve enough reliable
borrowers to support growth, while also controlling credit losses from applicants who are more
likely to have payment difficulties.

The project uses the Home Credit Default Risk dataset to support this decision. The model is
intended to estimate default risk for each applicant, where:

- `TARGET = 1` means the applicant had payment difficulties.
- `TARGET = 0` means the applicant repaid normally.

The business goal is not to fully automate lending decisions. The goal is to create a risk score
that can help underwriting and risk teams make more consistent, data-supported decisions.

## Intended Business Use

The model output should be used as a decision-support signal. A practical workflow could be:

- Low-risk applicants: eligible for auto-approval or standard loan terms.
- Medium-risk applicants: sent to manual review.
- High-risk applicants: declined, offered stricter terms, or asked for additional information.

This approach is more realistic than using one fixed approve-or-decline rule. It allows the
business to balance credit loss, approval rate, customer experience, and governance review.

## Key Stakeholders

- Credit risk team: uses the score to understand portfolio risk and adjust policy.
- Underwriting team: uses the score to support application review.
- Data science team: builds, validates, monitors, and improves the model.
- Compliance and governance stakeholders: review fairness, explainability, and model risk.
- Applicants: are affected by the lending decision and should not be harmed by unfair or
  unexplained model behavior.

## Machine Learning Success Metrics

The project should not rely on accuracy as the primary metric because the dataset is highly
imbalanced. Most applicants repay normally, so a model can appear accurate while failing to
identify risky applicants.

Recommended machine learning metrics:

- ROC-AUC: measures how well the model ranks risky applicants above lower-risk applicants.
- PR-AUC: useful because default cases are the minority class.
- Recall for `TARGET = 1`: measures how many risky applicants the model captures.
- Precision for `TARGET = 1`: measures how many flagged applicants are actually risky.
- Calibration: checks whether predicted probabilities match observed default rates.

## Business Success Metrics

The model should also be evaluated through business outcomes, not only technical scores.

Recommended business metrics:

- Approval rate: the percentage of applicants approved under the decision policy.
- Default rate among approved applicants: risk level of the approved portfolio.
- Expected credit loss: estimated loss after considering default probability and loss severity.
- Manual review rate: percentage of applications requiring human review.
- False negative cost: cost of approving a risky borrower.
- False positive cost: cost of rejecting or delaying a reliable borrower.

These metrics help translate model performance into business value.

## Decision Threshold Considerations

The model produces a probability score. A threshold turns that score into an action, but the
threshold should be treated as a business policy choice rather than a purely technical setting.

If the business wants to reduce missed risky borrowers, it may choose a lower threshold. This can
catch more high-risk applicants, but it may also send more reliable borrowers to review or
decline.

If the business wants to protect approval volume and customer growth, it may choose a higher
threshold. This can reduce false positives, but it may allow more risky borrowers to be approved.

A three-band policy is likely more useful than a single threshold:

| Risk band | Example action | Business rationale |
| --- | --- | --- |
| Low risk | Auto-approve or standard terms | Protect growth and reduce review workload |
| Medium risk | Manual review | Use human judgment where model confidence is less clear |
| High risk | Decline, stricter terms, or additional checks | Reduce expected credit losses |

## Fairness and Ethical AI Considerations

Credit decisions affect access to financial products, so fairness should be part of the project
evaluation. The model should be checked for performance differences across applicant groups when
group information is available and appropriate to review.

Recommended fairness checks:

- Compare approval rates across groups.
- Compare recall for risky applicants across groups.
- Compare false positive rates across groups.
- Review whether explanations are reasonable and business-relevant.
- Avoid using the model as the only decision-maker without policy and governance oversight.

The project should communicate that model predictions support decision-making but do not remove
the need for human review, business policy, and compliance judgment.

## Course 2 Improvement Focus

The previous project already created a working credit risk model and reported a test ROC-AUC of
about `0.7858`. Course 2 should focus on improving the project beyond basic model training.

Recommended Course 2 improvements:

- Add experiment tracking so model parameters, metrics, and artifacts are easier to compare.
- Strengthen monitoring for data drift, score drift, and model performance changes.
- Expand fairness analysis and explainability discussion.
- Make the deployment and model-serving workflow clearer.
- Connect threshold selection to business costs and approval policy.
- Maintain GitHub collaboration through branches, commits, pull requests, and CI checks.

## Current Course 2 Evidence to Incorporate

The technical branch now includes a sampled MLflow training run, Optuna tuning evidence,
fairness report, monitoring report, and deployment documentation.

Current evidence includes:

- Course 2 sampled MLflow validation ROC-AUC: `0.7619`.
- Course 2 sampled validation PR-AUC: `0.2545`.
- Optuna best sampled validation ROC-AUC: `0.7679`.
- Selected threshold: `0.549`.
- Validation approval rate at selected threshold: `89.49%`.
- Approved default rate at selected threshold: `5.74%`.
- Expected validation cost per applicant: `0.3330`.

These results should be presented carefully. The Course 2 run is not only a model-score
comparison against the Course 1 reference result. It also adds production-oriented capabilities
such as experiment tracking, hyperparameter tuning, monitoring, fairness review, Docker, CI, and
deployment documentation.

## Business Lead Contribution

The business lead should ensure the final project answers these questions clearly:

1. What business problem does the model solve?
2. Who will use the model output?
3. How does the score become a lending decision?
4. Which metrics define success?
5. What are the risks of using the model?
6. How should fairness, explainability, and monitoring be handled?
7. What should be improved before the model is used in a real lending process?
