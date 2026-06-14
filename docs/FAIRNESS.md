# Fairness and Ethical AI Report

## Scope

This report reviews subgroup behavior for the Home Credit default-risk model. The model estimates payment-difficulty probability, so fairness review focuses on risk-ranking quality, recall, specificity, approval rate, and default rate within approved applicants.

## Model Context

- Validation ROC-AUC: `0.7619`
- Test ROC-AUC: `n/a`
- Decision threshold used: `0.549`

## Gender Subgroup Snapshot

| Group | Count | Positive rate | AUC | Recall | Approval rate | Approved default rate |
| --- | --- | --- | --- | --- | --- | --- |
| F | 6618 | 0.0706 | 0.7636 | 0.3126 | 0.9184 | 0.0528 |
| M | 3382 | 0.0999 | 0.7489 | 0.4290 | 0.8489 | 0.0672 |

## Largest Validation Gaps

| Column | Metric | Lowest group | Highest group | Gap |
| --- | --- | --- | --- | --- |
| NAME_FAMILY_STATUS | recall | Widow (0.1395) | Single / not married (0.4266) | 0.2870 |
| NAME_INCOME_TYPE | recall | Pensioner (0.1800) | Working (0.4188) | 0.2388 |
| NAME_EDUCATION_TYPE | recall | Higher education (0.2256) | Secondary / secondary special (0.3876) | 0.1620 |
| NAME_INCOME_TYPE | pr_auc | Pensioner (0.1467) | Working (0.2925) | 0.1458 |
| CODE_GENDER | recall | F (0.3126) | M (0.4290) | 0.1164 |
| NAME_FAMILY_STATUS | approval_rate | Single / not married (0.8562) | Widow (0.9472) | 0.0910 |
| NAME_INCOME_TYPE | approval_rate | Working (0.8641) | Pensioner (0.9482) | 0.0841 |
| NAME_EDUCATION_TYPE | approval_rate | Secondary / secondary special (0.8757) | Higher education (0.9548) | 0.0791 |

## Largest Test Gaps

| Column | Metric | Lowest group | Highest group | Gap |
| --- | --- | --- | --- | --- |
| NAME_INCOME_TYPE | recall | Pensioner (0.2757) | Working (0.5234) | 0.2477 |
| NAME_CONTRACT_TYPE | recall | Revolving loans (0.2420) | Cash loans (0.4811) | 0.2390 |
| NAME_FAMILY_STATUS | recall | Widow (0.3043) | Single / not married (0.5248) | 0.2205 |
| NAME_EDUCATION_TYPE | recall | Higher education (0.3065) | Secondary / secondary special (0.4977) | 0.1912 |
| NAME_CONTRACT_TYPE | pr_auc | Revolving loans (0.1453) | Cash loans (0.2910) | 0.1458 |

## Ethical AI Interpretation

The model should not be used as a fully automated credit denial engine. Its scores are best treated as decision-support signals paired with policy rules, manual review for borderline cases, and regular subgroup monitoring. Gaps in recall or approval rate should trigger review because they can change who receives credit access or who is escalated for additional documentation.

## Recommended Controls

- Recompute subgroup metrics after each retraining run.
- Track recall, specificity, approval rate, and approved default rate by subgroup.
- Investigate large subgroup gaps before deployment or threshold changes.
- Keep human review for high-impact credit decisions.
- Document approved model thresholds and business cost assumptions.
