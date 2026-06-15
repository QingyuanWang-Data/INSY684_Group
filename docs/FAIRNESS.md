# Fairness and Ethical AI Report

## Scope

This report reviews subgroup behavior for the Home Credit default-risk model. The model estimates payment-difficulty probability, so fairness review focuses on risk-ranking quality, recall, specificity, approval rate, and default rate within approved applicants.

## Model Context

- Validation ROC-AUC: `0.7666`
- Test ROC-AUC: `0.7647`
- Decision threshold used: `0.549`

## Gender Subgroup Snapshot

| Group | Count | Positive rate | AUC | Recall | Approval rate | Approved default rate |
| --- | --- | --- | --- | --- | --- | --- |
| F | 6618 | 0.0706 | 0.7698 | 0.3790 | 0.8909 | 0.0492 |
| M | 3382 | 0.0999 | 0.7507 | 0.4822 | 0.8179 | 0.0633 |

## Largest Validation Gaps

| Column | Metric | Lowest group | Highest group | Gap |
| --- | --- | --- | --- | --- |
| NAME_FAMILY_STATUS | recall | Widow (0.1628) | Single / not married (0.4965) | 0.3337 |
| NAME_INCOME_TYPE | recall | Pensioner (0.2400) | Working (0.4810) | 0.2410 |
| NAME_EDUCATION_TYPE | recall | Higher education (0.2707) | Secondary / secondary special (0.4545) | 0.1839 |
| NAME_INCOME_TYPE | pr_auc | Pensioner (0.1559) | Working (0.2929) | 0.1370 |
| NAME_FAMILY_STATUS | approval_rate | Single / not married (0.8149) | Widow (0.9295) | 0.1146 |
| NAME_INCOME_TYPE | approval_rate | Working (0.8277) | Pensioner (0.9340) | 0.1063 |
| CODE_GENDER | recall | F (0.3790) | M (0.4822) | 0.1032 |
| NAME_CONTRACT_TYPE | pr_auc | Revolving loans (0.1641) | Cash loans (0.2657) | 0.1016 |

## Largest Test Gaps

| Column | Metric | Lowest group | Highest group | Gap |
| --- | --- | --- | --- | --- |
| NAME_EDUCATION_TYPE | recall | Higher education (0.2800) | Secondary / secondary special (0.4478) | 0.1678 |
| NAME_INCOME_TYPE | recall | Pensioner (0.3265) | Working (0.4502) | 0.1237 |
| NAME_INCOME_TYPE | approval_rate | Working (0.8281) | Pensioner (0.9463) | 0.1183 |
| NAME_INCOME_TYPE | specificity | Working (0.8583) | Pensioner (0.9625) | 0.1043 |
| NAME_EDUCATION_TYPE | pr_auc | Higher education (0.1595) | Secondary / secondary special (0.2462) | 0.0867 |

## Ethical AI Interpretation

The model should not be used as a fully automated credit denial engine. Its scores are best treated as decision-support signals paired with policy rules, manual review for borderline cases, and regular subgroup monitoring. Gaps in recall or approval rate should trigger review because they can change who receives credit access or who is escalated for additional documentation.

## Recommended Controls

- Recompute subgroup metrics after each retraining run.
- Track recall, specificity, approval rate, and approved default rate by subgroup.
- Investigate large subgroup gaps before deployment or threshold changes.
- Keep human review for high-impact credit decisions.
- Document approved model thresholds and business cost assumptions.
