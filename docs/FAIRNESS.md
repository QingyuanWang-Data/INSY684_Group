# Fairness and Ethical AI Report

## Scope

This report reviews subgroup behavior for the Home Credit default-risk model. The model estimates payment-difficulty probability, so fairness review focuses on risk-ranking quality, recall, specificity, approval rate, and default rate within approved applicants.

## Model Context

- Validation ROC-AUC: `0.7844`
- Test ROC-AUC: `0.7858`
- Decision threshold used: `0.549`

## Gender Subgroup Snapshot

| Group | Count | Positive rate | AUC | Recall | Approval rate | Approved default rate |
| --- | --- | --- | --- | --- | --- | --- |
| F | 40357 | 0.0699 | 0.7859 | 0.4114 | 0.8892 | 0.0463 |
| M | 21145 | 0.1013 | 0.7717 | 0.5250 | 0.8014 | 0.0601 |

## Largest Validation Gaps

| Column | Metric | Lowest group | Highest group | Gap |
| --- | --- | --- | --- | --- |
| NAME_INCOME_TYPE | recall | Pensioner (0.2457) | Working (0.5197) | 0.2740 |
| NAME_FAMILY_STATUS | recall | Widow (0.2961) | Single / not married (0.5333) | 0.2372 |
| NAME_EDUCATION_TYPE | recall | Higher education (0.3121) | Secondary / secondary special (0.4961) | 0.1840 |
| NAME_EDUCATION_TYPE | approval_rate | Lower secondary (0.7971) | Higher education (0.9349) | 0.1378 |
| NAME_INCOME_TYPE | pr_auc | Pensioner (0.1709) | Working (0.3047) | 0.1338 |
| NAME_EDUCATION_TYPE | specificity | Lower secondary (0.8173) | Higher education (0.9494) | 0.1321 |
| NAME_CONTRACT_TYPE | recall | Revolving loans (0.3366) | Cash loans (0.4685) | 0.1318 |
| NAME_FAMILY_STATUS | pr_auc | Widow (0.1901) | Single / not married (0.3149) | 0.1249 |

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
