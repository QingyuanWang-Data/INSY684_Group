# Business Assumptions and Open Questions

## Current Business Assumptions

The project currently assumes:

- The Home Credit Default Risk use case will continue from the Course 1 project.
- The main business problem is credit default risk ranking for loan applicants.
- The model output will be used as a decision-support score, not as a fully automated final
  lending decision.
- The business decision process can be described through low-, medium-, and high-risk bands.
- Course 1 provides the baseline model and business context.
- Course 2 should improve production readiness, tracking, explainability, fairness, monitoring,
  and deployment clarity.

## Financial Impact Assumptions

The current business impact scenario table uses planning assumptions from the previous project:

| Assumption | Value |
| --- | ---: |
| Annual applications | 100,000 |
| Approval rate | 60% |
| Approved loans | 60,000 |
| Average funded amount | 8,000 |
| Baseline default rate among approved applicants | 6% |
| Loss given default | 45% |
| Baseline expected loss | 12.96M |

The current scenarios estimate possible annual savings if a model-supported policy reduces
approved defaults by 4%, 6.5%, or 9%.

These are planning estimates only. They should not be presented as proven savings.

## Questions for the Technical Team

The business lead needs the following information from the technical team as the project
progresses. Some items now have initial answers from the technical branch, but they should be
confirmed again before final merge to `main`.

| Question | Current answer / next step |
| --- | --- |
| Which model will be treated as the Course 2 main model? | Initial sampled LightGBM/focal-loss run exists; confirm final model before merging to `main`. |
| What is the new validation and test ROC-AUC? | Current final validation ROC-AUC is `0.7666`; final held-out test ROC-AUC is `0.7647`. |
| Is PR-AUC being reported because the default class is rare? | Yes, current validation PR-AUC is `0.2588` and test PR-AUC is `0.2303`. |
| What threshold or risk band cutoffs are being used? | Current selected threshold is `0.549`; final risk-band cutoffs still need agreement. |
| What are the confusion matrix results at the selected threshold? | Validation matrix: true 0 predicted 0/1 = `8197/998`; true 1 predicted 0/1 = `465/340`. Test matrix: true 0 predicted 0/1 = `4109/489`; true 1 predicted 0/1 = `233/169`. |
| What approval rate does the selected policy imply? | Current validation approval rate is `86.62%`. |
| What is the estimated default rate among approved applicants? | Current approved default rate is `5.37%`. |
| Are MLflow or other tracking tools recording parameters, metrics, and artifacts? | MLflow hooks and sampled run evidence are documented; final run evidence should be captured for slides. |
| Which subgroup metrics are available for fairness review? | Gender, income type, contract type, education, and family status subgroup metrics are available. |
| Are there drift monitoring outputs? | Score and feature PSI monitoring outputs are available in the technical branch. |
| How will the model be served or demonstrated? | FastAPI, readiness endpoint, Docker runtime, and deployment notes are documented. |
| Which tests and CI checks are required before merging work? | Quality checks, pytest, and Docker build workflow exist; final branch should pass checks before merging to `main`. |

## Questions for the Business Lead

The business lead should continue refining:

1. How should the final presentation explain the business value?
2. Which business metrics should be emphasized alongside ROC-AUC?
3. How should the project explain false positives and false negatives?
4. Should the final recommendation use one threshold or three risk bands?
5. What fairness risks should be highlighted?
6. What limitations should be communicated clearly to stakeholders?
7. What would be required before real-world deployment?

## Final Main-Branch Review Checklist

Most of the business content is now drafted. The remaining work is mainly a final consistency
check after the team merges the technical and business branches into `main`.

Expected final checks:

- Confirm the final `main` branch uses the same model evidence: validation ROC-AUC `0.7666`, test ROC-AUC `0.7647`, and test PR-AUC `0.2303`.
- Add any updated final Course 2 model metrics if the team reruns training before submission.
- Confirm the selected threshold remains `0.549`. If the threshold changes, update approval rate, approved default rate, recall, specificity, expected cost, and business impact scenarios.
- Confirm the business recommendation still uses a three-band policy: low risk for auto-approval or standard terms, medium risk for manual review, and high risk for decline, stricter terms, lower credit limit, or additional documentation.
- Confirm confusion matrix interpretation is consistent with the final report. At the current threshold, validation recall is `42.24%`, so the model catches some risky applicants but still misses a meaningful share. This supports using the model as decision support instead of a fully automated denial tool.
- Confirm fairness findings are still consistent with the final subgroup report. The current evidence shows recall gaps across gender, income type, education, and family status. These gaps should be presented as governance risks that require monitoring, not as proof that the model is unusable.
- Confirm monitoring recommendations are aligned with the final drift outputs. Current score and feature PSI values are stable, but the team should still monitor score drift, feature drift, calibration, approval rate, approved default rate, and subgroup metrics after retraining or threshold changes.
- Confirm deployment and demo notes match the final technical branch. The current technical story is FastAPI serving, `/health` for process status, `/ready` for model artifact readiness, Docker runtime, and a future cloud-native deployment option.
- Confirm the final `main` branch does not contain conflicting metrics across documents. In particular, use `700` training-frame columns including `SK_ID_CURR` and `TARGET`, and `698` model features.
- Confirm `docs/SUBMISSION_INFO.md` includes the final team name, member GitHub IDs, and contribution descriptions before LMS submission.

## Business Presentation Notes

The final business narrative should emphasize:

1. The model can rank default risk meaningfully better than random, with final test ROC-AUC `0.7647`.
2. The model should support lending decisions, not replace human judgment.
3. Threshold selection is a business policy choice. Lower thresholds capture more risky applicants but reduce approval volume; higher thresholds protect growth but may allow more risky applicants through.
4. Business value depends on the approval policy. At the same approval volume, reducing approved default rate from `6.00%` to `5.37%` can lower expected loss in the planning scenario. But if approval volume increases sharply, total expected loss can rise even with a lower default rate.
5. Fairness and monitoring are part of the product, not optional add-ons. Subgroup gaps, drift, and calibration should be reviewed before any real lending deployment.
