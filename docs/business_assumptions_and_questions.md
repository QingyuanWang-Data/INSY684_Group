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
progresses:

1. Which model will be treated as the Course 2 main model?
2. What is the new validation and test ROC-AUC?
3. Is PR-AUC being reported because the default class is rare?
4. What threshold or risk band cutoffs are being used?
5. What are the confusion matrix results at the selected threshold?
6. What approval rate does the selected policy imply?
7. What is the estimated default rate among approved applicants?
8. Are MLflow or other tracking tools recording parameters, metrics, and artifacts?
9. Are model artifacts versioned clearly?
10. Which subgroup metrics are available for fairness review?
11. Are there drift monitoring outputs for data, score, or performance changes?
12. How will the model be served or demonstrated in the final project?
13. Which tests and CI checks are required before merging work?

## Questions for the Business Lead

The business lead should continue refining:

1. How should the final presentation explain the business value?
2. Which business metrics should be emphasized alongside ROC-AUC?
3. How should the project explain false positives and false negatives?
4. Should the final recommendation use one threshold or three risk bands?
5. What fairness risks should be highlighted?
6. What limitations should be communicated clearly to stakeholders?
7. What would be required before real-world deployment?

## Items to Update Later

This document should be updated after the team completes more technical work.

Expected updates:

- Replace planning assumptions with model-based policy results if available.
- Add the selected Course 2 model metrics.
- Add the selected threshold or risk band cutoffs.
- Add business interpretation of confusion matrix results.
- Add fairness findings from subgroup analysis.
- Add monitoring recommendations based on drift outputs.
- Add final deployment and demo notes.


