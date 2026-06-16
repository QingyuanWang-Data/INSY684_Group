# Business Impact Simulation

## Purpose

This report estimates expected credit loss under simple planning scenarios. It should be treated as directional business analysis, not as proven financial savings.

## Assumptions

| Assumption | Value |
| --- | ---: |
| Annual applications | `100,000` |
| Average funded amount | `$8,000` |
| Loss given default | `45.00%` |
| Baseline approval rate | `60.00%` |
| Baseline approved default rate | `6.00%` |
| Current model threshold | `0.549` |
| Current validation approval rate | `86.62%` |
| Current approved default rate | `5.37%` |

## Scenario Results

| Scenario | Approval rate | Approved loans | Approved default rate | Expected defaults | Expected loss | Savings vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_policy | `60.00%` | `60,000` | `6.00%` | `3,600` | `$12,960,000` | `$0` |
| same_approval_volume_model_risk | `60.00%` | `60,000` | `5.37%` | `3,222` | `$11,599,200` | `$1,360,800` |
| current_validation_policy | `86.62%` | `86,620` | `5.37%` | `4,651` | `$16,745,339` | `$-3,785,339` |

## Interpretation

- Holding approval volume constant, reducing the approved default rate from `6.00%` to `5.37%` lowers expected loss in this planning scenario.
- The current validation threshold has a much higher approval rate (`86.62%`) than the Course 1 planning baseline (`60.00%`). Because more applicants are approved, total expected loss can increase even when the approved default rate is lower.
- This is why the model score should be paired with a business approval policy. A lower default rate is valuable, but approval volume, manual review capacity, and risk appetite determine the final business impact.
- These estimates should be updated if the team changes threshold, average loan amount, approval policy, or loss-given-default assumptions.

## Recommendation

Use this simulation to frame business tradeoffs in the final presentation. Avoid claiming realized savings. A careful wording is: under planning assumptions, the model-supported policy can reduce expected loss at the same approval volume, but final business value depends on the approval policy chosen by the lender.

