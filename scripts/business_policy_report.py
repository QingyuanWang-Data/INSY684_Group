"""Generate a business-facing policy summary from model evidence files.

This script is intentionally lightweight: it reads the portable CSV evidence
committed by the technical team and writes a Markdown report for non-technical
stakeholders. It does not train or score the model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_BASELINE_AUC = 0.7858
DEFAULT_CURRENT_AUC = 0.7666
DEFAULT_TEST_AUC = 0.7647
DEFAULT_TUNED_AUC = 0.7679
DEFAULT_TEST_PR_AUC = 0.2303
DEFAULT_THRESHOLD = 0.549


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def find_threshold_row(
    rows: list[dict[str, str]],
    threshold: float,
) -> dict[str, str] | None:
    for row in rows:
        if abs(as_float(row, "selected_threshold") - threshold) < 1e-6:
            return row
    return rows[0] if rows else None


def largest_gap(
    rows: list[dict[str, str]],
    column: str,
    metric: str,
) -> tuple[str, str, float] | None:
    filtered = [row for row in rows if row.get("column") == column and row.get(metric)]
    if len(filtered) < 2:
        return None
    ordered = sorted(filtered, key=lambda row: as_float(row, metric))
    low = ordered[0]
    high = ordered[-1]
    gap = as_float(high, metric) - as_float(low, metric)
    return low.get("group", "n/a"), high.get("group", "n/a"), gap


def best_tuning_auc(rows: list[dict[str, str]]) -> float:
    values = [as_float(row, "value") for row in rows if row.get("value")]
    return max(values) if values else DEFAULT_TUNED_AUC


def pct(value: float) -> str:
    return f"{value:.2%}"


def num(value: float) -> str:
    return f"{value:.4f}"


def build_report(artifact_dir: Path) -> str:
    cost_rows = read_csv_rows(artifact_dir / "cost_sensitivity.csv")
    subgroup_rows = read_csv_rows(artifact_dir / "validation_subgroup_metrics.csv")
    tuning_rows = read_csv_rows(artifact_dir / "tuning_trials.csv")

    policy_row = find_threshold_row(cost_rows, DEFAULT_THRESHOLD)
    approval_rate = as_float(policy_row or {}, "approval_rate", 0.8662)
    approved_default_rate = as_float(policy_row or {}, "default_rate_within_approved", 0.0537)
    expected_cost = as_float(policy_row or {}, "expected_cost_per_applicant", 0.3323)
    recall = as_float(policy_row or {}, "recall", 0.4224)
    specificity = as_float(policy_row or {}, "specificity", 0.8915)
    tuned_auc = best_tuning_auc(tuning_rows)

    gender_gap = largest_gap(subgroup_rows, "CODE_GENDER", "recall")
    income_gap = largest_gap(subgroup_rows, "NAME_INCOME_TYPE", "recall")

    lines = [
        "# Business Policy Summary",
        "",
        "## Purpose",
        "",
        "This report translates the current model evidence into business-facing "
        "policy language. It is intended for the business lead, final "
        "presentation, and project documentation. It should be refreshed when "
        "model artifacts or threshold policy change.",
        "",
        "## Model Evidence Snapshot",
        "",
        "| Item | Current evidence |",
        "| --- | ---: |",
        f"| Course 1 reference test ROC-AUC | `{DEFAULT_BASELINE_AUC:.4f}` |",
        f"| Course 2 final validation ROC-AUC | `{DEFAULT_CURRENT_AUC:.4f}` |",
        f"| Course 2 final test ROC-AUC | `{DEFAULT_TEST_AUC:.4f}` |",
        f"| Course 2 Optuna best validation ROC-AUC | `{tuned_auc:.4f}` |",
        f"| Course 2 test PR-AUC | `{DEFAULT_TEST_PR_AUC:.4f}` |",
        f"| Selected threshold | `{DEFAULT_THRESHOLD:.3f}` |",
        f"| Approval rate at selected threshold | `{pct(approval_rate)}` |",
        f"| Approved default rate | `{pct(approved_default_rate)}` |",
        f"| Expected cost per applicant | `{num(expected_cost)}` |",
        f"| Recall for payment-difficulty applicants | `{pct(recall)}` |",
        f"| Specificity for normal-repayment applicants | `{pct(specificity)}` |",
        "",
        f"The final test ROC-AUC of `{DEFAULT_TEST_AUC:.4f}` confirms that the "
        "current Course 2 model ranks applicants meaningfully better than "
        "random on the held-out test split. The Course 2 result should still "
        "not be interpreted only as a score competition against the Course 1 "
        "baseline. The main Course 2 value is the production-oriented "
        "workflow: MLflow hooks, Optuna tuning, Docker packaging, CI checks, "
        "monitoring reports, fairness review, and deployment documentation.",
        "",
        "## Threshold and Business Policy Interpretation",
        "",
        f"At threshold `{DEFAULT_THRESHOLD:.3f}`, the current policy approves "
        f"about `{pct(approval_rate)}` of applicants in the validation sample, "
        "with an approved default rate of about "
        f"`{pct(approved_default_rate)}`. This supports a growth-oriented "
        "policy, but it still misses a meaningful share of risky applicants "
        f"because recall is about `{pct(recall)}`.",
        "",
        "The cost-sensitivity table shows the expected policy tradeoff: when "
        "false negatives become more expensive, the selected threshold moves "
        "lower. Lower thresholds catch more risky applicants but reduce "
        "approval volume and increase manual review or stricter-treatment "
        "cases.",
        "",
        "| Business objective | Likely threshold direction | Expected impact |",
        "| --- | --- | --- |",
        "| Protect approval volume and customer growth | Higher threshold | "
        "More approvals, but higher risk leakage |",
        "| Reduce missed risky borrowers | Lower threshold | More risk capture, "
        "but more applicants delayed or rejected |",
        "| Balance growth and risk | Three-band policy | Low risk auto-approve, "
        "medium risk review, high risk stricter treatment |",
        "",
        "## Fairness and Governance Interpretation",
        "",
    ]

    if gender_gap:
        low, high, gap = gender_gap
        lines.append(
            f"- Gender recall gap: `{low}` has lower recall than `{high}` "
            f"by about `{gap:.4f}`."
        )
    if income_gap:
        low, high, gap = income_gap
        lines.append(
            f"- Income-type recall gap: `{low}` has lower recall than `{high}` "
            f"by about `{gap:.4f}`."
        )
    if not gender_gap and not income_gap:
        lines.append(
            "- Subgroup metrics were not found locally. Use "
            "`artifacts/validation_subgroup_metrics.csv` when available."
        )

    lines.extend(
        [
            "- These gaps do not automatically prove unfairness, but they are "
            "strong reasons to avoid fully automated denial decisions.",
            "- Medium-risk applicants should remain eligible for manual review.",
            "- Subgroup metrics should be recomputed after retraining, threshold "
            "changes, or data updates.",
            "",
            "## Business Recommendation",
            "",
            "Use the model as a credit-risk decision-support system rather than "
            "a fully automated lending decision engine. The recommended "
            "business framing is:",
            "",
            "1. Low-risk band: consider auto-approval or standard terms.",
            "2. Medium-risk band: route to manual review.",
            "3. High-risk band: consider decline, lower credit limit, stricter "
            "pricing, or additional documentation.",
            "",
            "Before final submission, the business section should be checked "
            "against the final main branch results, final threshold policy, "
            "and any new fairness or monitoring outputs that the team "
            "generates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts", type=Path)
    parser.add_argument("--output", default=Path("reports/business_policy_summary.md"), type=Path)
    args = parser.parse_args()

    report = build_report(args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
