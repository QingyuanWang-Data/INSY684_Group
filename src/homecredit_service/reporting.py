from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class GapSummary:
    column: str
    metric: str
    min_group: str
    min_value: float
    max_group: str
    max_value: float
    gap: float


FAIRNESS_METRICS = (
    "auc",
    "pr_auc",
    "recall",
    "specificity",
    "approval_rate",
    "default_rate_within_approved",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_group_gaps(
    frame: pd.DataFrame,
    metrics: tuple[str, ...] = FAIRNESS_METRICS,
) -> list[GapSummary]:
    summaries: list[GapSummary] = []
    for column_name, group_frame in frame.groupby("column", sort=True):
        for metric in metrics:
            values = pd.to_numeric(group_frame[metric], errors="coerce")
            valid = group_frame.loc[values.notna()].copy()
            if len(valid) < 2:
                continue
            valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
            min_idx = valid[metric].idxmin()
            max_idx = valid[metric].idxmax()
            min_value = float(valid.loc[min_idx, metric])
            max_value = float(valid.loc[max_idx, metric])
            summaries.append(
                GapSummary(
                    column=str(column_name),
                    metric=metric,
                    min_group=str(valid.loc[min_idx, "group"]),
                    min_value=min_value,
                    max_group=str(valid.loc[max_idx, "group"]),
                    max_value=max_value,
                    gap=max_value - min_value,
                )
            )
    return sorted(summaries, key=lambda item: item.gap, reverse=True)


def _format_float(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_fairness_report(artifact_dir: Path) -> str:
    validation_path = artifact_dir / "validation_subgroup_metrics.csv"
    test_path = artifact_dir / "test_subgroup_metrics.csv"
    report_path = artifact_dir / "training_report.json"

    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    training_report = load_json(report_path) if report_path.exists() else {}

    validation_gaps = summarize_group_gaps(validation)
    test_gaps = summarize_group_gaps(test) if not test.empty else []

    gender_rows = validation[validation["column"] == "CODE_GENDER"]
    gender_table_rows = [
        [
            str(row["group"]),
            str(int(row["count"])),
            _format_float(row["positive_rate"]),
            _format_float(row["auc"]),
            _format_float(row["recall"]),
            _format_float(row["approval_rate"]),
            _format_float(row["default_rate_within_approved"]),
        ]
        for _, row in gender_rows.iterrows()
    ]

    top_gap_rows = [
        [
            gap.column,
            gap.metric,
            f"{gap.min_group} ({_format_float(gap.min_value)})",
            f"{gap.max_group} ({_format_float(gap.max_value)})",
            _format_float(gap.gap),
        ]
        for gap in validation_gaps[:8]
    ]

    test_gap_rows = [
        [
            gap.column,
            gap.metric,
            f"{gap.min_group} ({_format_float(gap.min_value)})",
            f"{gap.max_group} ({_format_float(gap.max_value)})",
            _format_float(gap.gap),
        ]
        for gap in test_gaps[:5]
    ]

    threshold = training_report.get("threshold_used")
    validation_auc = training_report.get("validation_auc")
    test_auc = training_report.get("test_auc")

    sections = [
        "# Fairness and Ethical AI Report",
        "## Scope",
        (
            "This report reviews subgroup behavior for the Home Credit default-risk model. "
            "The model estimates payment-difficulty probability, so fairness review focuses on "
            "risk-ranking quality, recall, specificity, approval rate, and default rate within "
            "approved applicants."
        ),
        "## Model Context",
        "\n".join(
            [
                f"- Validation ROC-AUC: `{_format_float(validation_auc)}`",
                f"- Test ROC-AUC: `{_format_float(test_auc)}`",
                f"- Decision threshold used: `{_format_float(threshold, digits=3)}`",
            ]
        ),
        "## Gender Subgroup Snapshot",
        _markdown_table(
            [
                "Group",
                "Count",
                "Positive rate",
                "AUC",
                "Recall",
                "Approval rate",
                "Approved default rate",
            ],
            gender_table_rows,
        ),
        "## Largest Validation Gaps",
        _markdown_table(
            ["Column", "Metric", "Lowest group", "Highest group", "Gap"],
            top_gap_rows,
        ),
        "## Largest Test Gaps",
        _markdown_table(
            ["Column", "Metric", "Lowest group", "Highest group", "Gap"],
            test_gap_rows,
        ),
        "## Ethical AI Interpretation",
        (
            "The model should not be used as a fully automated credit denial engine. Its scores "
            "are best treated as decision-support signals paired with policy rules, manual review "
            "for borderline cases, and regular subgroup monitoring. Gaps in recall or approval "
            "rate should trigger review because they can change who receives credit access or who "
            "is escalated for additional documentation."
        ),
        "## Recommended Controls",
        "\n".join(
            [
                "- Recompute subgroup metrics after each retraining run.",
                (
                    "- Track recall, specificity, approval rate, and approved default rate "
                    "by subgroup."
                ),
                "- Investigate large subgroup gaps before deployment or threshold changes.",
                "- Keep human review for high-impact credit decisions.",
                "- Document approved model thresholds and business cost assumptions.",
            ]
        ),
    ]
    return "\n\n".join(sections) + "\n"


def build_monitoring_report(artifact_dir: Path) -> str:
    training_report = load_json(artifact_dir / "training_report.json")
    drift_summary = load_json(artifact_dir / "drift_summary.json")

    score_distribution = drift_summary.get("score_distribution", {})
    feature_distribution = drift_summary.get("feature_distribution", {})
    feature_rows = feature_distribution.get("rows", [])
    sorted_features = sorted(
        feature_rows,
        key=lambda row: max(
            float(row.get("train_vs_validation_psi") or 0.0),
            float(row.get("train_vs_test_psi") or 0.0),
        ),
        reverse=True,
    )

    feature_table_rows = [
        [
            str(row.get("feature")),
            _format_float(row.get("train_vs_validation_psi"), digits=6),
            str(row.get("train_vs_validation_severity")),
            _format_float(row.get("train_vs_test_psi"), digits=6),
            str(row.get("train_vs_test_severity")),
        ]
        for row in sorted_features
    ]

    calibration = training_report.get("calibration", {})
    validation_calibration = (
        calibration.get("validation", {}) if isinstance(calibration, dict) else {}
    )
    policy = training_report.get("policy_simulation", {})
    policy_splits = policy.get("splits", {}) if isinstance(policy, dict) else {}
    validation_policy = (
        policy_splits.get("validation", {}) if isinstance(policy_splits, dict) else {}
    )
    pr_auc = training_report.get("pr_auc", {})
    validation_pr_auc = pr_auc.get("validation") if isinstance(pr_auc, dict) else None
    score_validation_psi = _format_float(
        score_distribution.get("train_vs_validation_psi"),
        digits=6,
    )
    score_test_psi = _format_float(score_distribution.get("train_vs_test_psi"), digits=6)

    sections = [
        "# Model Monitoring Report",
        "## Current Model Snapshot",
        "\n".join(
            [
                f"- Train ROC-AUC: `{_format_float(training_report.get('train_auc'))}`",
                f"- Validation ROC-AUC: `{_format_float(training_report.get('validation_auc'))}`",
                f"- Test ROC-AUC: `{_format_float(training_report.get('test_auc'))}`",
                f"- Validation PR-AUC: `{_format_float(validation_pr_auc)}`",
                (
                    "- Threshold used: "
                    f"`{_format_float(training_report.get('threshold_used'), digits=3)}`"
                ),
                (
                    "- Validation Brier score: "
                    f"`{_format_float(validation_calibration.get('brier_score'))}`"
                ),
                (
                    "- Validation expected calibration error: "
                    f"`{_format_float(validation_calibration.get('expected_calibration_error'))}`"
                ),
            ]
        ),
        "## Prediction Drift",
        "\n".join(
            [
                (
                    "- Score PSI train vs validation: "
                    f"`{score_validation_psi}` "
                    f"({score_distribution.get('train_vs_validation_severity')})"
                ),
                (
                    "- Score PSI train vs test: "
                    f"`{score_test_psi}` "
                    f"({score_distribution.get('train_vs_test_severity')})"
                ),
            ]
        ),
        "## Feature Drift",
        _markdown_table(
            ["Feature", "Train vs validation PSI", "Severity", "Train vs test PSI", "Severity"],
            feature_table_rows,
        ),
        "## Policy Monitoring",
        "\n".join(
            [
                (
                    "- Validation approval rate: "
                    f"`{_format_float(validation_policy.get('approval_rate'))}`"
                ),
                (
                    "- Validation default rate within approved: "
                    f"`{_format_float(validation_policy.get('default_rate_within_approved'))}`"
                ),
                (
                    "- Expected validation cost per applicant: "
                    f"`{_format_float(validation_policy.get('expected_cost_per_applicant'))}`"
                ),
            ]
        ),
        "## Alert Rules",
        "\n".join(
            [
                "- PSI below 0.10: stable.",
                "- PSI from 0.10 to 0.25: investigate moderate distribution shift.",
                "- PSI above 0.25: block automatic promotion and perform model review.",
                "- Recheck subgroup fairness metrics whenever threshold or training data changes.",
                (
                    "- Recalibrate or retrain if calibration error or business cost worsens "
                    "materially."
                ),
            ]
        ),
    ]
    return "\n\n".join(sections) + "\n"


def build_model_card(artifact_dir: Path) -> str:
    training_report = load_json(artifact_dir / "training_report.json")
    class_distribution = training_report.get("class_distribution", {})
    dataset_size = training_report.get("dataset_size", {})
    hyperparameters = training_report.get("hyperparameters", {})
    positive_rate = _format_float(class_distribution.get("positive_rate"))
    negative_rate = _format_float(class_distribution.get("negative_rate"))
    threshold_used = _format_float(training_report.get("threshold_used"), digits=3)

    key_params = [
        ["Objective", str(hyperparameters.get("objective", "n/a"))],
        ["Learning rate", str(hyperparameters.get("learning_rate", "n/a"))],
        ["Num leaves", str(hyperparameters.get("num_leaves", "n/a"))],
        ["Max depth", str(hyperparameters.get("max_depth", "n/a"))],
        ["Imbalance strategy", str(training_report.get("imbalance_strategy", "n/a"))],
    ]

    sections = [
        "# Model Card",
        "## Intended Use",
        (
            "This model estimates payment-difficulty risk for Home Credit loan applicants. "
            "It is intended for credit-risk decision support, portfolio monitoring, and "
            "policy simulation, not as a fully automated denial system."
        ),
        "## Data",
        "\n".join(
            [
                f"- Rows: `{dataset_size.get('rows', 'n/a')}`",
                f"- Columns: `{dataset_size.get('columns', 'n/a')}`",
                f"- Positive class rate: `{positive_rate}`",
                f"- Negative class rate: `{negative_rate}`",
            ]
        ),
        "## Performance",
        "\n".join(
            [
                f"- Train ROC-AUC: `{_format_float(training_report.get('train_auc'))}`",
                f"- Validation ROC-AUC: `{_format_float(training_report.get('validation_auc'))}`",
                f"- Test ROC-AUC: `{_format_float(training_report.get('test_auc'))}`",
                f"- Threshold used: `{threshold_used}`",
            ]
        ),
        "## Key Parameters",
        _markdown_table(["Parameter", "Value"], key_params),
        "## Limitations",
        "\n".join(
            [
                "- Historical data may encode past lending and socioeconomic patterns.",
                "- Subgroup gaps require monitoring before threshold or policy changes.",
                "- Calibration should be reviewed before using scores as absolute probabilities.",
                "- Production performance must be monitored after deployment for drift.",
            ]
        ),
        "## Governance Controls",
        "\n".join(
            [
                "- Use `/ready` to confirm model artifact availability before routing traffic.",
                "- Log training runs with MLflow for reproducibility.",
                "- Recompute fairness and monitoring reports after retraining.",
                "- Keep final credit decisions subject to policy and human review.",
            ]
        ),
    ]
    return "\n\n".join(sections) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate INSY684 model governance reports.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--report",
        choices=["fairness", "monitoring", "model-card", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    if args.report in {"fairness", "all"}:
        fairness = build_fairness_report(args.artifact_dir)
        (args.docs_dir / "FAIRNESS.md").write_text(fairness, encoding="utf-8")

    if args.report in {"monitoring", "all"}:
        monitoring = build_monitoring_report(args.artifact_dir)
        (args.docs_dir / "MONITORING.md").write_text(monitoring, encoding="utf-8")

    if args.report in {"model-card", "all"}:
        model_card = build_model_card(args.artifact_dir)
        (args.docs_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")


if __name__ == "__main__":
    main()
