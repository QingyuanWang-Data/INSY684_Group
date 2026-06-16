from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

DEFAULT_ARTIFACT_DIR = Path("artifacts")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_metric(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def psi_status_color(severity: str | None) -> str:
    if severity == "stable":
        return "normal"
    if severity == "moderate":
        return "inverse"
    if severity == "high":
        return "off"
    return "normal"


def render_header(training_report: dict[str, Any]) -> None:
    st.title("Home Credit Model Monitoring")
    st.caption(
        "Offline baseline dashboard built from saved training artifacts. "
        "It demonstrates the monitoring workflow before live production data exists."
    )

    trained_at = training_report.get("trained_at_utc", "n/a")
    threshold = format_metric(training_report.get("threshold_used"), digits=3)
    st.info(
        f"Current snapshot: trained at `{trained_at}`, selected decision threshold `{threshold}`. "
        "When new production batches arrive, the same sections can compare those batches against "
        "this baseline."
    )


def render_model_health(training_report: dict[str, Any]) -> None:
    pr_auc = training_report.get("pr_auc", {})
    calibration = training_report.get("calibration", {})
    validation_calibration = (
        calibration.get("validation", {}) if isinstance(calibration, dict) else {}
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric("Train ROC-AUC", format_metric(training_report.get("train_auc"), 4))
    metric_columns[1].metric(
        "Validation ROC-AUC",
        format_metric(training_report.get("validation_auc"), 4),
    )
    metric_columns[2].metric(
        "Validation PR-AUC",
        format_metric(pr_auc.get("validation") if isinstance(pr_auc, dict) else None, 4),
    )
    metric_columns[3].metric(
        "Validation Brier",
        format_metric(validation_calibration.get("brier_score"), 4),
    )
    metric_columns[4].metric(
        "Feature Count",
        format_metric(training_report.get("feature_count"), 0),
    )

    class_distribution = training_report.get("class_distribution", {})
    split_counts = training_report.get("split_counts", {})
    size_frame = pd.DataFrame(
        [
            {
                "Dataset": "All sampled data",
                "Rows": training_report.get("dataset_size", {}).get("rows"),
                "Positive rate": class_distribution.get("positive_rate"),
            },
            {
                "Dataset": "Train",
                "Rows": split_counts.get("train_rows"),
                "Positive rate": (
                    split_counts.get("train_positive_count", 0)
                    / split_counts.get("train_rows", 1)
                ),
            },
            {
                "Dataset": "Validation",
                "Rows": split_counts.get("valid_rows"),
                "Positive rate": (
                    split_counts.get("valid_positive_count", 0)
                    / split_counts.get("valid_rows", 1)
                ),
            },
            {
                "Dataset": "Test",
                "Rows": split_counts.get("test_rows"),
                "Positive rate": (
                    split_counts.get("test_positive_count", 0)
                    / split_counts.get("test_rows", 1)
                ),
            },
        ]
    )
    st.subheader("Data Snapshot")
    st.dataframe(size_frame, hide_index=True, use_container_width=True)


def render_drift(drift_summary: dict[str, Any]) -> None:
    score_distribution = drift_summary.get("score_distribution", {})
    feature_distribution = drift_summary.get("feature_distribution", {})
    feature_rows = feature_distribution.get("rows", [])

    st.subheader("Prediction Drift")
    columns = st.columns(2)
    validation_severity = score_distribution.get("train_vs_validation_severity")
    test_severity = score_distribution.get("train_vs_test_severity")
    columns[0].metric(
        "Score PSI: Train vs Validation",
        format_metric(score_distribution.get("train_vs_validation_psi"), 6),
        validation_severity,
        delta_color=psi_status_color(validation_severity),
    )
    columns[1].metric(
        "Score PSI: Train vs Test",
        format_metric(score_distribution.get("train_vs_test_psi"), 6),
        test_severity,
        delta_color=psi_status_color(test_severity),
    )

    st.subheader("Feature Drift")
    drift_frame = pd.DataFrame(feature_rows)
    if drift_frame.empty:
        st.warning("No feature drift rows found in artifacts/drift_summary.json.")
        return

    drift_frame = drift_frame.sort_values("train_vs_validation_psi", ascending=False)
    chart_frame = drift_frame.set_index("feature")[["train_vs_validation_psi"]]
    st.bar_chart(chart_frame)
    st.dataframe(drift_frame, hide_index=True, use_container_width=True)

    st.caption(
        "PSI below 0.10 is stable, 0.10 to 0.25 needs investigation, "
        "and above 0.25 should trigger model review before automatic promotion."
    )


def render_policy(training_report: dict[str, Any], artifact_dir: Path) -> None:
    policy = training_report.get("policy_simulation", {})
    splits = policy.get("splits", {}) if isinstance(policy, dict) else {}
    validation_policy = splits.get("validation", {}) if isinstance(splits, dict) else {}

    st.subheader("Policy Monitoring")
    columns = st.columns(3)
    columns[0].metric(
        "Validation Approval Rate",
        format_metric(validation_policy.get("approval_rate"), 4),
    )
    columns[1].metric(
        "Default Rate Within Approved",
        format_metric(validation_policy.get("default_rate_within_approved"), 4),
    )
    columns[2].metric(
        "Expected Cost per Applicant",
        format_metric(validation_policy.get("expected_cost_per_applicant"), 4),
    )

    cost_path = artifact_dir / "cost_sensitivity.csv"
    cost_frame = load_csv(cost_path)
    if not cost_frame.empty:
        st.write("Cost sensitivity by threshold")
        st.dataframe(cost_frame, hide_index=True, use_container_width=True)


def render_confusion_matrices(training_report: dict[str, Any], artifact_dir: Path) -> None:
    matrices = training_report.get("confusion_matrices", {})
    if not isinstance(matrices, dict) or not matrices:
        st.warning("No confusion matrices found in artifacts/training_report.json.")
        return

    available_splits = [split for split in ("train", "validation", "test") if split in matrices]
    selected_split = st.selectbox("Split", available_splits, index=0)
    payload = matrices.get(selected_split, {})
    if not isinstance(payload, dict):
        st.warning(f"No confusion matrix payload found for `{selected_split}`.")
        return

    threshold = payload.get("threshold")
    matrix = payload.get("matrix", [])
    normalized_matrix = payload.get("normalized_matrix", [])
    labels = ["Predicted repay normal", "Predicted default risk"]
    index = ["Actual repay normal", "Actual default risk"]

    st.subheader(f"{selected_split.title()} Confusion Matrix")
    st.caption(f"Decision threshold: `{format_metric(threshold, 3)}`")

    if matrix:
        count_frame = pd.DataFrame(matrix, index=index, columns=labels)
        st.dataframe(count_frame, use_container_width=True)
        st.bar_chart(count_frame)

    if normalized_matrix:
        normalized_frame = pd.DataFrame(normalized_matrix, index=index, columns=labels)
        st.write("Row-normalized rates")
        st.dataframe(normalized_frame, use_container_width=True)

    plot_path = artifact_dir / "plots" / f"{selected_split}_confusion_matrix.png"
    if plot_path.exists():
        st.image(str(plot_path), caption=plot_path.name, use_container_width=True)

    if selected_split == "test" and not training_report.get("test_evaluated", False):
        st.info(
            "The test split exists, but final test evaluation was not enabled for this report. "
            "Run final evaluation to populate test metrics and train-vs-test drift."
        )


def render_fairness(artifact_dir: Path) -> None:
    validation = load_csv(artifact_dir / "validation_subgroup_metrics.csv")
    test = load_csv(artifact_dir / "test_subgroup_metrics.csv")

    st.subheader("Subgroup Monitoring")
    if validation.empty:
        st.warning("No validation subgroup metrics found.")
        return

    selected_column = st.selectbox(
        "Subgroup column",
        sorted(validation["column"].dropna().unique()),
    )
    filtered = validation[validation["column"] == selected_column].copy()
    display_columns = [
        "group",
        "count",
        "positive_rate",
        "auc",
        "pr_auc",
        "recall",
        "specificity",
        "approval_rate",
        "default_rate_within_approved",
    ]
    st.dataframe(filtered[display_columns], hide_index=True, use_container_width=True)

    chart_columns = [
        "auc",
        "recall",
        "approval_rate",
        "default_rate_within_approved",
    ]
    chart_frame = filtered.set_index("group")[chart_columns]
    st.bar_chart(chart_frame)

    if not test.empty:
        with st.expander("Test subgroup metrics"):
            st.dataframe(test, hide_index=True, use_container_width=True)


def render_evidence_plots(artifact_dir: Path) -> None:
    plots_dir = artifact_dir / "plots"
    plot_names = [
        "figure2_roc_curve_validation_test.png",
        "figure3_precision_recall_curve_validation_test.png",
        "validation_calibration_curve.png",
        "figure6_subgroup_performance_comparison_validation.png",
        "figure7_top15_feature_importance.png",
        "cost_sensitivity_threshold_curve.png",
    ]
    existing_plots = [plots_dir / name for name in plot_names if (plots_dir / name).exists()]

    if not existing_plots:
        st.warning("No monitoring plot images found under artifacts/plots.")
        return

    for plot_path in existing_plots:
        st.image(str(plot_path), caption=plot_path.name, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Home Credit Monitoring",
        page_icon="HC",
        layout="wide",
    )

    with st.sidebar:
        st.header("Settings")
        artifact_dir_text = st.text_input("Artifact directory", str(DEFAULT_ARTIFACT_DIR))
        artifact_dir = Path(artifact_dir_text)
        st.caption("Use the saved baseline artifacts generated by training and reports.")

    training_report = load_json(artifact_dir / "training_report.json")
    drift_summary = load_json(artifact_dir / "drift_summary.json")

    if not training_report:
        st.error(f"Could not load `{artifact_dir / 'training_report.json'}`.")
        return

    render_header(training_report)

    tabs = st.tabs(
        [
            "Model Health",
            "Drift",
            "Confusion Matrix",
            "Policy",
            "Fairness",
            "Evidence Plots",
        ]
    )
    with tabs[0]:
        render_model_health(training_report)
    with tabs[1]:
        if drift_summary:
            render_drift(drift_summary)
        else:
            st.warning(f"Could not load `{artifact_dir / 'drift_summary.json'}`.")
    with tabs[2]:
        render_confusion_matrices(training_report, artifact_dir)
    with tabs[3]:
        render_policy(training_report, artifact_dir)
    with tabs[4]:
        render_fairness(artifact_dir)
    with tabs[5]:
        render_evidence_plots(artifact_dir)


if __name__ == "__main__":
    main()
