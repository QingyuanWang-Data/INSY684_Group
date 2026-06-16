from __future__ import annotations

import argparse
import base64
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PLOTS = [
    "figure2_roc_curve_validation_test.png",
    "figure3_precision_recall_curve_validation_test.png",
    "validation_calibration_curve.png",
    "figure6_subgroup_performance_comparison_validation.png",
    "figure7_top15_feature_importance.png",
    "cost_sensitivity_threshold_curve.png",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def esc(value: Any) -> str:
    return html.escape(str(value))


def card(label: str, value: Any, detail: str = "") -> str:
    detail_html = f"<span>{esc(detail)}</span>" if detail else ""
    return f"""
    <div class="metric-card">
      <span>{esc(label)}</span>
      <strong>{esc(value)}</strong>
      {detail_html}
    </div>
    """


def table_html(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return '<p class="muted">No rows available.</p>'
    display = frame.head(max_rows) if max_rows else frame
    return display.to_html(index=False, classes="data-table", border=0, escape=True)


def image_html(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"""
    <figure>
      <img src="data:image/png;base64,{encoded}" alt="{esc(path.name)}">
      <figcaption>{esc(path.name)}</figcaption>
    </figure>
    """


def model_health_section(training_report: dict[str, Any]) -> str:
    pr_auc = training_report.get("pr_auc", {})
    calibration = training_report.get("calibration", {})
    validation_calibration = (
        calibration.get("validation", {}) if isinstance(calibration, dict) else {}
    )
    class_distribution = training_report.get("class_distribution", {})
    split_counts = training_report.get("split_counts", {})

    snapshot = pd.DataFrame(
        [
            {
                "Dataset": "All sampled data",
                "Rows": training_report.get("dataset_size", {}).get("rows"),
                "Positive rate": class_distribution.get("positive_rate"),
            },
            {
                "Dataset": "Train",
                "Rows": split_counts.get("train_rows"),
                "Positive rate": split_counts.get("train_positive_count", 0)
                / split_counts.get("train_rows", 1),
            },
            {
                "Dataset": "Validation",
                "Rows": split_counts.get("valid_rows"),
                "Positive rate": split_counts.get("valid_positive_count", 0)
                / split_counts.get("valid_rows", 1),
            },
            {
                "Dataset": "Test",
                "Rows": split_counts.get("test_rows"),
                "Positive rate": split_counts.get("test_positive_count", 0)
                / split_counts.get("test_rows", 1),
            },
        ]
    )

    metrics = "".join(
        [
            card("Train ROC-AUC", fmt(training_report.get("train_auc"), 4)),
            card("Validation ROC-AUC", fmt(training_report.get("validation_auc"), 4)),
            card(
                "Validation PR-AUC",
                fmt(pr_auc.get("validation") if isinstance(pr_auc, dict) else None, 4),
            ),
            card("Validation Brier", fmt(validation_calibration.get("brier_score"), 4)),
            card("Feature Count", fmt(training_report.get("feature_count"), 0)),
        ]
    )
    return f"""
    <section>
      <h2>Model Health</h2>
      <div class="metric-grid">{metrics}</div>
      <h3>Data Snapshot</h3>
      {table_html(snapshot)}
    </section>
    """


def drift_section(drift_summary: dict[str, Any]) -> str:
    score_distribution = drift_summary.get("score_distribution", {})
    feature_distribution = drift_summary.get("feature_distribution", {})
    drift_frame = pd.DataFrame(feature_distribution.get("rows", []))
    if not drift_frame.empty and "train_vs_validation_psi" in drift_frame:
        drift_frame = drift_frame.sort_values("train_vs_validation_psi", ascending=False)

    metrics = "".join(
        [
            card(
                "Score PSI: Train vs Validation",
                fmt(score_distribution.get("train_vs_validation_psi"), 6),
                score_distribution.get("train_vs_validation_severity", ""),
            ),
            card(
                "Score PSI: Train vs Test",
                fmt(score_distribution.get("train_vs_test_psi"), 6),
                score_distribution.get("train_vs_test_severity", ""),
            ),
        ]
    )
    return f"""
    <section>
      <h2>Drift</h2>
      <div class="metric-grid two">{metrics}</div>
      <h3>Feature Drift</h3>
      {table_html(drift_frame)}
      <p class="muted">PSI below 0.10 is stable, 0.10 to 0.25 needs investigation, and above 0.25 should trigger model review.</p>
    </section>
    """


def confusion_section(training_report: dict[str, Any]) -> str:
    matrices = training_report.get("confusion_matrices", {})
    if not isinstance(matrices, dict) or not matrices:
        return """
        <section>
          <h2>Confusion Matrices</h2>
          <p class="muted">No confusion matrices found in training_report.json.</p>
        </section>
        """

    parts = ["<section><h2>Confusion Matrices</h2>"]
    labels = ["Predicted repay normal", "Predicted default risk"]
    index = ["Actual repay normal", "Actual default risk"]
    for split in ("train", "validation", "test"):
        payload = matrices.get(split)
        if not isinstance(payload, dict):
            continue
        matrix = payload.get("matrix", [])
        normalized_matrix = payload.get("normalized_matrix", [])
        parts.append(f"<h3>{esc(split.title())}</h3>")
        parts.append(f"<p class=\"muted\">Decision threshold: {fmt(payload.get('threshold'), 3)}</p>")
        if matrix:
            parts.append(table_html(pd.DataFrame(matrix, index=index, columns=labels).reset_index()))
        if normalized_matrix:
            parts.append("<h4>Row-normalized rates</h4>")
            parts.append(
                table_html(pd.DataFrame(normalized_matrix, index=index, columns=labels).reset_index())
            )
    parts.append("</section>")
    return "\n".join(parts)


def policy_section(training_report: dict[str, Any], artifact_dir: Path) -> str:
    policy = training_report.get("policy_simulation", {})
    splits = policy.get("splits", {}) if isinstance(policy, dict) else {}
    validation_policy = splits.get("validation", {}) if isinstance(splits, dict) else {}
    metrics = "".join(
        [
            card("Validation Approval Rate", fmt(validation_policy.get("approval_rate"), 4)),
            card(
                "Default Rate Within Approved",
                fmt(validation_policy.get("default_rate_within_approved"), 4),
            ),
            card(
                "Expected Cost per Applicant",
                fmt(validation_policy.get("expected_cost_per_applicant"), 4),
            ),
        ]
    )
    return f"""
    <section>
      <h2>Policy Monitoring</h2>
      <div class="metric-grid">{metrics}</div>
      <h3>Cost Sensitivity by Threshold</h3>
      {table_html(load_csv(artifact_dir / "cost_sensitivity.csv"))}
    </section>
    """


def fairness_section(artifact_dir: Path) -> str:
    validation = load_csv(artifact_dir / "validation_subgroup_metrics.csv")
    test = load_csv(artifact_dir / "test_subgroup_metrics.csv")
    return f"""
    <section>
      <h2>Subgroup Monitoring</h2>
      <h3>Validation Subgroups</h3>
      {table_html(validation)}
      <h3>Test Subgroups</h3>
      {table_html(test)}
    </section>
    """


def plot_section(artifact_dir: Path) -> str:
    plots_dir = artifact_dir / "plots"
    images = "\n".join(image_html(plots_dir / name) for name in DEFAULT_PLOTS)
    if not images.strip():
        images = '<p class="muted">No monitoring plot images found under artifacts/plots.</p>'
    return f"""
    <section>
      <h2>Evidence Plots</h2>
      <div class="plot-grid">{images}</div>
    </section>
    """


def build_html(artifact_dir: Path) -> str:
    training_report = load_json(artifact_dir / "training_report.json")
    drift_summary = load_json(artifact_dir / "drift_summary.json")
    if not training_report:
        raise FileNotFoundError(f"Could not load {artifact_dir / 'training_report.json'}")

    trained_at = training_report.get("trained_at_utc", "n/a")
    threshold = fmt(training_report.get("threshold_used"), 3)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Home Credit Model Monitoring</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #607080;
      --line: #d8dee6;
      --accent: #0f766e;
      --accent-soft: #e0f2ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.5;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 32px max(24px, calc((100vw - 1160px) / 2));
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2, h3, h4 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 32px; }}
    h2 {{
      margin-top: 8px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
      font-size: 24px;
    }}
    h3 {{ margin-top: 20px; font-size: 18px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 20px;
      padding: 22px;
    }}
    .snapshot {{
      margin-top: 14px;
      color: var(--muted);
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 14px 0;
    }}
    .metric-grid.two {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfd;
    }}
    .metric-card span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
    }}
    .metric-card strong {{
      display: block;
      margin-top: 5px;
      font-size: 24px;
      color: var(--accent);
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 4px;
      font-size: 13px;
    }}
    .data-table th, .data-table td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      background: var(--accent-soft);
      color: #123b37;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{
      padding: 8px 10px;
      color: var(--muted);
      font-size: 12px;
      border-top: 1px solid var(--line);
    }}
    .muted {{ color: var(--muted); }}
    @media print {{
      body {{ background: #fff; }}
      section {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Home Credit Model Monitoring</h1>
    <p class="snapshot">Trained at {esc(trained_at)} | selected threshold {esc(threshold)} | static report generated {esc(generated_at)}</p>
  </header>
  <main>
    {model_health_section(training_report)}
    {drift_section(drift_summary)}
    {confusion_section(training_report)}
    {policy_section(training_report, artifact_dir)}
    {fairness_section(artifact_dir)}
    {plot_section(artifact_dir)}
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static HTML monitoring dashboard.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/monitoring_dashboard.html"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_text = build_html(args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
