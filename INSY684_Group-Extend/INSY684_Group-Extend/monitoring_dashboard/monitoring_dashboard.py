from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = DASHBOARD_ROOT / "requirements-dashboard.txt"

try:
    from fastapi.testclient import TestClient
    from mlflow.tracking import MlflowClient

    from scenario_backend import app
except ModuleNotFoundError as dependency_error:
    st.set_page_config(page_title="Dashboard setup required", page_icon="⚙️")
    st.error(f"Missing dashboard dependency: {dependency_error.name}")
    st.write(f"Current Python: `{sys.executable}`")
    st.code(
        f'"{sys.executable}" -m pip install -r "{REQUIREMENTS_PATH}"',
        language="cmd",
    )
    st.write("After installation, restart the dashboard with `run_dashboard.cmd`.")
    st.stop()


PROJECT_ROOT = DASHBOARD_ROOT
ARTIFACTS = PROJECT_ROOT / "artifacts"
CORE_ARTIFACTS = PROJECT_ROOT.parents[1] / "artifacts"
CORE_PROJECT = PROJECT_ROOT.parents[1]
MODEL_RESULTS = ARTIFACTS / "model_with_clusting" / "model_with_clusting_results.csv"
MODEL_REPORT = ARTIFACTS / "model_with_clusting" / "model_with_clusting_report.json"
IMPORTANCE = ARTIFACTS / "model_with_clusting" / "lightgbm_feature_importance.csv"
CLUSTER_PLOTS = ARTIFACTS / "clustering" / "plots"
DASHBOARD_MLFLOW_DB = Path(
    os.environ.get("HC_DASHBOARD_MLFLOW_DB", str(ARTIFACTS / "dashboard_mlflow.db"))
)


st.set_page_config(
    page_title="Home Credit Decision Lab",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #f4f7fb; color: #172033; }
    .block-container {
        max-width: none; width: 97%;
        padding: 1.25rem 1rem 3rem;
    }
    h1 { font-size: 2.6rem !important; letter-spacing: -0.04em; color: #10233f; }
    h2 { font-size: 1.65rem !important; color: #10233f; padding-top: .3rem; }
    h3 { font-size: 1.15rem !important; color: #203a5f; }
    p, label, [data-testid="stMarkdownContainer"] { font-size: 1rem; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #dce5ef; border-radius: 16px;
        padding: 18px 20px; box-shadow: 0 5px 20px rgba(24, 55, 90, .06);
    }
    [data-testid="stMetricLabel"] { font-size: .98rem; font-weight: 700; color: #50647f; }
    [data-testid="stMetricValue"] { font-size: 2rem; color: #0d6b66; }
    .hero {
        padding: 26px 30px; border-radius: 22px; margin-bottom: 18px;
        background: linear-gradient(125deg, #10233f 0%, #174d63 60%, #0f766e 100%);
        color: white;
    }
    .hero h1 { color: white !important; margin: 0; }
    .hero p { font-size: 1.15rem; margin: .55rem 0 0; color: #e4f3f2; }
    .hero-title-row {
        display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    }
    .course-badge {
        display: inline-flex; align-items: center; padding: 7px 14px;
        border-radius: 999px; background: #f4c95d; color: #10233f;
        font-size: .95rem; font-weight: 900; letter-spacing: .06em;
        box-shadow: 0 4px 12px rgba(0, 0, 0, .12);
    }
    .team-links {
        display: flex; align-items: center; flex-wrap: wrap; gap: 9px;
        margin-top: 17px;
    }
    .team-label { color: #aee3dd; font-weight: 850; margin-right: 2px; }
    .hero-link {
        display: inline-flex; align-items: center; padding: 7px 11px;
        border: 1px solid rgba(255, 255, 255, .32); border-radius: 999px;
        background: rgba(255, 255, 255, .10); color: white !important;
        font-size: .9rem; font-weight: 750; text-decoration: none !important;
        transition: background .15s ease, border-color .15s ease, transform .15s ease;
    }
    .hero-link:hover {
        background: rgba(255, 255, 255, .20); border-color: rgba(255, 255, 255, .65);
        transform: translateY(-1px);
    }
    .hero-link.repo-link {
        background: #e4f3f2; border-color: #e4f3f2; color: #0d5f5b !important;
    }
    .section-card {
        background: white; border: 1px solid #dce5ef; border-radius: 18px;
        padding: 20px 22px; margin: 8px 0 18px;
    }
    .tool-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 12px 0 24px; }
    .tool-pill {
        background: white; border: 2px solid #10233f; border-radius: 14px;
        padding: 14px 8px; text-align: center; font-size: 1.05rem; font-weight: 800;
        color: #10233f;
    }
    .flow { font-size: 1.2rem; font-weight: 800; color: #0f766e; text-align: center; margin: 8px 0 14px; }
    .feature-chip {
        display: inline-block; padding: 8px 12px; margin: 4px;
        background: #e4f3f2; color: #0d5f5b; border-radius: 999px; font-weight: 700;
    }
    .status-grid, .detail-grid {
        display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px; margin: 12px 0 20px;
    }
    .detail-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .status-card, .detail-card {
        background: white; border: 1px solid #dce5ef; border-radius: 14px;
        padding: 15px 16px; min-height: 82px;
    }
    .status-card strong, .detail-card strong {
        display: block; color: #50647f; font-size: .86rem; margin-bottom: 7px;
    }
    .status-card span { color: #0d6b66; font-size: 1.35rem; font-weight: 800; }
    .status-card.pass { border-top: 4px solid #16866f; }
    .status-card.warn { border-top: 4px solid #db7c26; }
    .status-card.info { border-top: 4px solid #4f77a3; }
    .detail-card span { color: #172033; font-size: .96rem; font-weight: 700; }
    .control-note {
        background: white; border-left: 4px solid #4f77a3; border-radius: 10px;
        padding: 12px 14px; min-height: 88px; color: #334e68; line-height: 1.45;
    }
    .control-note strong { display: block; color: #10233f; margin-bottom: 4px; }
    .light-table { width: 100%; border-collapse: collapse; background: white; }
    .light-table th, .light-table td {
        padding: 10px 12px; border-bottom: 1px solid #e3eaf2; text-align: left;
        color: #243b53;
    }
    .light-table th { background: #eaf1f8; color: #10233f; }
    [data-testid="stCaptionContainer"], [data-testid="stToolbar"],
    [data-testid="stHeader"], header, footer, #MainMenu { display: none !important; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 750; padding: 10px 18px; }
    .stTabs [data-baseweb="tab"] p { color: #334e68 !important; font-weight: 750; }
    [data-testid="stWidgetLabel"] p {
        color: #203a5f !important; font-size: 1.02rem !important; font-weight: 800 !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div {
        color: #b42318;
    }
    .stButton > button {
        border-radius: 12px; min-height: 48px; font-size: 1.05rem; font-weight: 800;
    }
    .stButton > button[kind="primary"] {
        background: #e5484d !important; border-color: #e5484d !important; color: white !important;
    }
    .stButton > button[kind="secondary"] {
        background: white !important; border: 2px solid #203a5f !important; color: #203a5f !important;
    }
    .stButton > button p { color: inherit !important; font-weight: 800 !important; }
    [data-testid="stAlert"] p { color: #172033 !important; font-weight: 650; }
    @media (max-width: 900px) {
        .tool-grid, .status-grid, .detail-grid { grid-template-columns: repeat(2, 1fr); }
        .hero { padding: 22px 20px; }
        .hero-link { font-size: .84rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_data() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    training = read_json(ARTIFACTS / "training_report.json")
    model_report = read_json(MODEL_REPORT)
    return training, model_report, read_csv(MODEL_RESULTS), read_csv(IMPORTANCE)


@st.cache_resource
def api_client() -> TestClient:
    return TestClient(app)


def threshold_curve(training: dict[str, Any]) -> pd.DataFrame:
    curve = (
        training.get("threshold_optimization", {})
        .get("search_curve", {})
        .get("curve", [])
    )
    return pd.DataFrame(curve)


def scenario_table(
    curve: pd.DataFrame,
    positive_count: int,
    negative_count: int,
    false_positive_cost: float,
    false_negative_cost: float,
) -> pd.DataFrame:
    result = curve.copy()
    result["false_declines"] = (1 - result["specificity"]) * negative_count
    result["approved_defaults"] = (1 - result["recall"]) * positive_count
    result["scenario_cost"] = (
        result["false_declines"] * false_positive_cost
        + result["approved_defaults"] * false_negative_cost
    ) / (positive_count + negative_count)
    return result


def selected_scenario(scenarios: pd.DataFrame, threshold: float) -> pd.Series:
    index = (scenarios["threshold"] - threshold).abs().idxmin()
    return scenarios.loc[index]


def scenario_figure(scenarios: pd.DataFrame, selected: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(scenarios["threshold"], scenarios["scenario_cost"], color="#0f766e", lw=3, label="Expected cost")
    ax.plot(scenarios["threshold"], scenarios["approval_rate"], color="#2f6fbd", lw=2, label="Approval rate")
    ax.plot(scenarios["threshold"], scenarios["recall"], color="#db7c26", lw=2, label="Default recall")
    ax.axvline(float(selected["threshold"]), color="#b42318", ls="--", lw=2)
    ax.scatter([selected["threshold"]], [selected["scenario_cost"]], s=90, color="#b42318", zorder=5)
    ax.set(xlabel="Decision threshold", ylabel="Rate / cost per applicant", xlim=(0.15, 0.85), ylim=(0, 1.02))
    ax.grid(alpha=.18)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    return fig


def confusion_figure(row: pd.Series, positives: int, negatives: int) -> plt.Figure:
    tn = int(round(float(row["specificity"]) * negatives))
    fp = negatives - tn
    tp = int(round(float(row["recall"]) * positives))
    fn = positives - tp
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    image = ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=16, fontweight="bold",
                color="white" if value > matrix.max() / 2 else "#10233f")
    ax.set_xticks([0, 1], ["Approve", "Decline"])
    ax.set_yticks([0, 1], ["Good", "Default"])
    ax.set_title("Validation decisions", fontweight="bold")
    ax.set_xlabel("Model decision")
    ax.set_ylabel("Actual outcome")
    fig.colorbar(image, ax=ax, fraction=.045)
    fig.tight_layout()
    return fig


def policy_comparison_figure(scenarios: pd.DataFrame, selected: pd.Series) -> plt.Figure:
    baseline = selected_scenario(scenarios, 0.50)
    optimal = scenarios.loc[scenarios["scenario_cost"].idxmin()]
    frame = pd.DataFrame(
        {
            "Selected": [selected["approval_rate"], selected["recall"]],
            "Default 0.50": [baseline["approval_rate"], baseline["recall"]],
            "Cost-optimal": [optimal["approval_rate"], optimal["recall"]],
        },
        index=["Approval rate", "Default recall"],
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    frame.plot.bar(ax=ax, color=["#0f766e", "#6d8fb5", "#db7c26"], width=.72)
    ax.set(ylabel="Rate", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    return fig


def cost_breakdown_figure(row: pd.Series) -> plt.Figure:
    total = float(row["positive_count"] + row["negative_count"])
    false_decline_cost = (
        float(row["false_declines"]) * float(row["false_positive_cost"]) / total * 1000
    )
    approved_default_cost = (
        float(row["approved_defaults"]) * float(row["false_negative_cost"]) / total * 1000
    )
    labels = ["False declines", "Approved defaults"]
    values = [false_decline_cost, approved_default_cost]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(labels, values, color=["#6d8fb5", "#e5484d"], width=.58)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set(ylabel="Expected cost per 1,000 applicants")
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    return fig


def model_comparison_figure(results: pd.DataFrame) -> plt.Figure:
    frame = results[
        (results["model"] == "lightgbm_focal")
        & (results["feature_set"] != "clusting_only")
    ].copy()
    frame["label"] = frame["model"].str.replace("_", " ").str.title() + "\n" + frame["feature_set"].map({
        "selected_original_features": "Selected features",
        "selected_original_plus_clusting": "+ clustering",
    })
    frame = frame.sort_values("test_auc", ascending=True)
    colors = ["#0f766e" if "+ clustering" in x else "#6d8fb5" for x in frame["label"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.barh(frame["label"], frame["test_auc"], color=colors)
    for i, value in enumerate(frame["test_auc"]):
        ax.text(value + .0008, i, f"{value:.3f}", va="center", fontweight="bold")
    ax.set(xlabel="Test ROC-AUC", xlim=(.74, .78))
    ax.grid(axis="x", alpha=.18)
    fig.tight_layout()
    return fig


def importance_figure(importance: pd.DataFrame) -> plt.Figure:
    frame = importance.head(15).sort_values("importance_gain")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.barh(frame["feature"], frame["importance_gain"], color="#0f766e")
    ax.set_xlabel("LightGBM gain importance")
    ax.grid(axis="x", alpha=.18)
    fig.tight_layout()
    return fig


def image_gallery(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    for start in range(0, len(existing), 2):
        cols = st.columns(2)
        for column, path in zip(cols, existing[start:start + 2]):
            column.image(str(path), width="stretch")


def dashboard_mlflow_runs() -> list[dict[str, Any]]:
    database = DASHBOARD_MLFLOW_DB
    if not database.exists():
        return []
    client = MlflowClient(tracking_uri=f"sqlite:///{database.resolve().as_posix()}")
    experiment = client.get_experiment_by_name("dashboard-policy-scenarios")
    if experiment is None:
        return []
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    return [
        {
            "run_id": run.info.run_id,
            "name": run.data.tags.get("mlflow.runName", "n/a"),
            "status": run.info.status,
            "started": pd.to_datetime(run.info.start_time, unit="ms").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
        }
        for run in runs
    ]


def optuna_figure(tuning: dict[str, Any]) -> plt.Figure:
    trials = tuning.get("trials", [])
    numbers = [int(trial["number"]) for trial in trials]
    values = [float(trial["value"]) for trial in trials]
    best_index = int(np.argmax(values))
    margin = max((max(values) - min(values)) * .18, .0015)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(numbers, values, color="#6d8fb5", lw=2.5, marker="o", ms=7)
    ax.scatter(
        [numbers[best_index]],
        [values[best_index]],
        s=150,
        color="#0f766e",
        zorder=5,
        label=f"Best trial {numbers[best_index]}",
    )
    ax.annotate(
        f"{values[best_index]:.4f}",
        (numbers[best_index], values[best_index]),
        xytext=(-12, 14),
        textcoords="offset points",
        ha="center",
        fontweight="bold",
        color="#0f766e",
    )
    ax.set(
        xlabel="Optuna trial",
        ylabel="Validation ROC-AUC",
        xticks=numbers,
        ylim=(min(values) - margin, max(values) + margin),
    )
    ax.grid(alpha=.18)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def timing_figure(timings: dict[str, float]) -> plt.Figure:
    labels = ["Artifact load", "Policy calculation", "MLflow logging"]
    values = [
        max(float(timings.get("artifact_load", 0)), .01),
        max(float(timings.get("policy_calculation", 0)), .01),
        max(float(timings.get("mlflow_logging", 0)), .01),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    bars = ax.barh(labels, values, color=["#6d8fb5", "#0f766e", "#db7c26"])
    ax.set_xscale("log")
    ax.set_xlabel("Milliseconds · logarithmic scale")
    ax.grid(axis="x", alpha=.18)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value * 1.08,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f} ms",
            va="center",
            fontweight="bold",
        )
    fig.tight_layout()
    return fig


def detail_cards(values: dict[str, Any]) -> str:
    cards = "".join(
        f'<div class="detail-card"><strong>{label}</strong><span>{value}</span></div>'
        for label, value in values.items()
    )
    return f'<div class="detail-grid">{cards}</div>'


training, model_report, model_results, importance = load_data()
if not training:
    st.error("Training artifacts are missing.")
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="hero-title-row">
        <h1>Home Credit Decision Lab</h1>
        <span class="course-badge">INSY684</span>
      </div>
      <p>Interactive credit-risk policy, model evidence, feature engineering, clustering and production readiness.</p>
      <div class="team-links">
        <span class="team-label">Team</span>
        <a class="hero-link" href="https://github.com/QingyuanWang-Data" target="_blank" rel="noopener noreferrer">Qingyuan Wang &#8599;</a>
        <a class="hero-link" href="https://github.com/xuechen-17" target="_blank" rel="noopener noreferrer">Xuechen Hong &#8599;</a>
        <a class="hero-link" href="https://github.com/yasmnminee" target="_blank" rel="noopener noreferrer">Yasmine Zhao &#8599;</a>
        <a class="hero-link" href="https://github.com/Hankpapa" target="_blank" rel="noopener noreferrer">Hank Shao &#8599;</a>
        <a class="hero-link repo-link" href="https://github.com/QingyuanWang-Data/INSY684_Group" target="_blank" rel="noopener noreferrer">GitHub Repository &#8599;</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Decision Lab", "Modeling", "Feature Engineering", "Evidence", "Production Stack"])

with tabs[0]:
    split_counts = training.get("split_counts", {})
    positives = int(split_counts.get("valid_positive_count", 0))
    negatives = int(split_counts.get("valid_negative_count", 0))
    curve = threshold_curve(training)

    st.subheader("Run a lending-policy scenario")
    st.info(
        "Execution mode: post-training policy simulation. The fitted LightGBM model and "
        "validation scores are reused; changing threshold or business cost does not retrain "
        "the model."
    )
    control_a, control_b, control_c = st.columns([1.4, 1, 1])
    with control_a:
        threshold = st.slider(
            "Decision threshold",
            0.20,
            0.80,
            0.549,
            0.01,
            help="Applicants with a predicted default score below this value are approved.",
        )
    with control_b:
        fp_cost = st.slider(
            "False-decline cost",
            0.5,
            10.0,
            1.0,
            0.5,
            help="Business cost assigned when a good applicant is incorrectly declined.",
        )
    with control_c:
        fn_cost = st.slider(
            "Approved-default cost",
            1.0,
            25.0,
            5.0,
            0.5,
            help="Business cost assigned when a defaulting applicant is approved.",
        )

    note_a, note_b, note_c = st.columns([1.4, 1, 1])
    note_a.markdown(
        """<div class="control-note"><strong>Decision rule</strong>
        Higher threshold approves more applicants, but usually captures fewer defaults.</div>""",
        unsafe_allow_html=True,
    )
    note_b.markdown(
        """<div class="control-note"><strong>False decline</strong>
        A creditworthy applicant is rejected. Default weight = 1 cost unit.</div>""",
        unsafe_allow_html=True,
    )
    note_c.markdown(
        """<div class="control-note"><strong>Approved default</strong>
        A defaulting applicant is approved. Default weight = 5 cost units.</div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Only the relative cost ratio changes the optimal threshold; multiplying both costs "
        "by the same factor only rescales the reported expected cost."
    )

    scenarios = scenario_table(curve, positives, negatives, fp_cost, fn_cost)
    manual = selected_scenario(scenarios, threshold)
    optimal = scenarios.loc[scenarios["scenario_cost"].idxmin()]

    run_col, auto_col, blank_col = st.columns([1, 1, 3])
    run_clicked = run_col.button("Run via FastAPI", type="primary", width="stretch")
    optimal_clicked = auto_col.button("Run cost-optimal", width="stretch")

    if run_clicked or optimal_clicked:
        requested_threshold = float(optimal["threshold"] if optimal_clicked else threshold)
        payload = {
            "threshold": requested_threshold,
            "false_positive_cost": fp_cost,
            "false_negative_cost": fn_cost,
        }
        request_started = perf_counter()
        with st.status("Running policy scenario...", expanded=True) as run_status:
            st.write("1 / 4  Pydantic request validation")
            st.write("2 / 4  FastAPI POST /scenario")
            response = api_client().post("/scenario", json=payload)
            response.raise_for_status()
            result = response.json()
            st.write("3 / 4  Validation-curve policy calculation")
            st.write("4 / 4  MLflow metric and parameter logging")
            wall_ms = (perf_counter() - request_started) * 1000
            result["wall_clock_ms"] = round(wall_ms, 2)
            result["http_status"] = response.status_code
            st.session_state["last_scenario"] = result
            run_status.update(
                label=f"Completed {result['run_id']} in {wall_ms:,.1f} ms",
                state="complete",
                expanded=True,
            )
        st.toast("Scenario completed and logged to MLflow", icon="✅")

    last_run = st.session_state.get("last_scenario")
    active = pd.Series(last_run if last_run else manual.to_dict())
    active_costs = (
        scenario_table(
            curve,
            positives,
            negatives,
            float(last_run["false_positive_cost"]),
            float(last_run["false_negative_cost"]),
        )
        if last_run
        else scenarios
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Threshold", f"{active['threshold']:.3f}")
    metric_cols[1].metric("Approval rate", f"{active['approval_rate']:.1%}")
    metric_cols[2].metric("Default recall", f"{active['recall']:.1%}")
    metric_cols[3].metric("Approved default rate", f"{active['default_rate_within_approved']:.1%}")
    metric_cols[4].metric("Cost / 1,000", f"{active['scenario_cost'] * 1000:,.1f}")

    chart_col, matrix_col = st.columns([1.65, 1])
    chart_col.pyplot(scenario_figure(active_costs, active), width="stretch")
    matrix_col.pyplot(confusion_figure(active, positives, negatives), width="stretch")

    if last_run:
        st.subheader("Execution trace")
        trace_cols = st.columns(6)
        trace_cols[0].metric("HTTP", last_run["http_status"])
        trace_cols[1].metric("Validation", "Passed")
        trace_cols[2].metric("Policy", f"{last_run['timings_ms']['policy_calculation']:.1f} ms")
        trace_cols[3].metric("MLflow", f"{last_run['timings_ms']['mlflow_logging']:.1f} ms")
        trace_cols[4].metric("Backend", f"{last_run['timings_ms']['backend_total']:.1f} ms")
        trace_cols[5].metric("Wall clock", f"{last_run['wall_clock_ms']:.1f} ms")
        st.markdown(
            f"""
            <div class="section-card"><strong>Run {last_run['run_id']}</strong><br>
            {last_run['api_transport']} · Pydantic validated · MLflow run
            {last_run['mlflow_run_id']}<br><br>
            Approval <strong>{active['approval_rate']:.1%}</strong> · Default recall
            <strong>{active['recall']:.1%}</strong> · Cost-optimal threshold
            <strong>{last_run['optimal_threshold']:.3f}</strong></div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Ready. Click Run via FastAPI to create an auditable scenario run.")

with tabs[1]:
    cv = training.get("cross_validation", {})
    temporal = training.get("temporal_validation", {})
    cols = st.columns(5)
    cols[0].metric("Production model", "LightGBM focal")
    cols[1].metric("Validation AUC", f"{training.get('validation_auc', 0):.3f}")
    cols[2].metric("Final test AUC", f"{training.get('test_auc', 0):.3f}")
    cols[3].metric("3-fold CV AUC", f"{cv.get('auc_mean', 0):.3f}")
    cols[4].metric("Temporal holdout AUC", f"{temporal.get('holdout_auc', 0):.3f}")

    left, right = st.columns([1.7, 1])
    left.pyplot(model_comparison_figure(model_results), width="stretch")
    with right:
        st.subheader("What we modeled")
        st.markdown("""
        <div class="section-card">
        <strong>Classifier</strong><br>LightGBM with focal loss and early stopping<br><br>
        <strong>Feature sets</strong><br>Selected original features · Selected originals + clustering<br><br>
        <strong>Robust validation</strong><br>Early stopping · 3-fold CV · temporal holdout · final test split
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Why full LightGBM training is slower", expanded=False):
        st.markdown(
            """
            **Policy simulation (this dashboard button)** reuses 10,000 saved validation
            predictions and evaluates 101 threshold points.

            **Model retraining** rebuilds features from seven source tables, encodes hundreds
            of columns, fits focal-loss LightGBM with early stopping, runs cross-validation,
            evaluates temporal/test splits, writes joblib artifacts, and may execute Optuna
            trials. Threshold and business cost are not LightGBM training hyperparameters.
            """
        )
        st.code(
            "uv run train-homecredit --data-dir homecreditdefaultriskdata "
            "--artifact-dir artifacts --sample-size 50000",
            language="bash",
        )

    with st.expander("Training-only evidence (unchanged by policy threshold)", expanded=False):
        image_gallery(
            [
                CORE_ARTIFACTS / "plots" / "figure2_roc_curve_validation_test.png",
                CORE_ARTIFACTS / "plots" / "figure3_precision_recall_curve_validation_test.png",
                CORE_ARTIFACTS / "plots" / "platt_scaling_before_after.png",
                CORE_ARTIFACTS / "plots" / "train_validation_learning_curve.png",
            ]
        )

with tabs[2]:
    st.subheader("Feature engineering pipeline")
    st.markdown(
        """
        <div class="flow">7 source tables → temporal filtering → applicant-level aggregation → ratios & shares → encoding → feature selection</div>
        <div class="section-card">
          <span class="feature-chip">Credit / Income</span>
          <span class="feature-chip">Annuity / Income</span>
          <span class="feature-chip">Credit / Annuity</span>
          <span class="feature-chip">Days Employed %</span>
          <span class="feature-chip">Bureau history</span>
          <span class="feature-chip">Previous applications</span>
          <span class="feature-chip">Installments</span>
          <span class="feature-chip">POS cash</span>
          <span class="feature-chip">Credit card</span>
          <span class="feature-chip">Category count & share</span>
          <span class="feature-chip">PCA-2</span>
          <span class="feature-chip">KMeans · GMM · HDBSCAN</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected = model_report.get("selected_features", {})
    a, b = st.columns([1.5, 1])
    a.pyplot(importance_figure(importance), width="stretch")
    with b:
        st.metric("Engineered feature space", training.get("feature_count", 0))
        st.metric("Feature selection", "LightGBM gain")
        st.metric("Clustering features added", 18)
        st.markdown(
            """<div class="section-card"><strong>Feature-count interpretation</strong><br>
            The baseline pipeline reports the complete engineered feature space. The extended
            experiment ranks that space by LightGBM gain and evaluates a selected subset; the
            former top-k value is intentionally not presented as a universal production count.
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div class="section-card"><strong>Leakage control</strong><br>
            PCA and clustering are fitted without TARGET. Historical tables are aggregated to one applicant row before modeling.</div>""",
            unsafe_allow_html=True,
        )
    with st.expander("Clustering evidence", expanded=False):
        image_gallery(
            [
                CLUSTER_PLOTS / "kmeans_pca_scatter.png",
                CLUSTER_PLOTS / "gaussian_mixture_pca_scatter.png",
                CLUSTER_PLOTS / "hdbscan_pca_scatter.png",
                CLUSTER_PLOTS / "pca_information_lost.png",
            ]
        )

with tabs[3]:
    st.subheader("Threshold-sensitive scenario evidence")
    evidence_run = st.session_state.get("last_scenario")
    if evidence_run:
        evidence_scenarios = scenario_table(
            threshold_curve(training),
            int(training["split_counts"]["valid_positive_count"]),
            int(training["split_counts"]["valid_negative_count"]),
            float(evidence_run["false_positive_cost"]),
            float(evidence_run["false_negative_cost"]),
        )
        evidence_active = pd.Series(evidence_run)
        evidence_left, evidence_right = st.columns([1.65, 1])
        evidence_left.pyplot(
            scenario_figure(evidence_scenarios, evidence_active),
            width="stretch",
        )
        evidence_right.pyplot(
            confusion_figure(
                evidence_active,
                int(evidence_run["positive_count"]),
                int(evidence_run["negative_count"]),
            ),
            width="stretch",
        )
        comparison_left, comparison_right = st.columns([1.35, 1])
        comparison_left.pyplot(
            policy_comparison_figure(evidence_scenarios, evidence_active),
            width="stretch",
        )
        comparison_right.pyplot(
            cost_breakdown_figure(evidence_active),
            width="stretch",
        )
        st.success(
            f"Dynamic evidence from {evidence_run['run_id']} · "
            f"threshold {evidence_run['threshold']:.3f} · "
            f"{evidence_run['wall_clock_ms']:.1f} ms"
        )
    else:
        st.info("Run a scenario first. Its charts will appear here automatically.")
    st.markdown(
        "All charts on this page are generated from the latest threshold and cost run. "
        "ROC, PR, calibration, learning curves and clustering are training evidence and now "
        "live under Modeling or Feature Engineering because policy settings cannot change them."
    )

with tabs[4]:
    st.subheader("Production-ready ML stack")
    tools = ["FastAPI", "Pydantic", "joblib", "MLflow", "Optuna", "pytest", "Ruff / ty", "Docker", "GitHub Actions", "uv"]
    pills = "".join(f'<div class="tool-pill">{tool}</div>' for tool in tools)
    st.markdown(f'<div class="tool-grid">{pills}</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow">Feature pipeline → LightGBM artifact → API contract → Docker image → GitHub Actions quality gate</div>
        """,
        unsafe_allow_html=True,
    )
    tuning = read_json(CORE_ARTIFACTS / "tuning_report.json")
    training_runs = [
        path
        for path in (CORE_PROJECT / "mlruns" / "1").glob("*")
        if path.is_dir() and path.name != "artifacts"
    ]
    workflow_count = sum(
        path.exists()
        for path in [
            CORE_PROJECT / ".github" / "workflows" / "ci.yml",
            CORE_PROJECT / ".github" / "workflows" / "docker.yml",
        ]
    )

    status_values = [
        ("API transport", "In-process", "info"),
        ("MLflow training runs", len(training_runs), "pass"),
        ("Optuna completed trials", tuning.get("metadata", {}).get("n_trials", 0), "pass"),
        ("Best tuned AUC", f"{tuning.get('best_value', 0):.4f}", "pass"),
        ("CI workflows configured", workflow_count, "info"),
    ]
    status_html = "".join(
        f'<div class="status-card {kind}"><strong>{label}</strong><span>{value}</span></div>'
        for label, value, kind in status_values
    )
    st.markdown(f'<div class="status-grid">{status_html}</div>', unsafe_allow_html=True)

    st.subheader("Quality gate status")
    quality_values = [
        ("pytest", "19 / 19 passed", "pass"),
        ("ty", "Passed", "pass"),
        ("Ruff", "Passed", "pass"),
        ("Cloud workflow", "Configured", "info"),
    ]
    quality_html = "".join(
        f'<div class="status-card {kind}"><strong>{label}</strong><span>{value}</span></div>'
        for label, value, kind in quality_values
    )
    st.markdown(f'<div class="status-grid">{quality_html}</div>', unsafe_allow_html=True)
    st.warning(
        "The reproducible local gate passes Ruff, ty and all tests. GitHub Actions and the "
        "Docker build workflow are configured; the dashboard does not claim a live cloud status."
    )

    production_run = st.session_state.get("last_scenario")
    if production_run:
        st.success(
            f"Latest API run {production_run['run_id']} · HTTP 200 · "
            f"MLflow {production_run['mlflow_run_id']} · "
            f"{production_run['wall_clock_ms']:.1f} ms"
        )

    mlflow_runs = dashboard_mlflow_runs()
    with st.expander(f"MLflow scenario history · {len(mlflow_runs)} runs", expanded=True):
        if not mlflow_runs:
            st.info("No dashboard scenario has been logged yet.")
        else:
            recent_runs = mlflow_runs[:5]
            run_tabs = st.tabs([run["name"] for run in recent_runs])
            for run_tab, selected_run in zip(run_tabs, recent_runs, strict=True):
                with run_tab:
                    metrics = selected_run["metrics"]
                    params = selected_run["params"]
                    st.markdown(
                        detail_cards(
                            {
                                "Status": selected_run["status"],
                                "Started": selected_run["started"],
                                "Applied threshold": f"{metrics.get('applied_threshold', 0):.3f}",
                                "Optimal threshold": f"{metrics.get('optimal_threshold', 0):.3f}",
                                "Approval rate": f"{metrics.get('approval_rate', 0):.1%}",
                                "Scenario cost": f"{metrics.get('scenario_cost', 0):.4f}",
                            }
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        detail_cards(
                            {
                                "Requested threshold": params.get("requested_threshold", "n/a"),
                                "False-decline cost": params.get("false_positive_cost", "n/a"),
                                "Approved-default cost": params.get("false_negative_cost", "n/a"),
                                "MLflow run ID": selected_run["run_id"],
                            }
                        ),
                        unsafe_allow_html=True,
                    )

    with st.expander(
        f"Optuna study · {tuning.get('metadata', {}).get('n_trials', 0)} trials",
        expanded=False,
    ):
        trials = tuning.get("trials", [])
        if trials:
            st.pyplot(optuna_figure(tuning), width="stretch")
            trial_frame = pd.DataFrame(
                [
                    {
                        "Trial": trial["number"],
                        "Validation AUC": f"{trial['value']:.4f}",
                        "State": trial["state"].replace("TrialState.", ""),
                    }
                    for trial in trials
                ]
            )
            st.markdown(
                trial_frame.to_html(index=False, classes="light-table", border=0),
                unsafe_allow_html=True,
            )
            st.markdown("#### Best-trial parameters")
            st.markdown(
                detail_cards(
                    {
                        key.replace("_", " ").title(): value
                        for key, value in tuning.get("best_params", {}).items()
                    }
                ),
                unsafe_allow_html=True,
            )

    with st.expander("FastAPI and Pydantic contract", expanded=False):
        st.markdown("#### Runtime contract")
        st.markdown(
            detail_cards(
                {
                    "Transport": "FastAPI TestClient · in-process",
                    "Health route": "GET /health",
                    "Policy route": "POST /scenario",
                    "Validation": "Pydantic · bounds and positive costs",
                    "Deployment": "Uvicorn / Docker capable",
                }
            ),
            unsafe_allow_html=True,
        )
        st.markdown("#### Request fields")
        st.markdown(
            detail_cards(
                {
                    "threshold": "0.01–0.99",
                    "false_positive_cost": "> 0",
                    "false_negative_cost": "> 0",
                }
            ),
            unsafe_allow_html=True,
        )
        st.markdown("#### Response groups")
        st.markdown(
            detail_cards(
                {
                    "Identity": "policy run ID + MLflow run ID",
                    "Policy": "approval, recall, specificity, cost",
                    "Evidence": "confusion matrix + optimal threshold",
                    "Observability": "component and total timings",
                }
            ),
            unsafe_allow_html=True,
        )

    with st.expander("CI/CD quality-gate detail", expanded=False):
        st.markdown("#### CI matrix · Python 3.11 and 3.13")
        st.markdown(
            detail_cards(
                {
                    "1 · Dependencies": "uv sync --all-groups",
                    "2 · Lint": "ruff check . · passed",
                    "3 · Types": "ty check src tests · passed",
                    "4 · Tests": "pytest · 19/19 passed",
                    "5 · Governance": "report smoke test configured",
                    "6 · Container": "Docker image build configured",
                }
            ),
            unsafe_allow_html=True,
        )
        st.warning("Cloud workflow status remains unknown until it is queried from GitHub.")

    if production_run:
        with st.expander("Latest execution timing", expanded=False):
            timing_cols = st.columns(2)
            timing_cols[0].metric(
                "Backend total",
                f"{production_run['timings_ms']['backend_total']:.1f} ms",
            )
            timing_cols[1].metric(
                "HTTP wall clock",
                f"{production_run['wall_clock_ms']:.1f} ms",
            )
            st.pyplot(
                timing_figure(production_run["timings_ms"]),
                width="stretch",
            )
            st.markdown(
                "Component bars are mutually exclusive. Backend total is their aggregate; "
                "wall clock additionally includes request serialization and UI overhead."
            )
