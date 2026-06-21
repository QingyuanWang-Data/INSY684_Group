from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
TRAINING_REPORT = ARTIFACTS / "training_report.json"
MLFLOW_DB = Path(
    os.environ.get("HC_DASHBOARD_MLFLOW_DB", str(ARTIFACTS / "dashboard_mlflow.db"))
)


class ScenarioRequest(BaseModel):
    threshold: float = Field(ge=0.01, le=0.99)
    false_positive_cost: float = Field(gt=0, le=100)
    false_negative_cost: float = Field(gt=0, le=100)


class ScenarioResponse(BaseModel):
    run_id: str
    mlflow_run_id: str
    api_transport: str
    requested_threshold: float
    threshold: float
    optimal_threshold: float
    approval_rate: float
    recall: float
    specificity: float
    default_rate_within_approved: float
    scenario_cost: float
    false_declines: int
    approved_defaults: int
    confusion_matrix: list[list[int]]
    positive_count: int
    negative_count: int
    false_positive_cost: float
    false_negative_cost: float
    timings_ms: dict[str, float]


@lru_cache(maxsize=1)
def load_policy_data() -> tuple[pd.DataFrame, int, int]:
    report = json.loads(TRAINING_REPORT.read_text(encoding="utf-8"))
    curve = pd.DataFrame(
        report["threshold_optimization"]["search_curve"]["curve"]
    )
    counts = report["split_counts"]
    return (
        curve,
        int(counts["valid_positive_count"]),
        int(counts["valid_negative_count"]),
    )


def score_scenarios(request: ScenarioRequest) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    curve, positives, negatives = load_policy_data()
    scenarios = curve.copy()
    scenarios["false_declines"] = (1 - scenarios["specificity"]) * negatives
    scenarios["approved_defaults"] = (1 - scenarios["recall"]) * positives
    scenarios["scenario_cost"] = (
        scenarios["false_declines"] * request.false_positive_cost
        + scenarios["approved_defaults"] * request.false_negative_cost
    ) / (positives + negatives)
    selected_index = (scenarios["threshold"] - request.threshold).abs().idxmin()
    selected = scenarios.loc[selected_index]
    optimal = scenarios.loc[scenarios["scenario_cost"].idxmin()]
    return scenarios, selected, optimal


def log_to_mlflow(
    run_id: str,
    request: ScenarioRequest,
    selected: pd.Series,
    optimal: pd.Series,
) -> str:
    tracking_uri = f"sqlite:///{MLFLOW_DB.resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("dashboard-policy-scenarios")
    with mlflow.start_run(run_name=run_id) as run:
        mlflow.set_tag("source", "streamlit-fastapi-decision-lab")
        mlflow.log_params(
            {
                "requested_threshold": request.threshold,
                "false_positive_cost": request.false_positive_cost,
                "false_negative_cost": request.false_negative_cost,
            }
        )
        mlflow.log_metrics(
            {
                "applied_threshold": float(selected["threshold"]),
                "optimal_threshold": float(optimal["threshold"]),
                "approval_rate": float(selected["approval_rate"]),
                "default_recall": float(selected["recall"]),
                "scenario_cost": float(selected["scenario_cost"]),
            }
        )
        return run.info.run_id


app = FastAPI(title="Home Credit Policy Scenario API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/scenario", response_model=ScenarioResponse)
def run_scenario(request: ScenarioRequest) -> ScenarioResponse:
    total_started = perf_counter()
    artifact_started = perf_counter()
    _, positives, negatives = load_policy_data()
    artifact_ms = (perf_counter() - artifact_started) * 1000

    calculation_started = perf_counter()
    _, selected, optimal = score_scenarios(request)
    calculation_ms = (perf_counter() - calculation_started) * 1000

    tn = round(float(selected["specificity"]) * negatives)
    fp = negatives - tn
    tp = round(float(selected["recall"]) * positives)
    fn = positives - tp
    run_id = f"POL-{uuid4().hex[:10].upper()}"

    mlflow_started = perf_counter()
    mlflow_run_id = log_to_mlflow(run_id, request, selected, optimal)
    mlflow_ms = (perf_counter() - mlflow_started) * 1000
    total_ms = (perf_counter() - total_started) * 1000

    return ScenarioResponse(
        run_id=run_id,
        mlflow_run_id=mlflow_run_id,
        api_transport="FastAPI TestClient (in-process)",
        requested_threshold=request.threshold,
        threshold=float(selected["threshold"]),
        optimal_threshold=float(optimal["threshold"]),
        approval_rate=float(selected["approval_rate"]),
        recall=float(selected["recall"]),
        specificity=float(selected["specificity"]),
        default_rate_within_approved=float(selected["default_rate_within_approved"]),
        scenario_cost=float(selected["scenario_cost"]),
        false_declines=round(float(selected["false_declines"])),
        approved_defaults=round(float(selected["approved_defaults"])),
        confusion_matrix=[[tn, fp], [fn, tp]],
        positive_count=positives,
        negative_count=negatives,
        false_positive_cost=request.false_positive_cost,
        false_negative_cost=request.false_negative_cost,
        timings_ms={
            "artifact_load": round(artifact_ms, 2),
            "policy_calculation": round(calculation_ms, 2),
            "mlflow_logging": round(mlflow_ms, 2),
            "backend_total": round(total_ms, 2),
        },
    )
