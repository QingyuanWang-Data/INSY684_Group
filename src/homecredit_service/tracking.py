from __future__ import annotations

import json
import logging
from importlib import import_module
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


def _numeric_metrics(prefix: str, payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        metric_name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[metric_name] = float(value)
        elif isinstance(value, dict):
            metrics.update(_numeric_metrics(metric_name, value))
    return metrics


def log_training_run_to_mlflow(
    report: dict[str, Any],
    artifact_dir: Path,
    experiment_name: str,
    tracking_uri: str | None = None,
    run_name: str | None = None,
) -> str | None:
    try:
        mlflow = cast(Any, import_module("mlflow"))
    except ModuleNotFoundError:
        logger.warning("MLflow is not installed; skipping experiment logging.")
        return None

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(
            {
                "random_seed": report.get("random_seed"),
                "feature_count": report.get("feature_count"),
                "threshold_used": report.get("threshold_used"),
                "threshold_default": report.get("threshold_default"),
                "imbalance_strategy": report.get("imbalance_strategy"),
                "test_evaluated": report.get("test_evaluated"),
            }
        )

        hyperparameters = report.get("hyperparameters", {})
        if isinstance(hyperparameters, dict):
            mlflow.log_params(
                {
                    f"model_{key}": value
                    for key, value in hyperparameters.items()
                    if isinstance(value, (str, int, float, bool))
                }
            )

        metrics = _numeric_metrics("roc_auc", report.get("roc_auc", {}))
        pr_auc = report.get("pr_auc", {})
        if isinstance(pr_auc, dict):
            metrics.update(
                {
                    f"pr_auc_{key}": float(value)
                    for key, value in pr_auc.items()
                    if isinstance(value, (int, float))
                }
            )
        mlflow.log_metrics(metrics)

        report_path = artifact_dir / "training_report.json"
        if report_path.exists():
            mlflow.log_artifact(str(report_path), artifact_path="reports")

        plots_dir = artifact_dir / "plots"
        if plots_dir.exists():
            mlflow.log_artifacts(str(plots_dir), artifact_path="plots")

        mlflow.log_text(
            json.dumps(report, indent=2),
            artifact_file="reports/report_snapshot.json",
        )
        return active_run.info.run_id
