from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from homecredit_service.features import build_training_frame
from homecredit_service.modeling import train_lightgbm_model

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Home Credit LightGBM hyperparameters.")
    parser.add_argument("--data-dir", type=Path, default=Path("homecreditdefaultriskdata"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--max-estimators", type=int, default=900)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--subgroup-min-size", type=int, default=500)
    return parser.parse_args()


def suggest_lightgbm_params(
    trial: Any,
    random_state: int,
    max_estimators: int,
) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, max_estimators),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 24, 96),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 140),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.6),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 2.0),
        "max_depth": trial.suggest_int("max_depth", 5, 14),
        "random_state": random_state,
    }


def write_tuning_outputs(study: Any, output_dir: Path, metadata: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "tuning_report.json"
    trials_path = output_dir / "tuning_trials.csv"

    trials: list[dict[str, Any]] = []
    for trial in study.trials:
        trials.append(
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": trial.params,
            }
        )

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "metadata": metadata,
        "trials": trials,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with trials_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["number", "state", "value", "params"])
        writer.writeheader()
        for row in trials:
            writer.writerow({**row, "params": json.dumps(row["params"])})

    return report_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()

    try:
        import optuna
    except ModuleNotFoundError as exc:
        msg = (
            "Optuna is required for tuning. "
            "Install dependencies with `pip install -r requirements.txt`."
        )
        raise RuntimeError(msg) from exc

    training_frame = build_training_frame(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    def objective(trial: Any) -> float:
        params = suggest_lightgbm_params(
            trial=trial,
            random_state=args.random_state,
            max_estimators=args.max_estimators,
        )
        bundle = train_lightgbm_model(
            train_df=training_frame,
            random_state=args.random_state,
            valid_size=args.valid_size,
            test_size=args.test_size,
            evaluate_test=False,
            cv_folds=args.cv_folds,
            temporal_max_estimators=min(args.max_estimators, 400),
            subgroup_min_size=args.subgroup_min_size,
            hyperparameter_overrides=params,
        )
        return float(bundle["metrics"]["validation_auc"])

    sampler = optuna.samplers.TPESampler(seed=args.random_state)
    study = optuna.create_study(
        direction="maximize",
        study_name="homecredit_lightgbm_tuning",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    report_path = write_tuning_outputs(
        study=study,
        output_dir=args.artifact_dir,
        metadata={
            "sample_size": args.sample_size,
            "n_trials": args.n_trials,
            "timeout": args.timeout,
            "random_state": args.random_state,
            "metric": "validation_auc",
        },
    )
    logger.info("Tuning report saved to %s", report_path)
    print(json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2))


if __name__ == "__main__":
    main()
