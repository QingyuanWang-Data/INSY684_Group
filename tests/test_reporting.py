from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from homecredit_service.reporting import (
    build_fairness_report,
    build_model_card,
    build_monitoring_report,
    summarize_group_gaps,
)


def test_summarize_group_gaps_orders_largest_gap_first() -> None:
    frame = pd.DataFrame(
        {
            "column": ["A", "A", "B", "B"],
            "group": ["x", "y", "m", "n"],
            "auc": [0.8, 0.7, 0.9, 0.88],
            "pr_auc": [0.2, 0.1, 0.5, 0.49],
            "recall": [0.6, 0.2, 0.4, 0.39],
            "specificity": [0.8, 0.7, 0.9, 0.89],
            "approval_rate": [0.5, 0.9, 0.7, 0.71],
            "default_rate_within_approved": [0.05, 0.07, 0.04, 0.05],
        }
    )

    gaps = summarize_group_gaps(frame)

    assert gaps[0].column == "A"
    assert gaps[0].metric in {"recall", "approval_rate"}
    assert gaps[0].gap == 0.4


def test_build_reports_from_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    subgroup = pd.DataFrame(
        {
            "column": ["CODE_GENDER", "CODE_GENDER"],
            "group": ["F", "M"],
            "count": [100, 90],
            "positive_rate": [0.07, 0.10],
            "auc": [0.78, 0.77],
            "pr_auc": [0.25, 0.30],
            "recall": [0.41, 0.53],
            "specificity": [0.91, 0.84],
            "approval_rate": [0.89, 0.80],
            "default_rate_within_approved": [0.046, 0.060],
        }
    )
    subgroup.to_csv(artifact_dir / "validation_subgroup_metrics.csv", index=False)
    subgroup.to_csv(artifact_dir / "test_subgroup_metrics.csv", index=False)

    (artifact_dir / "training_report.json").write_text(
        json.dumps(
            {
                "train_auc": 0.9,
                "validation_auc": 0.78,
                "test_auc": 0.79,
                "threshold_used": 0.55,
                "imbalance_strategy": "scale_pos_weight",
                "dataset_size": {"rows": 1000, "columns": 20},
                "class_distribution": {"positive_rate": 0.1, "negative_rate": 0.9},
                "hyperparameters": {
                    "objective": "binary",
                    "learning_rate": 0.03,
                    "num_leaves": 48,
                    "max_depth": 12,
                },
                "pr_auc": {"validation": 0.28},
                "calibration": {
                    "validation": {
                        "brier_score": 0.18,
                        "expected_calibration_error": 0.34,
                    }
                },
                "policy_simulation": {
                    "splits": {
                        "validation": {
                            "approval_rate": 0.86,
                            "default_rate_within_approved": 0.05,
                            "expected_cost_per_applicant": 0.32,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "drift_summary.json").write_text(
        json.dumps(
            {
                "score_distribution": {
                    "train_vs_validation_psi": 0.01,
                    "train_vs_validation_severity": "stable",
                    "train_vs_test_psi": 0.02,
                    "train_vs_test_severity": "stable",
                },
                "feature_distribution": {
                    "rows": [
                        {
                            "feature": "AMT_CREDIT",
                            "train_vs_validation_psi": 0.03,
                            "train_vs_validation_severity": "stable",
                            "train_vs_test_psi": 0.04,
                            "train_vs_test_severity": "stable",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    fairness = build_fairness_report(artifact_dir)
    monitoring = build_monitoring_report(artifact_dir)
    model_card = build_model_card(artifact_dir)

    assert "Fairness and Ethical AI Report" in fairness
    assert "CODE_GENDER" in fairness
    assert "Model Monitoring Report" in monitoring
    assert "AMT_CREDIT" in monitoring
    assert "Model Card" in model_card
    assert "Intended Use" in model_card
