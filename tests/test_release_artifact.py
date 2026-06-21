from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from homecredit_service.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "model_bundle.joblib"
REPORT_PATH = REPO_ROOT / "artifacts" / "training_report.json"


def test_versioned_release_artifact_is_ready_and_matches_report() -> None:
    """Protect the exact model bundle that reviewers receive in the repository."""
    assert ARTIFACT_PATH.is_file(), "The release model bundle must be versioned."

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    app = create_app(artifact_path=ARTIFACT_PATH)

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200

        response = client.get("/metadata")
        assert response.status_code == 200
        metadata = response.json()

    assert metadata["feature_count"] == report["feature_count"] == 698
    assert metadata["test_auc"] == pytest.approx(report["test_auc"])
    assert metadata["validation_auc"] == pytest.approx(report["validation_auc"])
    assert metadata["test_pr_auc"] == pytest.approx(report["pr_auc"]["test"])
    assert metadata["test_rows"] == report["test_rows"] == 5000
