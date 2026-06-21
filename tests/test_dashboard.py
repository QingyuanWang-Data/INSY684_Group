from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = (
    REPO_ROOT
    / "INSY684_Group-Extend"
    / "INSY684_Group-Extend"
    / "monitoring_dashboard"
)


def test_dashboard_renders_and_runs_policy_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(DASHBOARD_DIR)
    monkeypatch.setenv("HC_DASHBOARD_MLFLOW_DB", str(tmp_path / "dashboard_mlflow.db"))
    sys.modules.pop("scenario_backend", None)
    sys.path.insert(0, str(DASHBOARD_DIR))

    try:
        app = AppTest.from_file("monitoring_dashboard.py", default_timeout=60).run()
        assert not app.exception
        assert [tab.label for tab in app.tabs[:5]] == [
            "Decision Lab",
            "Modeling",
            "Feature Engineering",
            "Evidence",
            "Production Stack",
        ]

        threshold = next(slider for slider in app.slider if slider.label == "Decision threshold")
        run_button = next(button for button in app.button if button.label == "Run via FastAPI")
        threshold.set_value(0.45)
        run_button.click()
        app.run()

        assert not app.exception
        assert any("HTTP 200" in message.value for message in app.success)
        assert any("Dynamic evidence" in message.value for message in app.success)
    finally:
        sys.path.remove(str(DASHBOARD_DIR))
        sys.modules.pop("scenario_backend", None)
