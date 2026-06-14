from __future__ import annotations

import json
from pathlib import Path

from homecredit_service.tune import suggest_lightgbm_params, write_tuning_outputs


class DummyTrial:
    number = 0
    state = "COMPLETE"
    value = 0.75

    def __init__(self) -> None:
        self.params: dict[str, float | int] = {}

    def suggest_int(self, name: str, low: int, high: int) -> int:
        value = low + ((high - low) // 2)
        self.params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        value = (low * high) ** 0.5 if log else (low + high) / 2.0
        self.params[name] = value
        return value


class DummyStudy:
    def __init__(self) -> None:
        self.best_value = 0.75
        self.best_params = {"num_leaves": 60}
        self.trials = [DummyTrial()]


def test_suggest_lightgbm_params_respects_ranges() -> None:
    trial = DummyTrial()
    params = suggest_lightgbm_params(trial=trial, random_state=42, max_estimators=500)

    assert 200 <= params["n_estimators"] <= 500
    assert 0.01 <= params["learning_rate"] <= 0.08
    assert params["random_state"] == 42


def test_write_tuning_outputs(tmp_path: Path) -> None:
    report_path = write_tuning_outputs(
        study=DummyStudy(),
        output_dir=tmp_path,
        metadata={"metric": "validation_auc"},
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["best_value"] == 0.75
    assert report["metadata"]["metric"] == "validation_auc"
    assert (tmp_path / "tuning_trials.csv").exists()
