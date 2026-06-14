import json
from pathlib import Path

from src.data_processing import (
    build_data_profile,
    build_drift_baseline,
    build_training_frame,
    run_data_quality_checks,
)

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("artifacts/data_profile")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_frame = build_training_frame(DATA_DIR)

    print(f"Final training frame shape: {training_frame.shape}")

    profile = build_data_profile(training_frame)
    profile.to_csv(OUTPUT_DIR / "training_frame_profile.csv", index=False)

    quality_checks = run_data_quality_checks(training_frame)
    with open(OUTPUT_DIR / "data_quality_checks.json", "w") as f:
        json.dump(quality_checks, f, indent=2)

    class_balance = (
        training_frame["TARGET"]
        .value_counts()
        .rename_axis("target")
        .reset_index(name="count")
    )
    class_balance["rate"] = class_balance["count"] / class_balance["count"].sum()
    class_balance.to_csv(OUTPUT_DIR / "class_balance.csv", index=False)

    drift_baseline = build_drift_baseline(training_frame)
    drift_baseline.to_csv(OUTPUT_DIR / "drift_baseline.csv", index=False)

    print("Saved outputs to artifacts/data_profile/")


if __name__ == "__main__":
    main()