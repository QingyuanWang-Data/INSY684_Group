# Home Credit Decision Lab

This directory contains the Streamlit policy dashboard and the clustering/model
evidence used by the integrated INSY684 project. The canonical setup and launch
instructions live in the repository root `README.md`.

## Recommended launch

From the repository root:

```bash
uv sync --all-groups --locked
uv run --group dashboard streamlit run INSY684_Group-Extend/INSY684_Group-Extend/monitoring_dashboard/monitoring_dashboard.py
```

On Windows, double-click the root-level `run_dashboard.cmd`. The compatibility
launcher in this directory delegates to that portable launcher and does not
assume a user-specific Python installation.

## What the Run button does

`Run via FastAPI` validates the threshold and cost inputs with Pydantic, calls
the in-process FastAPI policy endpoint, recalculates validation decisions and
costs from saved model scores, and records the scenario in MLflow. It is a
post-training policy simulation; it does not retrain LightGBM.

## Optional offline report

```bash
uv run --group dashboard python INSY684_Group-Extend/INSY684_Group-Extend/monitoring_dashboard/generate_monitoring_html_offline.py
```

The report is written to `artifacts/monitoring_dashboard.html` inside this
directory.

## Retraining

Retraining requires the raw Home Credit CSV files described in the root
`README.md`. Use the canonical root training command so feature engineering,
validation, MLflow and governance artifacts stay synchronized:

```bash
uv run train-homecredit --data-dir homecreditdefaultriskdata --artifact-dir artifacts --sample-size 50000
```
