UV ?= uv
DATA_DIR ?= homecreditdefaultriskdata
ARTIFACT_DIR ?= artifacts

.PHONY: install lock lint format typecheck test quality train train-mlflow tune reports mlflow-ui final-eval run dashboard

install:
	$(UV) sync --all-groups

lock:
	$(UV) lock

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

typecheck:
	$(UV) run ty check src tests

test:
	$(UV) run pytest

quality: lint typecheck test

train:
	$(UV) run train-homecredit --data-dir $(DATA_DIR) --artifact-dir $(ARTIFACT_DIR)

train-mlflow:
	$(UV) run train-homecredit --data-dir $(DATA_DIR) --artifact-dir $(ARTIFACT_DIR) --enable-mlflow

tune:
	$(UV) run tune-homecredit --data-dir $(DATA_DIR) --artifact-dir $(ARTIFACT_DIR)

reports:
	$(UV) run generate-governance-reports --artifact-dir $(ARTIFACT_DIR) --docs-dir docs

mlflow-ui:
	$(UV) run mlflow ui --backend-store-uri mlruns

final-eval:
	$(UV) run train-homecredit --data-dir $(DATA_DIR) --artifact-dir $(ARTIFACT_DIR) --final-eval

run:
	HC_ARTIFACT_PATH=$(ARTIFACT_DIR)/model_bundle.joblib $(UV) run uvicorn homecredit_service.main:app --host 0.0.0.0 --port 8000

dashboard:
	$(UV) run --group dashboard streamlit run INSY684_Group-Extend/INSY684_Group-Extend/monitoring_dashboard/monitoring_dashboard.py
