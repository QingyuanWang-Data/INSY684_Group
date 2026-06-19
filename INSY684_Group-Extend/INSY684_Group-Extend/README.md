# INSY684 Group — Extended

## Run the monitoring dashboard

Recommended on Windows: double-click `run_dashboard.cmd`, or run it from CMD:

```cmd
run_dashboard.cmd
```

This launcher always uses the compatible Anaconda environment and checks all dashboard
dependencies before starting. To launch manually from this project directory:

```powershell
& "C:\Users\ayuan\anaconda3\python.exe" -m streamlit run monitoring_dashboard/monitoring_dashboard.py
```

The dashboard resolves `artifacts/` relative to the project, so the command also works
when launched from another directory. You may select a different artifact folder in the
sidebar.

To generate the self-contained offline HTML report:

```powershell
& "C:\Users\ayuan\anaconda3\python.exe" monitoring_dashboard/generate_monitoring_html_offline.py
```

The generated report is written to `artifacts/monitoring_dashboard.html`.

## Train the model

run the model file:example
uv run python -m homecredit_service.model_with_clusting `
  --data-dir "C:\Users\Hank\Desktop\study\INSY684\Assignment\data" `
  --artifact-dir "artifacts/model_with_clusting" `
  --sample-size 50000 `
  --valid-size 0.2 `
  --test-size 0.1 `
  --max-estimators 3000 `
  --kmeans-clusters 2 `
  --gmm-components 5 `
  --hdbscan-min-cluster-size 50
