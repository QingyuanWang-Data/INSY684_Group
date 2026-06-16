# INSY684_Group

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
