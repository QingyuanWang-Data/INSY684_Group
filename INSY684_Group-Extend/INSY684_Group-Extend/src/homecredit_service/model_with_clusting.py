from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from homecredit_service.features import ID_COLUMN, TARGET_COLUMN, build_training_frame
from homecredit_service.modeling import (
    FOCAL_GAMMA,
    fit_encoder,
    make_focal_loss_objective,
    split_feature_types,
    transform_features,
)

logger = logging.getLogger(__name__)

TOP_K_OPTIONS = (20, 50, 70, 100, 125, 150, 200, 250, 300, 400, 450)
MISSING_TEST_AUC = -1.0


@dataclass(frozen=True)
class PreparedMatrices:
    y: pd.Series
    ids: pd.Series
    original_features: pd.DataFrame
    clustering_features: pd.DataFrame
    categorical_columns: list[str]
    numeric_columns: list[str]
    clustering_summary: dict[str, Any]


@dataclass(frozen=True)
class SplitData:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _safe_auc(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _sigmoid(raw_scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-raw_scores))


def _predict_positive_probability(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    if isinstance(model, LGBMClassifier):
        objective_name = getattr(model, "objective", None)
        if callable(objective_name):
            return _sigmoid(model.predict(matrix, raw_score=True))
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)
        return np.asarray(probabilities)[:, 1]
    decision = model.decision_function(matrix)
    return _sigmoid(decision)


def _cluster_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels.astype("int64"), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts, strict=False)}


def make_split(
    y: pd.Series,
    valid_size: float,
    test_size: float,
    random_state: int,
) -> SplitData:
    row_indices = np.arange(len(y))
    train_valid_idx, test_idx = train_test_split(
        row_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    valid_share = valid_size / (1.0 - test_size)
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_share,
        random_state=random_state,
        stratify=y.iloc[train_valid_idx],
    )
    return SplitData(
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        y_train=y.iloc[train_idx],
        y_valid=y.iloc[valid_idx],
        y_test=y.iloc[test_idx],
    )


def build_pca2_clustering_features(
    original_features: pd.DataFrame,
    random_state: int,
    kmeans_clusters: int,
    gmm_components: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    preprocessing = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=random_state)),
        ]
    )
    pca_matrix = preprocessing.fit_transform(original_features)
    pca = preprocessing.named_steps["pca"]

    kmeans = KMeans(n_clusters=kmeans_clusters, n_init="auto", random_state=random_state)
    kmeans_labels = kmeans.fit_predict(pca_matrix).astype("int64")

    gmm = GaussianMixture(
        n_components=gmm_components,
        covariance_type="diag",
        random_state=random_state,
    )
    gmm_labels = gmm.fit_predict(pca_matrix).astype("int64")

    hdbscan = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
    )
    hdbscan_labels = hdbscan.fit_predict(pca_matrix).astype("int64")

    cluster_frame = pd.DataFrame(
        {
            "PCA2_COMPONENT_1": pca_matrix[:, 0],
            "PCA2_COMPONENT_2": pca_matrix[:, 1],
            "KMEANS_PCA2_CLUSTER": kmeans_labels.astype("str"),
            "GMM_PCA2_CLUSTER": gmm_labels.astype("str"),
            "HDBSCAN_PCA2_CLUSTER": hdbscan_labels.astype("str"),
        },
        index=original_features.index,
    )
    encoded_clusters = pd.get_dummies(
        cluster_frame,
        columns=[
            "KMEANS_PCA2_CLUSTER",
            "GMM_PCA2_CLUSTER",
            "HDBSCAN_PCA2_CLUSTER",
        ],
        dtype="float32",
    )

    summary = {
        "pca_components": 2,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "kmeans_clusters": int(kmeans_clusters),
        "gmm_components": int(gmm_components),
        "hdbscan_min_cluster_size": int(hdbscan_min_cluster_size),
        "hdbscan_min_samples": hdbscan_min_samples,
        "cluster_counts": {
            "kmeans": _cluster_counts(kmeans_labels),
            "gaussian_mixture": _cluster_counts(gmm_labels),
            "hdbscan": _cluster_counts(hdbscan_labels),
        },
        "note": (
            "PCA-2 and clustering features are fit without TARGET. "
            "The labels are added as one-hot features plus PCA component values."
        ),
    }
    return encoded_clusters.astype("float32"), summary


def prepare_matrices(
    training_frame: pd.DataFrame,
    random_state: int,
    kmeans_clusters: int,
    gmm_components: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int | None,
) -> PreparedMatrices:
    y = training_frame[TARGET_COLUMN].astype("int32")
    ids = training_frame[ID_COLUMN].astype("int64")
    raw_features = training_frame.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")

    categorical_columns, numeric_columns = split_feature_types(raw_features)
    encoder = fit_encoder(raw_features, categorical_columns)
    original_features = transform_features(
        raw_features,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        encoder=encoder,
    )
    original_features = original_features.replace([np.inf, -np.inf], np.nan)
    original_features = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(original_features),
        columns=original_features.columns,
        index=original_features.index,
    ).astype("float32")

    clustering_features, clustering_summary = build_pca2_clustering_features(
        original_features=original_features,
        random_state=random_state,
        kmeans_clusters=kmeans_clusters,
        gmm_components=gmm_components,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
    )

    return PreparedMatrices(
        y=y,
        ids=ids,
        original_features=original_features,
        clustering_features=clustering_features,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        clustering_summary=clustering_summary,
    )


def _class_balance_values(y_train: pd.Series) -> dict[str, float | int]:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    total = positives + negatives
    return {
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": float(positives / total) if total else 0.0,
        "negative_rate": float(negatives / total) if total else 0.0,
        "focal_alpha": float(negatives / total) if total else 0.5,
    }


def make_lightgbm_params(random_state: int, max_estimators: int) -> dict[str, Any]:
    return {
        "objective": "binary",
        "n_estimators": max_estimators,
        "learning_rate": 0.025,
        "num_leaves": 48,
        "min_child_samples": 70,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 0.8,
        "max_depth": 12,
        "random_state": random_state,
        "n_jobs": -1,
        "metric": "auc",
        "verbosity": -1,
        "importance_type": "gain",
    }


def fit_lightgbm_focal(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int,
    max_estimators: int,
) -> LGBMClassifier:
    balance = _class_balance_values(y_train)
    params = make_lightgbm_params(random_state=random_state, max_estimators=max_estimators)
    params["objective"] = make_focal_loss_objective(
        alpha=float(balance["focal_alpha"]),
        gamma=FOCAL_GAMMA,
    )
    model = LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid), (X_train, y_train)],
        eval_names=["validation", "train"],
        eval_metric="auc",
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )
    return model


def make_random_forest(random_state: int, max_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=min(max_estimators, 500),
        max_depth=12,
        min_samples_leaf=25,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def make_logistic(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_xgboost(random_state: int, max_estimators: int) -> Any | None:
    try:
        xgboost = import_module("xgboost")
    except ModuleNotFoundError:
        logger.warning("xgboost is not installed; skipping xgboost candidate.")
        return None
    return xgboost.XGBClassifier(
        n_estimators=max_estimators,
        learning_rate=0.025,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=120,
    )


def fit_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int,
    max_estimators: int,
) -> Any:
    if model_name == "lightgbm_focal":
        return fit_lightgbm_focal(
            X_train,
            y_train,
            X_valid,
            y_valid,
            random_state,
            max_estimators,
        )
    if model_name == "random_forest":
        model = make_random_forest(random_state, max_estimators)
        model.fit(X_train, y_train)
        return model
    if model_name == "logistic_regression":
        model = make_logistic(random_state)
        model.fit(X_train, y_train)
        return model
    if model_name == "xgboost":
        model = make_xgboost(random_state, max_estimators)
        if model is None:
            return None
        balance = _class_balance_values(y_train)
        pos_weight = float(balance["negative_count"]) / max(1.0, float(balance["positive_count"]))
        sample_weight = np.where(y_train.to_numpy() == 1, pos_weight, 1.0)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def best_iteration(model: Any) -> int | None:
    for attr in ("best_iteration_", "best_iteration"):
        value = getattr(model, attr, None)
        if isinstance(value, (int, np.integer)) and int(value) >= 0:
            return int(value)
    return None


def evaluate_fitted_model(
    model: Any,
    model_name: str,
    X: pd.DataFrame,
    split: SplitData,
    feature_set_name: str,
) -> dict[str, Any]:
    train_proba = _predict_positive_probability(model, X.iloc[split.train_idx])
    valid_proba = _predict_positive_probability(model, X.iloc[split.valid_idx])
    test_proba = _predict_positive_probability(model, X.iloc[split.test_idx])
    return {
        "feature_set": feature_set_name,
        "model": model_name,
        "train_auc": _safe_auc(split.y_train, train_proba),
        "validation_auc": _safe_auc(split.y_valid, valid_proba),
        "test_auc": _safe_auc(split.y_test, test_proba),
        "train_rows": int(len(split.train_idx)),
        "valid_rows": int(len(split.valid_idx)),
        "test_rows": int(len(split.test_idx)),
        "feature_count": int(X.shape[1]),
        "best_iteration": best_iteration(model),
    }


def fit_lightgbm_feature_importance(
    X: pd.DataFrame,
    split: SplitData,
    random_state: int,
    max_estimators: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model = fit_lightgbm_focal(
        X.iloc[split.train_idx],
        split.y_train,
        X.iloc[split.valid_idx],
        split.y_valid,
        random_state,
        max_estimators,
    )
    booster = model.booster_
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False, ignore_index=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    row = evaluate_fitted_model(model, "lightgbm_focal_full", X, split, "all_original_features")
    return importance, row


def select_best_top_k_features(
    X: pd.DataFrame,
    split: SplitData,
    importance: pd.DataFrame,
    random_state: int,
    max_estimators: int,
    top_k_options: tuple[int, ...] = TOP_K_OPTIONS,
) -> tuple[list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    feature_count = X.shape[1]
    searched_k_values = sorted({min(int(raw_k), feature_count) for raw_k in top_k_options})
    for k in searched_k_values:
        selected = importance.head(k)["feature"].tolist()
        model = fit_lightgbm_focal(
            X[selected].iloc[split.train_idx],
            split.y_train,
            X[selected].iloc[split.valid_idx],
            split.y_valid,
            random_state,
            max_estimators,
        )
        row = evaluate_fitted_model(
            model,
            "lightgbm_focal_feature_selector",
            X[selected],
            split,
            f"lgbm_top_{k}_original_features",
        )
        row["top_k"] = k
        rows.append(row)

    best = max(
        rows,
        key=lambda row: row["validation_auc"] if row["validation_auc"] is not None else -1.0,
    )
    best_features = importance.head(int(best["top_k"]))["feature"].tolist()
    return best_features, rows


def evaluate_model_set(
    X: pd.DataFrame,
    split: SplitData,
    random_state: int,
    max_estimators: int,
    feature_set_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    model_names = ["lightgbm_focal", "random_forest", "xgboost", "logistic_regression"]
    for model_name in model_names:
        logger.info("Training %s on %s", model_name, feature_set_name)
        model = fit_model(
            model_name,
            X.iloc[split.train_idx],
            split.y_train,
            X.iloc[split.valid_idx],
            split.y_valid,
            random_state,
            max_estimators,
        )
        if model is None:
            continue
        row = evaluate_fitted_model(model, model_name, X, split, feature_set_name)
        rows.append(row)
        fitted_models[model_name] = model

    valid_rows = [row for row in rows if row["validation_auc"] is not None]
    best_row = max(valid_rows, key=lambda row: row["validation_auc"])
    return rows, {"row": best_row, "model": fitted_models[best_row["model"]]}


def _format_auc(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Feature set",
        "Model",
        "Train AUC",
        "Validation AUC",
        "Test AUC",
        "Features",
        "Best iteration",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(rows, key=lambda item: item.get("validation_auc") or -1, reverse=True):
        lines.append(
            "| "
            f"{row['feature_set']} | "
            f"{row['model']} | "
            f"{_format_auc(row['train_auc'])} | "
            f"{_format_auc(row['validation_auc'])} | "
            f"{_format_auc(row['test_auc'])} | "
            f"{row['feature_count']} | "
            f"{row.get('best_iteration') or ''} |"
        )
    return "\n".join(lines)


def _feature_selection_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Top K | Train AUC | Validation AUC | Test AUC | Best iteration |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['top_k']} | {_format_auc(row['train_auc'])} | "
            f"{_format_auc(row['validation_auc'])} | {_format_auc(row['test_auc'])} | "
            f"{row.get('best_iteration') or ''} |"
        )
    return "\n".join(lines)


def write_report(
    artifact_dir: Path,
    report: dict[str, Any],
    all_results: list[dict[str, Any]],
    feature_importance: pd.DataFrame,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifact_dir / "model_with_clusting_report.json"
    markdown_path = artifact_dir / "model_with_clusting_report.md"
    results_path = artifact_dir / "model_with_clusting_results.csv"
    importance_path = artifact_dir / "lightgbm_feature_importance.csv"

    json_path.write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    pd.DataFrame(all_results).to_csv(results_path, index=False)
    feature_importance.to_csv(importance_path, index=False)

    best = report["best_overall"]
    selected = report["selected_features"]
    markdown = [
        "# Model With Clusting Report",
        "",
        "## Best Overall By Validation ROC-AUC",
        f"- Feature set: `{best['feature_set']}`",
        f"- Model: `{best['model']}`",
        f"- Validation ROC-AUC: `{_format_auc(best['validation_auc'])}`",
        f"- Test ROC-AUC: `{_format_auc(best['test_auc'])}`",
        f"- Train ROC-AUC: `{_format_auc(best['train_auc'])}`",
        f"- Best iteration: `{best.get('best_iteration')}`",
        "",
        "## Split Counts",
        f"- Train rows: `{report['split_counts']['train_rows']}`",
        f"- Validation rows: `{report['split_counts']['valid_rows']}`",
        f"- Test rows: `{report['split_counts']['test_rows']}`",
        "",
        "## LightGBM Feature Selection",
        "LightGBM focal-loss feature importance was fit on the training split with "
        "early stopping on validation data. Top 20, 50, 70, and 100 subsets were "
        "selected by validation ROC-AUC, matching the original train/valid/test logic.",
        "",
        _feature_selection_table(report["feature_selection_results"]),
        "",
        f"- Selected top-k: `{selected['top_k']}`",
        "- Selected features:",
        "",
    ]
    markdown.extend([f"  - `{feature}`" for feature in selected["features"]])
    markdown.extend(
        [
            "",
            "## Final Model Comparison",
            "Selected original features were used in three feature sets: selected only, "
            "selected plus PCA-2 clusting features, and clusting only.",
            "",
            _markdown_table(all_results),
            "",
            "## Best By Feature Set",
        ]
    )
    for feature_set, row in report["best_by_feature_set"].items():
        markdown.append(
            f"- `{feature_set}`: `{row['model']}` validation AUC "
            f"`{_format_auc(row['validation_auc'])}`, test AUC `{_format_auc(row['test_auc'])}`"
        )
    markdown.extend(
        [
            "",
            "## Imbalance And Early Stopping Logic",
            "- LightGBM uses the same custom focal-loss objective pattern as the old file.",
            "- LightGBM uses validation early stopping and reports best iteration.",
            "- Random Forest and Logistic Regression use balanced class weights.",
            "- XGBoost uses positive-class sample weights and validation early stopping.",
            "",
            "## Clusting Features Added",
            "The clusting feature set contains PCA-2 component values plus one-hot labels from "
            "KMeans, Gaussian Mixture, and HDBSCAN.",
            "",
            "```json",
            json.dumps(_json_ready(report["clustering_summary"]), indent=2),
            "```",
            "",
            "## Files",
            f"- Results CSV: `{results_path}`",
            f"- Feature importance CSV: `{importance_path}`",
            f"- Best model artifact: `{report['model_artifact_path']}`",
        ]
    )
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare models after LightGBM feature selection and PCA-2 clusting features."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("homecreditdefaultriskdata"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/model_with_clusting"))
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--max-estimators", type=int, default=3000)
    parser.add_argument("--kmeans-clusters", type=int, default=5)
    parser.add_argument("--gmm-components", type=int, default=5)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=100)
    parser.add_argument("--hdbscan-min-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    if not 0.0 < args.valid_size < 1.0:
        raise ValueError("--valid-size must be between 0 and 1.")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.valid_size + args.test_size >= 1.0:
        raise ValueError("--valid-size + --test-size must be less than 1.")

    training_frame = build_training_frame(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    prepared = prepare_matrices(
        training_frame=training_frame,
        random_state=args.random_state,
        kmeans_clusters=args.kmeans_clusters,
        gmm_components=args.gmm_components,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
    )
    split = make_split(
        prepared.y,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    feature_importance, full_lgbm_row = fit_lightgbm_feature_importance(
        X=prepared.original_features,
        split=split,
        random_state=args.random_state,
        max_estimators=args.max_estimators,
    )
    selected_features, feature_selection_results = select_best_top_k_features(
        X=prepared.original_features,
        split=split,
        importance=feature_importance,
        random_state=args.random_state,
        max_estimators=args.max_estimators,
    )

    selected_original = prepared.original_features[selected_features]
    selected_plus_clusting = pd.concat(
        [selected_original, prepared.clustering_features],
        axis=1,
    )
    clusting_only = prepared.clustering_features

    original_rows, best_original = evaluate_model_set(
        X=selected_original,
        split=split,
        random_state=args.random_state,
        max_estimators=args.max_estimators,
        feature_set_name="selected_original_features",
    )
    enhanced_rows, best_enhanced = evaluate_model_set(
        X=selected_plus_clusting,
        split=split,
        random_state=args.random_state,
        max_estimators=args.max_estimators,
        feature_set_name="selected_original_plus_clusting",
    )
    clusting_rows, best_clusting_only = evaluate_model_set(
        X=clusting_only,
        split=split,
        random_state=args.random_state,
        max_estimators=args.max_estimators,
        feature_set_name="clusting_only",
    )

    all_results = original_rows + enhanced_rows + clusting_rows
    valid_results = [row for row in all_results if row["validation_auc"] is not None]
    best_overall = max(valid_results, key=lambda row: row["validation_auc"])

    best_by_feature_set = {
        "selected_original_features": best_original["row"],
        "selected_original_plus_clusting": best_enhanced["row"],
        "clusting_only": best_clusting_only["row"],
    }
    best_model_map = {
        "selected_original_features": best_original["model"],
        "selected_original_plus_clusting": best_enhanced["model"],
        "clusting_only": best_clusting_only["model"],
    }

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifact_dir / "best_model_with_clusting.joblib"
    joblib.dump(best_model_map[best_overall["feature_set"]], model_path)

    best_k_row = max(
        feature_selection_results,
        key=lambda row: row["validation_auc"] if row["validation_auc"] is not None else -1.0,
    )
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_metric": "validation_roc_auc",
        "model_artifact_path": str(model_path),
        "full_original_lightgbm_focal": full_lgbm_row,
        "best_overall": best_overall,
        "best_by_feature_set": best_by_feature_set,
        "selected_features": {
            "top_k": int(best_k_row["top_k"]),
            "features": selected_features,
        },
        "feature_selection_results": feature_selection_results,
        "split_counts": {
            "train_rows": int(len(split.train_idx)),
            "valid_rows": int(len(split.valid_idx)),
            "test_rows": int(len(split.test_idx)),
        },
        "class_balance": _class_balance_values(split.y_train),
        "clustering_summary": prepared.clustering_summary,
        "all_results": all_results,
        "notes": [
            "Feature selection uses LightGBM focal-loss gain importance.",
            "Feature subset and final winner are selected by validation ROC-AUC.",
            "Test ROC-AUC is held out for final diagnostics, matching the old-file logic.",
            "XGBoost is skipped automatically if xgboost is not installed.",
        ],
    }
    write_report(args.artifact_dir, report, all_results, feature_importance)
    print(json.dumps(_json_ready(report), indent=2))


if __name__ == "__main__":
    main()
