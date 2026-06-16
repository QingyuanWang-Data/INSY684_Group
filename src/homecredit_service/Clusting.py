from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from homecredit_service.features import ID_COLUMN, TARGET_COLUMN, build_training_frame
from homecredit_service.modeling import fit_encoder, split_feature_types, transform_features

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedClusteringData:
    ids: pd.Series
    raw_features: pd.DataFrame
    target: pd.Series | None
    matrix: np.ndarray
    plot_matrix: np.ndarray
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    preprocessing: dict[str, Any]
    pca_diagnostics: pd.DataFrame
    pca_n_90: int | None


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


def _safe_score_value(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _nearest_pca_90(cumulative: np.ndarray) -> int | None:
    if len(cumulative) == 0:
        return None
    reached = np.flatnonzero(cumulative >= 0.90)
    if len(reached) > 0:
        return int(reached[0] + 1)
    return int(np.argmin(np.abs(cumulative - 0.90)) + 1)


def prepare_clustering_data(
    training_frame: pd.DataFrame,
    random_state: int = 42,
    pca_components: int | None = 25,
) -> PreparedClusteringData:
    ids = training_frame[ID_COLUMN].copy()
    target = (
        training_frame[TARGET_COLUMN].astype("int32").copy()
        if TARGET_COLUMN in training_frame.columns
        else None
    )
    raw_features = training_frame.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")
    categorical_columns, numeric_columns = split_feature_types(raw_features)
    encoder = fit_encoder(raw_features, categorical_columns)
    transformed = transform_features(raw_features, categorical_columns, numeric_columns, encoder)

    base_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    scaled_matrix = np.asarray(base_pipeline.fit_transform(transformed), dtype=np.float32)
    max_components = min(scaled_matrix.shape[0], scaled_matrix.shape[1])

    pca_diagnostics = pd.DataFrame()
    pca_n_90: int | None = None
    if max_components >= 2:
        full_pca = PCA(n_components=max_components, random_state=random_state)
        full_pca.fit(scaled_matrix)
        cumulative = np.cumsum(full_pca.explained_variance_ratio_)
        pca_n_90 = _nearest_pca_90(cumulative)
        pca_diagnostics = pd.DataFrame(
            {
                "component": np.arange(1, max_components + 1),
                "explained_variance_ratio": full_pca.explained_variance_ratio_,
                "cumulative_explained_variance": cumulative,
                "information_lost": 1.0 - cumulative,
                "distance_to_90pct": np.abs(cumulative - 0.90),
            }
        )

    pca_effective_components: int | None = None
    matrix = scaled_matrix
    if pca_components is not None and pca_components > 0 and max_components >= 2:
        pca_effective_components = min(int(pca_components), max_components)
        pca = PCA(n_components=pca_effective_components, random_state=random_state)
        matrix = np.asarray(pca.fit_transform(scaled_matrix), dtype=np.float32)

    plot_components = min(2, max_components)
    plot_matrix = scaled_matrix[:, :2]
    if max_components >= 2:
        plot_pca = PCA(n_components=plot_components, random_state=random_state)
        plot_matrix = np.asarray(plot_pca.fit_transform(scaled_matrix), dtype=np.float32)

    explained_variance = None
    if pca_effective_components is not None and not pca_diagnostics.empty:
        explained_variance = float(
            pca_diagnostics.loc[
                pca_diagnostics["component"] == pca_effective_components,
                "cumulative_explained_variance",
            ].iloc[0]
        )

    return PreparedClusteringData(
        ids=ids,
        raw_features=raw_features,
        target=target,
        matrix=matrix,
        plot_matrix=plot_matrix,
        feature_columns=transformed.columns.tolist(),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        preprocessing={
            "imputer": "median",
            "scaler": "standard",
            "pca_requested_components": pca_components,
            "pca_effective_components": pca_effective_components,
            "pca_explained_variance_ratio_sum": explained_variance,
            "pca_n_closest_or_reaching_90pct": pca_n_90,
        },
        pca_diagnostics=pca_diagnostics,
        pca_n_90=pca_n_90,
    )


def _cluster_count(labels: np.ndarray, include_noise: bool = False) -> int:
    unique_labels = set(labels.astype("int64").tolist())
    if not include_noise:
        unique_labels.discard(-1)
    return len(unique_labels)


def score_labels(
    matrix: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    silhouette_sample_size: int,
) -> dict[str, float | int | None]:
    labels = labels.astype("int64", copy=False)
    non_noise_mask = labels != -1
    scoring_matrix = matrix[non_noise_mask]
    scoring_labels = labels[non_noise_mask]
    cluster_count = _cluster_count(labels)
    noise_count = int(np.sum(labels == -1))

    metrics: dict[str, float | int | None] = {
        "cluster_count": int(cluster_count),
        "noise_count": noise_count,
        "noise_rate": float(noise_count / len(labels)) if len(labels) else 0.0,
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }
    if cluster_count < 2 or len(scoring_labels) <= cluster_count:
        return metrics

    sample_size = min(int(silhouette_sample_size), len(scoring_labels))
    if sample_size >= 3:
        metrics["silhouette"] = float(
            silhouette_score(
                scoring_matrix,
                scoring_labels,
                sample_size=sample_size,
                random_state=random_state,
            )
        )
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(scoring_matrix, scoring_labels))
    metrics["davies_bouldin"] = float(davies_bouldin_score(scoring_matrix, scoring_labels))
    return metrics


def run_kmeans(
    matrix: np.ndarray,
    n_clusters: int,
    random_state: int,
    silhouette_sample_size: int,
) -> dict[str, Any]:
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = model.fit_predict(matrix).astype("int64")
    metrics = score_labels(matrix, labels, random_state, silhouette_sample_size)
    metrics["inertia"] = float(model.inertia_)
    return {
        "labels": labels,
        "parameters": {"n_clusters": n_clusters},
        "metrics": metrics,
    }


def run_hdbscan(
    matrix: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
    random_state: int,
    silhouette_sample_size: int,
) -> dict[str, Any]:
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = model.fit_predict(matrix).astype("int64")
    metrics = score_labels(matrix, labels, random_state, silhouette_sample_size)
    return {
        "labels": labels,
        "parameters": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
        },
        "metrics": metrics,
    }


def run_gaussian_mixture(
    matrix: np.ndarray,
    n_components: int,
    covariance_type: str,
    random_state: int,
    silhouette_sample_size: int,
) -> dict[str, Any]:
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    labels = model.fit_predict(matrix).astype("int64")
    metrics = score_labels(matrix, labels, random_state, silhouette_sample_size)
    metrics["bic"] = float(model.bic(matrix))
    metrics["aic"] = float(model.aic(matrix))
    return {
        "labels": labels,
        "parameters": {
            "n_components": n_components,
            "covariance_type": covariance_type,
        },
        "metrics": metrics,
    }


def _best_by_silhouette(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        candidate
        for candidate in candidates
        if isinstance(candidate["metrics"].get("silhouette"), (int, float))
    ]
    if not valid:
        return candidates[0]
    return max(valid, key=lambda candidate: candidate["metrics"]["silhouette"])


def run_kmeans_scan(
    matrix: np.ndarray,
    n_min: int,
    n_max: int,
    random_state: int,
    silhouette_sample_size: int,
) -> dict[str, Any]:
    candidates = [
        run_kmeans(matrix, n, random_state, silhouette_sample_size)
        for n in range(n_min, n_max + 1)
    ]
    best = dict(_best_by_silhouette(candidates))
    best["candidate_scores"] = [
        {
            "n_clusters": row["parameters"]["n_clusters"],
            "silhouette": row["metrics"].get("silhouette"),
            "calinski_harabasz": row["metrics"].get("calinski_harabasz"),
            "davies_bouldin": row["metrics"].get("davies_bouldin"),
            "inertia": row["metrics"].get("inertia"),
        }
        for row in candidates
    ]
    best["candidate_results"] = candidates
    best["parameters"] = {
        **best["parameters"],
        "selection_metric": "silhouette",
        "searched_n_min": n_min,
        "searched_n_max": n_max,
    }
    return best


def run_gaussian_mixture_scan(
    matrix: np.ndarray,
    n_min: int,
    n_max: int,
    covariance_type: str,
    random_state: int,
    silhouette_sample_size: int,
) -> dict[str, Any]:
    candidates = [
        run_gaussian_mixture(
            matrix,
            n,
            covariance_type,
            random_state,
            silhouette_sample_size,
        )
        for n in range(n_min, n_max + 1)
    ]
    best = dict(_best_by_silhouette(candidates))
    best["candidate_scores"] = [
        {
            "n_components": row["parameters"]["n_components"],
            "silhouette": row["metrics"].get("silhouette"),
            "calinski_harabasz": row["metrics"].get("calinski_harabasz"),
            "davies_bouldin": row["metrics"].get("davies_bouldin"),
            "bic": row["metrics"].get("bic"),
            "aic": row["metrics"].get("aic"),
        }
        for row in candidates
    ]
    best["candidate_results"] = candidates
    best["parameters"] = {
        **best["parameters"],
        "selection_metric": "silhouette",
        "searched_n_min": n_min,
        "searched_n_max": n_max,
    }
    return best


def cluster_size_rows(labels: np.ndarray, target: pd.Series | None) -> list[dict[str, Any]]:
    labels_series = pd.Series(labels, name="cluster")
    rows: list[dict[str, Any]] = []
    for label, count in labels_series.value_counts().sort_index().items():
        mask = labels_series == label
        row: dict[str, Any] = {
            "cluster": int(label),
            "count": int(count),
            "share": float(count / len(labels_series)),
        }
        if target is not None:
            row["target_rate"] = float(target.loc[mask.to_numpy()].mean())
        rows.append(row)
    return rows


def profile_numeric_features(
    raw_features: pd.DataFrame,
    labels: np.ndarray,
    top_features: int,
) -> pd.DataFrame:
    numeric = raw_features.select_dtypes(include=[np.number]).apply(
        pd.to_numeric,
        errors="coerce",
    )
    if numeric.empty:
        return pd.DataFrame()

    overall_mean = numeric.mean(numeric_only=True)
    overall_std = numeric.std(numeric_only=True).replace({0.0: np.nan})
    standardized_diff = (
        numeric.assign(cluster=labels)
        .groupby("cluster")
        .mean(numeric_only=True)
        .subtract(overall_mean, axis="columns")
        .divide(overall_std, axis="columns")
    )

    rows: list[dict[str, Any]] = []
    for cluster_label, cluster_values in standardized_diff.iterrows():
        ranked = cluster_values.abs().sort_values(ascending=False).head(top_features).index
        for feature in ranked:
            rows.append(
                {
                    "cluster": int(cluster_label),
                    "feature": str(feature),
                    "cluster_mean": float(
                        numeric.loc[labels == cluster_label, feature].mean(skipna=True)
                    ),
                    "overall_mean": float(overall_mean[feature]),
                    "standardized_difference": float(cluster_values[feature]),
                }
            )
    return pd.DataFrame(rows)


def cluster_feature_percent_rows(
    raw_features: pd.DataFrame,
    labels: np.ndarray,
    top_features: int,
) -> list[dict[str, Any]]:
    numeric = raw_features.select_dtypes(include=[np.number]).apply(
        pd.to_numeric,
        errors="coerce",
    )
    if "DAYS_BIRTH" in numeric.columns:
        numeric["AGE_YEARS"] = -numeric["DAYS_BIRTH"] / 365.25
    if "DAYS_EMPLOYED" in numeric.columns:
        numeric["EMPLOYED_YEARS"] = -numeric["DAYS_EMPLOYED"] / 365.25

    if numeric.empty:
        return []

    overall_mean = numeric.mean(numeric_only=True)
    rows: list[dict[str, Any]] = []
    for cluster_label in sorted(set(labels.astype("int64").tolist())):
        mask = labels == cluster_label
        cluster_mean = numeric.loc[mask].mean(numeric_only=True)
        pct_diff = (
            (cluster_mean - overall_mean)
            / overall_mean.abs().replace(0.0, np.nan)
        ) * 100
        ranked = pct_diff.abs().replace([np.inf, -np.inf], np.nan).dropna()
        ranked = ranked.sort_values(ascending=False).head(top_features)
        feature_rows: list[dict[str, Any]] = []
        for feature in ranked.index:
            feature_rows.append(
                {
                    "feature": str(feature),
                    "cluster_mean": float(cluster_mean[feature]),
                    "overall_mean": float(overall_mean[feature]),
                    "percent_difference_from_overall": float(pct_diff[feature]),
                }
            )
        rows.append(
            {
                "cluster": int(cluster_label),
                "count": int(np.sum(mask)),
                "share": float(np.mean(mask)),
                "top_features": feature_rows,
            }
        )
    return rows


def build_top_silhouette_clusterings(
    prepared: PreparedClusteringData,
    results: dict[str, dict[str, Any]],
    top_runs: int = 5,
    top_features: int = 5,
) -> list[dict[str, Any]]:
    candidate_rows: list[dict[str, Any]] = []
    for method, result in results.items():
        candidates = result.get("candidate_results")
        candidate_list = candidates if isinstance(candidates, list) else [result]
        for candidate in candidate_list:
            metrics = candidate.get("metrics", {})
            silhouette = metrics.get("silhouette")
            if not isinstance(silhouette, (int, float)):
                continue
            candidate_rows.append(
                {
                    "method": method,
                    "parameters": candidate.get("parameters", {}),
                    "metrics": metrics,
                    "labels": candidate["labels"],
                }
            )

    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: float(row["metrics"]["silhouette"]),
        reverse=True,
    )[:top_runs]

    output: list[dict[str, Any]] = []
    for rank, row in enumerate(candidate_rows, start=1):
        labels = row["labels"]
        output.append(
            {
                "rank": rank,
                "method": row["method"],
                "parameters": row["parameters"],
                "metrics": row["metrics"],
                "cluster_feature_summary": cluster_feature_percent_rows(
                    prepared.raw_features,
                    labels,
                    top_features,
                ),
            }
        )
    return output


def interpret_cluster_profiles(profile: pd.DataFrame) -> list[dict[str, str]]:
    if profile.empty:
        return []

    rows: list[dict[str, str]] = []
    for (method, cluster), group in profile.groupby(["method", "cluster"], sort=True):
        sorted_index = group["standardized_difference"].abs().sort_values(ascending=False).index
        ranked = group.reindex(sorted_index)
        phrases: list[str] = []
        for _, row in ranked.head(4).iterrows():
            direction = "higher" if float(row["standardized_difference"]) > 0 else "lower"
            phrases.append(f"{direction} {row['feature']}")
        rows.append(
            {
                "method": str(method),
                "cluster": str(cluster),
                "meaning": "; ".join(phrases),
            }
        )
    return rows


def write_plots(
    plots_dir: Path,
    prepared: PreparedClusteringData,
    results: dict[str, dict[str, Any]],
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        logger.warning("matplotlib is not installed; skipping clustering plots.")
        return {}

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}

    if not prepared.pca_diagnostics.empty:
        pca_path = plots_dir / "pca_information_lost.png"
        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = prepared.pca_diagnostics["component"]
        ax1.plot(
            x,
            prepared.pca_diagnostics["cumulative_explained_variance"],
            color="#1f77b4",
            linewidth=2,
            label="Cumulative explained variance",
        )
        ax1.axhline(0.90, color="#d62728", linestyle="--", label="90% target")
        if prepared.pca_n_90 is not None:
            ax1.axvline(prepared.pca_n_90, color="#2ca02c", linestyle=":")
            ax1.scatter([prepared.pca_n_90], [0.90], color="#2ca02c", zorder=4)
        ax1.set_xlabel("Number of PCA components")
        ax1.set_ylabel("Cumulative explained variance")
        ax1.set_ylim(0.0, 1.02)
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            prepared.pca_diagnostics["information_lost"],
            color="#ff7f0e",
            linewidth=1.8,
            alpha=0.85,
            label="Information lost",
        )
        ax2.set_ylabel("Information lost")
        ax2.set_ylim(0.0, 1.02)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        ax1.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(pca_path, dpi=160)
        plt.close(fig)
        plot_paths["pca_information_lost"] = str(pca_path)

    for method, result in results.items():
        labels = result["labels"]
        scatter_path = plots_dir / f"{method}_pca_scatter.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            prepared.plot_matrix[:, 0],
            prepared.plot_matrix[:, 1],
            c=labels,
            cmap="tab20",
            s=14,
            alpha=0.75,
            linewidths=0,
        )
        ax.set_title(f"{method.replace('_', ' ').title()} Clusters on PCA Projection")
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.grid(alpha=0.2)
        fig.colorbar(scatter, ax=ax, label="Cluster label")
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=160)
        plt.close(fig)
        plot_paths[f"{method}_pca_scatter"] = str(scatter_path)

    return plot_paths


def write_markdown_report(
    output_dir: Path,
    report: dict[str, Any],
    interpretations: list[dict[str, str]],
) -> Path:
    pca = report["preprocessing"]
    rows = [
        "# Clustering Report",
        "",
        "## PCA Information Retention",
        "- PCA components closest to or reaching 90% explained variance: "
        f"`{pca.get('pca_n_closest_or_reaching_90pct')}`",
        f"- PCA components used for clustering: `{pca.get('pca_effective_components')}`",
        "- Explained variance kept by chosen PCA: "
        f"`{_safe_score_value(pca.get('pca_explained_variance_ratio_sum'))}`",
        "",
        "## Method Scores",
        "| Method | Cluster n | Noise rate | Silhouette | Calinski-Harabasz | Davies-Bouldin |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, payload in report["methods"].items():
        metrics = payload["metrics"]
        rows.append(
            "| "
            f"{method} | "
            f"{metrics.get('cluster_count')} | "
            f"{_safe_score_value(metrics.get('noise_rate'))} | "
            f"{_safe_score_value(metrics.get('silhouette'))} | "
            f"{_safe_score_value(metrics.get('calinski_harabasz'))} | "
            f"{_safe_score_value(metrics.get('davies_bouldin'))} |"
        )

    best_method = report.get("best_method_by_silhouette")
    rows.extend(
        [
            "",
            "## Best Silhouette Method",
            f"`{best_method}` has the best available silhouette score among the three methods.",
            "",
            "## Cluster Meanings",
            "These meanings are based on the largest standardized differences from "
            "the overall numeric feature means.",
            "",
        ]
    )
    for item in interpretations:
        rows.append(f"- `{item['method']}` cluster `{item['cluster']}`: {item['meaning']}")

    rows.extend(
        [
            "",
            "## Picture Meaning",
            "- PCA scatter plots show applicants projected into two dimensions; "
            "nearby points have similar engineered feature patterns.",
            "- Color is the cluster assignment. HDBSCAN label `-1` means noise/outlier applicants.",
            "- The PCA information-lost plot shows how much original feature variance "
            "is discarded as component count changes.",
        ]
    )
    markdown_path = output_dir / "clustering_report.md"
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return markdown_path


def write_outputs(
    output_dir: Path,
    prepared: PreparedClusteringData,
    results: dict[str, dict[str, Any]],
    top_profile_features: int,
    metadata: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame({ID_COLUMN: prepared.ids.to_numpy(dtype="int64")})
    if prepared.target is not None:
        labels[TARGET_COLUMN] = prepared.target.to_numpy(dtype="int32")

    report_methods: dict[str, Any] = {}
    profile_frames: list[pd.DataFrame] = []
    for method, result in results.items():
        method_labels = result["labels"]
        labels[f"{method}_cluster"] = method_labels
        profile = profile_numeric_features(
            prepared.raw_features,
            method_labels,
            top_profile_features,
        )
        if not profile.empty:
            profile.insert(0, "method", method)
            profile_frames.append(profile)
        report_methods[method] = {
            "parameters": result["parameters"],
            "metrics": result["metrics"],
            "cluster_sizes": cluster_size_rows(method_labels, prepared.target),
        }

    labels_path = output_dir / "cluster_labels.csv"
    profiles_path = output_dir / "cluster_profiles.csv"
    pca_path = output_dir / "pca_diagnostics.csv"
    report_path = output_dir / "clustering_report.json"

    labels.to_csv(labels_path, index=False)
    prepared.pca_diagnostics.to_csv(pca_path, index=False)
    if profile_frames:
        profile = pd.concat(profile_frames, ignore_index=True)
        profile.to_csv(profiles_path, index=False)
    else:
        profile = pd.DataFrame()
        profile.to_csv(profiles_path, index=False)

    interpretations = interpret_cluster_profiles(profile)
    plots = write_plots(output_dir / "plots", prepared, results)

    silhouette_scores = {
        method: payload["metrics"].get("silhouette")
        for method, payload in report_methods.items()
    }
    valid_silhouettes = {
        method: score
        for method, score in silhouette_scores.items()
        if isinstance(score, (int, float))
    }
    best_method = max(valid_silhouettes, key=valid_silhouettes.get) if valid_silhouettes else None

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "metadata": metadata,
        "rows": int(len(prepared.ids)),
        "input_feature_count": int(len(prepared.feature_columns)),
        "categorical_feature_count": int(len(prepared.categorical_columns)),
        "numeric_feature_count": int(len(prepared.numeric_columns)),
        "preprocessing": prepared.preprocessing,
        "best_method_by_silhouette": best_method,
        "silhouette_scores": silhouette_scores,
        "cluster_interpretations": interpretations,
        "methods": report_methods,
        "artifacts": {
            "labels": str(labels_path),
            "profiles": str(profiles_path),
            "pca_diagnostics": str(pca_path),
            "plots": plots,
        },
    }
    report_path.write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    markdown_path = write_markdown_report(output_dir, report, interpretations)
    report["artifacts"]["markdown_report"] = str(markdown_path)
    report_path.write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster Home Credit applicants using engineered features."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("homecreditdefaultriskdata"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/clustering"))
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=25)
    parser.add_argument("--cluster-n-min", type=int, default=2)
    parser.add_argument("--cluster-n-max", type=int, default=10)
    parser.add_argument(
        "--gmm-covariance-type",
        choices=["full", "tied", "diag", "spherical"],
        default="diag",
    )
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=100)
    parser.add_argument("--hdbscan-min-samples", type=int, default=None)
    parser.add_argument("--silhouette-sample-size", type=int, default=5000)
    parser.add_argument("--top-profile-features", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    if args.cluster_n_min < 2:
        raise ValueError("--cluster-n-min must be >= 2.")
    if args.cluster_n_max < args.cluster_n_min:
        raise ValueError("--cluster-n-max must be >= --cluster-n-min.")
    if args.hdbscan_min_cluster_size < 2:
        raise ValueError("--hdbscan-min-cluster-size must be >= 2.")

    logger.info("Building engineered training frame from %s", args.data_dir)
    training_frame = build_training_frame(
        args.data_dir,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    prepared = prepare_clustering_data(
        training_frame,
        random_state=args.random_state,
        pca_components=args.pca_components,
    )

    results = {
        "kmeans": run_kmeans_scan(
            prepared.matrix,
            args.cluster_n_min,
            args.cluster_n_max,
            args.random_state,
            args.silhouette_sample_size,
        ),
        "hdbscan": run_hdbscan(
            prepared.matrix,
            args.hdbscan_min_cluster_size,
            args.hdbscan_min_samples,
            args.random_state,
            args.silhouette_sample_size,
        ),
        "gaussian_mixture": run_gaussian_mixture_scan(
            prepared.matrix,
            args.cluster_n_min,
            args.cluster_n_max,
            args.gmm_covariance_type,
            args.random_state,
            args.silhouette_sample_size,
        ),
    }
    report_path = write_outputs(
        output_dir=args.artifact_dir,
        prepared=prepared,
        results=results,
        top_profile_features=args.top_profile_features,
        metadata={
            "sample_size": args.sample_size,
            "random_state": args.random_state,
            "source": "build_training_frame",
        },
    )
    logger.info("Clustering report saved to %s", report_path)
    print(report_path)


if __name__ == "__main__":
    main()

