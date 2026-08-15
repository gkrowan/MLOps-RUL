"""One canonical experiment runner for all direct-RUL model comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from femto_rul.config import ARTIFACTS_DIR
from femto_rul.evaluation.prefix_validation import (
    monotonicity_summary,
    prefix_lobo_cv,
    summarize_prefix_cv,
)
from femto_rul.experiments.config import PREFIX_DATASET_PATH, ExperimentConfig, load_experiment_config
from femto_rul.experiments.models import effective_model_params, make_model
from femto_rul.experiments.tracking import configure_mlflow, reproducibility_tags
from femto_rul.features.prefix import prefix_feature_columns


@dataclass
class ExperimentResult:
    experiment_id: str
    model_name: str
    summary: dict[str, float | int | str]
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    output_dir: Path
    mlflow_run_id: str | None = None


def _validate_dataset(frame: pd.DataFrame, cfg: ExperimentConfig) -> None:
    required = {
        "condition",
        "bearing",
        "cut_fraction",
        "cut_file_index",
        "observed_age_seconds",
        "rul_seconds",
        *prefix_feature_columns(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"canonical prefix dataset missing columns: {missing}")
    if frame["bearing"].nunique() != cfg.expected_training_bearings:
        raise ValueError(
            f"expected {cfg.expected_training_bearings} training bearings, "
            f"found {frame['bearing'].nunique()}"
        )
    if len(frame) != cfg.expected_prefix_rows:
        raise ValueError(f"expected {cfg.expected_prefix_rows} prefix rows, found {len(frame)}")
    observed_fractions = sorted(frame["cut_fraction"].astype(float).unique().round(8))
    expected_fractions = sorted(np.asarray(cfg.prefix_fractions).round(8))
    if observed_fractions != expected_fractions:
        raise ValueError(
            f"prefix grid mismatch: expected {expected_fractions}, got {observed_fractions}"
        )


def _design_matrix(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if model_name == "condition_life_prior":
        return frame[["condition", "observed_age_seconds"]].copy()
    return frame[prefix_feature_columns()].copy()


def _save_plots(folds: pd.DataFrame, predictions: pd.DataFrame, out: Path) -> None:
    order = folds.sort_values("held_out_bearing")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(order["held_out_bearing"], order["rmse"])
    ax.set_title("LOBO RMSE by held-out bearing")
    ax.set_ylabel("RMSE (seconds)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out / "rmse_by_bearing.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for bearing, frame in predictions.groupby("held_out_bearing", sort=True):
        ordered = frame.sort_values("observed_age_seconds")
        ax.plot(
            ordered["observed_age_seconds"],
            ordered["prediction_rul_seconds"],
            marker="o",
            label=str(bearing),
        )
    ax.set_title("Predicted RUL across observed bearing prefixes")
    ax.set_xlabel("Observed age (seconds)")
    ax.set_ylabel("Predicted RUL (seconds)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "prediction_trajectories.png", dpi=160)
    plt.close(fig)


def _unwrap_model(model: Any) -> Any:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def _feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    inner = _unwrap_model(model)
    if hasattr(inner, "feature_importances_"):
        values = np.asarray(inner.feature_importances_, dtype=float)
    elif hasattr(inner, "coef_"):
        values = np.abs(np.asarray(inner.coef_, dtype=float).reshape(-1))
    else:
        return None
    if len(values) != len(feature_names):
        return None
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_experiment(
    experiment_id: str,
    *,
    track_mlflow: bool = True,
    model_overrides: dict[str, Any] | None = None,
    config: ExperimentConfig | None = None,
    prefix_frame: pd.DataFrame | None = None,
) -> ExperimentResult:
    cfg = config or load_experiment_config()
    spec = cfg.model(experiment_id)
    frame = prefix_frame.copy() if prefix_frame is not None else pd.read_parquet(PREFIX_DATASET_PATH)
    _validate_dataset(frame, cfg)

    X = _design_matrix(frame, spec.model_name)
    y = frame["rul_seconds"].astype(float)
    groups = frame["bearing"].astype(str)
    metadata = frame[["cut_fraction", "cut_file_index", "observed_age_seconds"]].copy()

    estimator = make_model(
        spec.model_name,
        defaults=cfg.model_defaults,
        random_state=cfg.random_state,
        overrides=model_overrides,
    )
    folds, predictions = prefix_lobo_cv(
        estimator,
        X,
        y,
        groups,
        model_name=spec.model_name,
        metadata=metadata,
    )
    summary_df = summarize_prefix_cv(folds)
    monotonic = monotonicity_summary(predictions)
    violation_rate = float(monotonic["monotonic_violation_rate"].mean())
    summary_row = summary_df.iloc[0].to_dict()
    summary_row["mean_monotonic_violation_rate"] = violation_rate
    summary_row["experiment_id"] = experiment_id
    summary_row["benchmark_version"] = cfg.benchmark_version

    out = (ARTIFACTS_DIR / "modeling" / "experiments" / experiment_id).resolve()
    out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    monotonic.to_csv(out / "monotonicity_by_bearing.csv", index=False)
    pd.DataFrame([summary_row]).to_csv(out / "summary.csv", index=False)
    _save_plots(folds, predictions, out)

    fitted = clone(estimator).fit(X, y)
    importance = _feature_importance(fitted, list(X.columns))
    if importance is not None:
        importance.to_csv(out / "feature_importance.csv", index=False)

    manifest = {
        "experiment_id": experiment_id,
        "model_name": spec.model_name,
        "description": spec.description,
        "benchmark_version": cfg.benchmark_version,
        "prefix_fractions": list(cfg.prefix_fractions),
        "model_params": effective_model_params(spec.model_name, cfg.model_defaults, model_overrides),
        "training_bearings": int(groups.nunique()),
        "prefix_rows": int(len(frame)),
        "feature_count": int(X.shape[1]),
        "test_accessed": False,
        "validation_accessed": False,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    run_id: str | None = None
    if track_mlflow:
        mlflow = configure_mlflow(cfg.mlflow_experiment_name)
        import mlflow.sklearn
        from mlflow.models import infer_signature

        tags = reproducibility_tags(
            benchmark_version=cfg.benchmark_version,
            experiment_id=experiment_id,
            model_name=spec.model_name,
        )
        with mlflow.start_run(run_name=f"{experiment_id}-{spec.model_name}", tags=tags) as run:
            run_id = run.info.run_id
            mlflow.log_params(
                {
                    "experiment_id": experiment_id,
                    "model_name": spec.model_name,
                    "benchmark_version": cfg.benchmark_version,
                    "prefix_rows": len(frame),
                    "training_bearings": groups.nunique(),
                    "feature_count": X.shape[1],
                    "prefix_grid": ",".join(f"{v:.2f}" for v in cfg.prefix_fractions),
                    **{
                        f"model__{k}": v
                        for k, v in effective_model_params(
                            spec.model_name, cfg.model_defaults, model_overrides
                        ).items()
                    },
                }
            )
            metric_names = [
                "mean_rmse",
                "std_rmse",
                "median_rmse",
                "worst_bearing_rmse",
                "mean_mae",
                "mean_r2",
                "mean_phm12_prefix_score",
                "mean_monotonic_violation_rate",
            ]
            for name in metric_names:
                if name in summary_row and pd.notna(summary_row[name]):
                    mlflow.log_metric(name, float(summary_row[name]))
            for _, row in folds.iterrows():
                bearing = str(row["held_out_bearing"]).replace("/", "_")
                mlflow.log_metric(f"bearing_rmse__{bearing}", float(row["rmse"]))
                mlflow.log_metric(f"bearing_mae__{bearing}", float(row["mae"]))
            mlflow.log_artifacts(str(out), artifact_path="evaluation")
            signature = infer_signature(X, fitted.predict(X))
            mlflow.sklearn.log_model(
                fitted,
                name="model",
                signature=signature,
                input_example=X.head(min(5, len(X))),
            )

    return ExperimentResult(
        experiment_id=experiment_id,
        model_name=spec.model_name,
        summary=summary_row,
        fold_metrics=folds,
        predictions=predictions,
        output_dir=out,
        mlflow_run_id=run_id,
    )
