#!/usr/bin/env python3
"""Compare V1 snapshots, operating context, and V2 causal trend features.

This experiment reads Training_set-derived ``train_features.parquet`` only.
It never reads Test_set features or Validation_Set / ground-truth artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from femto_rul.config import ARTIFACTS_DIR, CONDITIONS, FEATURE_SCHEMA_PATH, TRAIN_FEATURES_PATH
from femto_rul.evaluation.cross_validation import leave_one_bearing_out_cv, summarize_cv_metrics
from femto_rul.models.baselines import MedianRULRegressor


def _add_operating_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["rotation_speed_rpm"] = result["condition"].map(
        {condition: values["rotation_speed_rpm"] for condition, values in CONDITIONS.items()}
    )
    result["radial_load_n"] = result["condition"].map(
        {condition: values["radial_load_n"] for condition, values in CONDITIONS.items()}
    )
    if result[["rotation_speed_rpm", "radial_load_n"]].isna().any().any():
        raise ValueError("unknown operating condition while deriving physical context")
    return result


def main() -> int:
    train = pd.read_parquet(TRAIN_FEATURES_PATH)
    schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
    train = _add_operating_context(train)

    if schema.get("feature_set_version") != "v2":
        raise SystemExit(
            "Feature Set V2 is required. Run `dvc repro` before this ablation."
        )

    base_features = list(schema["signal_feature_columns"])
    temporal_features = list(schema["default_model_feature_columns"])
    context_features = ["rotation_speed_rpm", "radial_load_n"]

    y = train[schema["target_column"]].astype(float)
    groups = train["bearing"].astype(str)

    experiments = {
        "median": (MedianRULRegressor(), base_features),
        "hgb_v1_snapshot": (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                l2_regularization=1.0,
                random_state=42,
            ),
            base_features,
        ),
        "hgb_v1_plus_context": (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                l2_regularization=1.0,
                random_state=42,
            ),
            [*base_features, *context_features],
        ),
        "hgb_v2_temporal": (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                l2_regularization=1.0,
                random_state=42,
            ),
            temporal_features,
        ),
        "hgb_v2_temporal_plus_context": (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                l2_regularization=1.0,
                random_state=42,
            ),
            [*temporal_features, *context_features],
        ),
    }

    blocked = set(schema["blocked_predictor_columns"])
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []

    print("=" * 80)
    print("FEMTO RUL feature ablation — Training_set only, LOBO CV")
    print("=" * 80)
    print(f"Rows: {len(train):,}")
    print(f"Bearings: {groups.nunique()}")
    print(f"V1 snapshot features: {len(base_features)}")
    print(f"V2 default features: {len(temporal_features)}")
    print("Physical context: rotation_speed_rpm + radial_load_n")
    print("Holdout access: NONE")

    for name, (estimator, feature_columns) in experiments.items():
        leaked = blocked.intersection(feature_columns)
        if leaked:
            raise AssertionError(f"{name}: blocked predictors selected: {sorted(leaked)}")

        X = train[feature_columns].copy()
        print(f"\nRunning {name} ({X.shape[1]} features)...")
        fold_metrics, predictions = leave_one_bearing_out_cv(
            estimator,
            X,
            y,
            groups,
            model_name=name,
            clip_nonnegative=True,
        )
        print(
            f"  mean RMSE: {fold_metrics['rmse'].mean():,.2f} "
            f"± {fold_metrics['rmse'].std():,.2f} sec"
        )
        all_metrics.append(fold_metrics)
        all_predictions.append(predictions)

    fold_metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_cv_metrics(fold_metrics)

    output_dir = (ARTIFACTS_DIR / "modeling" / "feature_ablation_v2").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    print("\n" + "=" * 80)
    print("Feature ablation summary — lower RMSE is better")
    print("=" * 80)
    print(
        summary[
            ["model", "mean_rmse", "std_rmse", "mean_mae", "mean_r2"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nArtifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
