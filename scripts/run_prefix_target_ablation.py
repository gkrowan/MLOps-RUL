#!/usr/bin/env python3
"""Expanded prefix experiment: direct RUL vs total-life target.

Uses Training_set only and Leave-One-Bearing-Out validation. Validation_Set and
Test_set are not read. This phase tests target formulation before any Optuna/HPO.
"""

from __future__ import annotations

import pandas as pd

from femto_rul.config import ARTIFACTS_DIR, TRAIN_FEATURES_PATH
from femto_rul.evaluation.prefix_validation import (
    monotonicity_summary,
    prefix_lobo_cv,
    summarize_prefix_cv,
)
from femto_rul.features.prefix import (
    DEFAULT_PREFIX_FRACTIONS,
    build_prefix_training_samples,
    prefix_feature_columns,
)
from femto_rul.models.prefix_models import prefix_target_estimators


def main() -> int:
    train = pd.read_parquet(TRAIN_FEATURES_PATH)
    prefix = build_prefix_training_samples(train)
    feature_cols = prefix_feature_columns()
    y = prefix["rul_seconds"].astype(float)
    groups = prefix["bearing"].astype(str)
    metadata = prefix[
        [
            "cut_fraction",
            "cut_file_index",
            "observed_age_seconds",
            "total_life_seconds",
        ]
    ].copy()

    print("=" * 92)
    print("FEMTO RUL target ablation — expanded pseudo-prefixes, Training_set only")
    print("=" * 92)
    print(f"Complete training bearings: {groups.nunique()}")
    print(f"Pseudo-test prefixes: {len(prefix)}")
    print(
        "Cut fractions: "
        + ", ".join(f"{int(v * 100)}%" for v in DEFAULT_PREFIX_FRACTIONS)
    )
    print(f"Compact predictors: {len(feature_cols)}")
    print("CV: Leave-One-Bearing-Out (6 folds)")
    print("Test_set / Validation_Set access: NONE")
    print("Goal: compare direct-RUL prediction with total-life -> RUL formulation")

    metrics_all: list[pd.DataFrame] = []
    preds_all: list[pd.DataFrame] = []

    for name, estimator in prefix_target_estimators().items():
        if name == "condition_life_prior":
            X = prefix[["condition", "observed_age_seconds"]].copy()
        else:
            X = prefix[feature_cols].copy()

        print(f"\nRunning {name} ({X.shape[1]} predictors)...")
        metrics, predictions = prefix_lobo_cv(
            estimator,
            X,
            y,
            groups,
            model_name=name,
            metadata=metadata,
        )
        print(
            f"  mean RMSE: {metrics['rmse'].mean():,.2f} "
            f"± {metrics['rmse'].std():,.2f} sec | "
            f"worst bearing: {metrics['rmse'].max():,.2f} sec"
        )
        metrics_all.append(metrics)
        preds_all.append(predictions)

    fold_metrics = pd.concat(metrics_all, ignore_index=True)
    predictions = pd.concat(preds_all, ignore_index=True)
    summary = summarize_prefix_cv(fold_metrics)

    monotonic = monotonicity_summary(predictions)
    model_monotonic = (
        monotonic.groupby("model", as_index=False)
        .agg(mean_monotonic_violation_rate=("monotonic_violation_rate", "mean"))
    )
    summary = summary.merge(model_monotonic, on="model", how="left")

    out = (ARTIFACTS_DIR / "modeling" / "prefix_rul_v2_target").resolve()
    out.mkdir(parents=True, exist_ok=True)
    prefix.to_parquet(out / "pseudo_test_prefixes.parquet", index=False)
    fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    monotonic.to_csv(out / "monotonicity_by_bearing.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    print("\n" + "=" * 92)
    print("Target ablation summary — lower RMSE/MAE/violation rate is better")
    print("=" * 92)
    print(
        summary[
            [
                "model",
                "mean_rmse",
                "std_rmse",
                "median_rmse",
                "worst_bearing_rmse",
                "mean_mae",
                "mean_monotonic_violation_rate",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    print("\nRMSE by held-out bearing:")
    print(
        fold_metrics.pivot(
            index="held_out_bearing", columns="model", values="rmse"
        ).round(1).to_string()
    )
    print(f"\nArtifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
