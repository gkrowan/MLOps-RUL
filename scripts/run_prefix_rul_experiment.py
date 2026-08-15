#!/usr/bin/env python3
"""Challenge-aligned pseudo-test RUL experiment using Training_set only.

Instead of treating every vibration snapshot as an independent RUL example,
this script simulates truncated bearing prefixes on each of the six complete
Training_set runs and evaluates them with Leave-One-Bearing-Out CV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from femto_rul.config import ARTIFACTS_DIR, TRAIN_FEATURES_PATH
from femto_rul.evaluation.prefix_validation import prefix_lobo_cv, summarize_prefix_cv
from femto_rul.features.prefix import build_prefix_training_samples, prefix_feature_columns
from femto_rul.models.prefix_models import prefix_estimators


def main() -> int:
    train = pd.read_parquet(TRAIN_FEATURES_PATH)
    prefix = build_prefix_training_samples(train)
    feature_cols = prefix_feature_columns()

    y = prefix["rul_seconds"].astype(float)
    groups = prefix["bearing"].astype(str)

    print("=" * 88)
    print("FEMTO RUL prefix experiment — pseudo-test truncations, Training_set only")
    print("=" * 88)
    print(f"Complete training bearings: {groups.nunique()}")
    print(f"Pseudo-test prefixes: {len(prefix)}")
    print("Cut fractions: 55%, 65%, 75%, 85%, 95%")
    print(f"Compact prefix predictors: {len(feature_cols)}")
    print("Observed age: ALLOWED (known at prediction time)")
    print("Test_set / Validation_Set access: NONE")

    metrics_all: list[pd.DataFrame] = []
    preds_all: list[pd.DataFrame] = []

    for name, estimator in prefix_estimators().items():
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
        )
        print(
            f"  mean RMSE: {metrics['rmse'].mean():,.2f} "
            f"± {metrics['rmse'].std():,.2f} sec"
        )
        metrics_all.append(metrics)
        preds_all.append(predictions)

    fold_metrics = pd.concat(metrics_all, ignore_index=True)
    predictions = pd.concat(preds_all, ignore_index=True)
    summary = summarize_prefix_cv(fold_metrics)

    out = (ARTIFACTS_DIR / "modeling" / "prefix_rul_v1").resolve()
    out.mkdir(parents=True, exist_ok=True)
    prefix.to_parquet(out / "pseudo_test_prefixes.parquet", index=False)
    fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    print("\n" + "=" * 88)
    print("Prefix experiment summary — lower RMSE/MAE is better")
    print("=" * 88)
    print(
        summary[
            [
                "model",
                "mean_rmse",
                "std_rmse",
                "mean_mae",
                "mean_r2",
                "mean_phm12_prefix_score",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print(f"\nArtifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
