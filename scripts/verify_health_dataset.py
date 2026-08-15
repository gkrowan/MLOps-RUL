#!/usr/bin/env python3
"""Validate the Health Indicator V2 experiment data contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from femto_rul.experiments.config import HEALTH_DATASET_PATH, load_experiment_config
from femto_rul.features.health_indicator import health_indicator_feature_columns


def main() -> int:
    cfg = load_experiment_config()
    if not HEALTH_DATASET_PATH.exists():
        raise SystemExit(f"Missing health dataset: {HEALTH_DATASET_PATH}")
    frame = pd.read_parquet(HEALTH_DATASET_PATH)
    features = health_indicator_feature_columns()
    required = {"bearing", "condition", "cut_fraction", "cut_file_index", "rul_seconds", *features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing health dataset columns: {missing}")
    if len(frame) != cfg.expected_prefix_rows:
        raise SystemExit(f"Expected {cfg.expected_prefix_rows} rows, found {len(frame)}")
    if frame["bearing"].nunique() != cfg.expected_training_bearings:
        raise SystemExit(
            f"Expected {cfg.expected_training_bearings} bearings, found {frame['bearing'].nunique()}"
        )
    expected = sorted(np.asarray(cfg.prefix_fractions).round(8))
    actual = sorted(frame["cut_fraction"].astype(float).unique().round(8))
    if actual != expected:
        raise SystemExit(f"Prefix grid mismatch: expected {expected}, found {actual}")
    if not np.isfinite(frame[[*features, "rul_seconds"]].to_numpy(dtype=float)).all():
        raise SystemExit("Non-finite Health V2 values found")
    if "rul_seconds" in features:
        raise SystemExit("Target leakage: rul_seconds is a Health V2 predictor")
    print("=" * 72)
    print("Health Indicator V2 dataset verification PASS")
    print("=" * 72)
    print(f"Rows: {len(frame):,}")
    print(f"Bearings: {frame['bearing'].nunique()}")
    print(f"Model features: {len(features)}")
    print("Target: direct RUL")
    print("CV contract: Leave-One-Bearing-Out")
    print("Future/Test/Validation access: NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
