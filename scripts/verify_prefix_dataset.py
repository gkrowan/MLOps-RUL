#!/usr/bin/env python3
"""Fail-fast checks for the canonical prefix experiment dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from femto_rul.experiments.config import PREFIX_DATASET_PATH, load_experiment_config
from femto_rul.features.prefix import prefix_feature_columns


def main() -> int:
    cfg = load_experiment_config()
    if not PREFIX_DATASET_PATH.exists():
        raise SystemExit(f"Missing prefix dataset: {PREFIX_DATASET_PATH}")
    frame = pd.read_parquet(PREFIX_DATASET_PATH)
    failures: list[str] = []

    if len(frame) != cfg.expected_prefix_rows:
        failures.append(f"rows={len(frame)}, expected={cfg.expected_prefix_rows}")
    if frame["bearing"].nunique() != cfg.expected_training_bearings:
        failures.append(
            f"bearings={frame['bearing'].nunique()}, expected={cfg.expected_training_bearings}"
        )
    observed = sorted(frame["cut_fraction"].astype(float).unique().round(8))
    expected = sorted(np.asarray(cfg.prefix_fractions).round(8))
    if observed != expected:
        failures.append(f"prefix grid mismatch: {observed} != {expected}")

    forbidden_predictors = {"rul_seconds", "total_life_seconds", "cut_fraction", "bearing"}
    overlap = forbidden_predictors & set(prefix_feature_columns())
    if overlap:
        failures.append(f"leakage predictors exposed by prefix_feature_columns(): {sorted(overlap)}")
    if not np.isfinite(frame[prefix_feature_columns() + ["rul_seconds"]].to_numpy(float)).all():
        failures.append("non-finite predictor/target value found")

    if failures:
        print("Prefix dataset verification FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("=" * 70)
    print("Canonical prefix dataset verification PASS")
    print("=" * 70)
    print(f"Benchmark: {cfg.benchmark_version}")
    print(f"Rows: {len(frame)}")
    print(f"Bearings: {frame['bearing'].nunique()}")
    print(f"Predictors: {len(prefix_feature_columns())}")
    print("Target: direct RUL")
    print("CV contract: Leave-One-Bearing-Out")
    print("Test_set / Validation_Set access: NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
