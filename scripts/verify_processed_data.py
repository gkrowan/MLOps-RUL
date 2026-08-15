"""Validate processed FEMTO dataset boundaries and feature schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from femto_rul.config import (
    FEATURE_SCHEMA_PATH,
    TEST_FEATURES_PATH,
    TEST_GROUND_TRUTH_PATH,
    TRAIN_FEATURES_PATH,
)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing required processed artifact: {path}")


def _assert_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicates = df.duplicated(keys).sum()
    if duplicates:
        raise SystemExit(f"{name}: found {duplicates} duplicate rows for keys {keys}")


def _assert_finite(df: pd.DataFrame, columns: list[str], name: str) -> None:
    values = df[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise SystemExit(f"{name}: non-finite values found in model feature columns")


def main() -> None:
    for path in [
        TRAIN_FEATURES_PATH,
        TEST_FEATURES_PATH,
        TEST_GROUND_TRUTH_PATH,
        FEATURE_SCHEMA_PATH,
    ]:
        _require_file(path)

    train = pd.read_parquet(TRAIN_FEATURES_PATH)
    test = pd.read_parquet(TEST_FEATURES_PATH)
    ground_truth = pd.read_parquet(TEST_GROUND_TRUTH_PATH)
    schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))

    target = schema["target_column"]
    model_features = list(schema["default_model_feature_columns"])
    blocked = set(schema["blocked_predictor_columns"])
    gt_keys = list(schema["ground_truth_key_columns"])

    if train.empty or test.empty or ground_truth.empty:
        raise SystemExit("processed artifacts must not be empty")

    if target not in train.columns:
        raise SystemExit(f"train_features is missing target column {target!r}")

    if target in test.columns:
        raise SystemExit("LEAKAGE: test_features contains rul_seconds")

    if target not in ground_truth.columns:
        raise SystemExit(f"test_ground_truth is missing {target!r}")

    missing_train = sorted(set(model_features) - set(train.columns))
    missing_test = sorted(set(model_features) - set(test.columns))
    if missing_train or missing_test:
        raise SystemExit(
            f"feature schema mismatch: missing_train={missing_train}, missing_test={missing_test}"
        )

    if blocked.intersection(model_features):
        raise SystemExit(
            "feature schema error: blocked leakage/metadata columns appear in default model features"
        )

    _assert_unique(train, ["bearing", "file_index"], "train_features")
    _assert_unique(test, ["bearing", "file_index"], "test_features")
    _assert_unique(ground_truth, ["bearing", "file_index"], "test_ground_truth")

    if set(train["split"].unique()) != {"Training_set"}:
        raise SystemExit("train_features contains a non-Training_set split")

    if set(test["split"].unique()) != {"Test_set"}:
        raise SystemExit("test_features contains a non-Test_set split")

    if set(train["bearing"]).intersection(set(test["bearing"])):
        raise SystemExit("training and official holdout bearing IDs overlap")

    _assert_finite(train, model_features, "train_features")
    _assert_finite(test, model_features, "test_features")

    # Every full-run training bearing should end at RUL=0.
    per_bearing_min = train.groupby("bearing")[target].min()
    if not (per_bearing_min == 0).all():
        raise SystemExit("one or more Training_set bearings do not end at RUL=0")

    # Test_set is truncated, so all official labels should remain above zero.
    if (ground_truth[target] <= 0).any():
        raise SystemExit("test_ground_truth contains non-positive RUL for a truncated Test_set row")

    test_keys = set(map(tuple, test[gt_keys].itertuples(index=False, name=None)))
    truth_keys = set(map(tuple, ground_truth[gt_keys].itertuples(index=False, name=None)))
    if test_keys != truth_keys:
        missing_truth = len(test_keys - truth_keys)
        extra_truth = len(truth_keys - test_keys)
        raise SystemExit(
            "test feature / ground-truth key mismatch: "
            f"missing_truth={missing_truth}, extra_truth={extra_truth}"
        )

    print("=" * 70)
    print("Processed dataset verification PASS")
    print("=" * 70)
    print(f"Feature set: {schema['feature_set_version']}")
    print(f"Train rows: {len(train):,}")
    print(f"Test feature rows: {len(test):,}")
    print(f"Ground-truth rows: {len(ground_truth):,}")
    print(f"Default model features: {len(model_features)}")
    print("Target in train: yes")
    print("Target in test features: no")
    print("Test feature / ground-truth keys: exact match")
    print("Leakage-blocked columns excluded from default model features")


if __name__ == "__main__":
    main()
