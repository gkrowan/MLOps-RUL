"""Build production-safe FEMTO processed datasets.

Modes are intentionally separable so training and inference feature generation
can run without access to Validation_Set.

Examples:
    python scripts/build_datasets.py --mode train
    python scripts/build_datasets.py --mode test-features
    python scripts/build_datasets.py --mode ground-truth
    python scripts/build_datasets.py --mode all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from femto_rul.config import (
    FEATURE_SCHEMA_PATH,
    TEST_FEATURES_PATH,
    TEST_GROUND_TRUTH_PATH,
    TEST_SET_DIR,
    TRAIN_FEATURES_PATH,
    TRAINING_SET_DIR,
    VALIDATION_SET_DIR,
)
from femto_rul.features.schema import write_feature_schema
from femto_rul.pipeline import (
    build_test_feature_dataset,
    build_test_ground_truth,
    build_training_dataset,
)

MODES = ("all", "train", "test-features", "ground-truth")


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    print(f"Wrote {path} ({len(df):,} rows x {len(df.columns)} columns)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe FEMTO processed datasets."
    )
    parser.add_argument("--mode", choices=MODES, default="all")
    args = parser.parse_args()

    mode = args.mode

    if mode in {"all", "train"}:
        print("Building Training_set features + RUL labels...")
        train = build_training_dataset(TRAINING_SET_DIR)
        if "rul_seconds" not in train.columns:
            raise AssertionError("train_features must contain rul_seconds")
        _write_parquet(train, TRAIN_FEATURES_PATH)

    if mode in {"all", "test-features"}:
        print("Building Test_set inference features (no ground truth access)...")
        test_features = build_test_feature_dataset(TEST_SET_DIR)
        if "rul_seconds" in test_features.columns:
            raise AssertionError("test_features must not contain rul_seconds")
        _write_parquet(test_features, TEST_FEATURES_PATH)

    if mode in {"all", "ground-truth"}:
        print("Building official Test_set ground truth from Validation_Set...")
        ground_truth = build_test_ground_truth(TEST_SET_DIR, VALIDATION_SET_DIR)
        _write_parquet(ground_truth, TEST_GROUND_TRUTH_PATH)

    write_feature_schema(FEATURE_SCHEMA_PATH)
    print(f"Wrote {FEATURE_SCHEMA_PATH}")


if __name__ == "__main__":
    main()
