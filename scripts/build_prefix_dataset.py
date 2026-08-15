#!/usr/bin/env python3
"""Build the canonical DVC-versioned pseudo-prefix training dataset."""

from __future__ import annotations

import pandas as pd

from femto_rul.config import TRAIN_FEATURES_PATH
from femto_rul.experiments.config import PREFIX_DATASET_PATH, load_experiment_config
from femto_rul.features.prefix import build_prefix_training_samples


def main() -> int:
    cfg = load_experiment_config()
    train = pd.read_parquet(TRAIN_FEATURES_PATH)
    prefix = build_prefix_training_samples(train, fractions=cfg.prefix_fractions)
    PREFIX_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    prefix.to_parquet(PREFIX_DATASET_PATH, index=False)
    print(f"Wrote {PREFIX_DATASET_PATH} ({len(prefix):,} rows x {prefix.shape[1]} columns)")
    print(f"Bearings: {prefix['bearing'].nunique()}")
    print("Prefix grid: " + ", ".join(f"{int(v * 100)}%" for v in cfg.prefix_fractions))
    print("Test_set / Validation_Set access: NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
