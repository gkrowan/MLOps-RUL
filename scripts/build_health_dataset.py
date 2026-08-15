#!/usr/bin/env python3
"""Build the DVC-versioned Health Indicator V2 prefix dataset."""

from __future__ import annotations

import pandas as pd

from femto_rul.config import TRAIN_FEATURES_PATH
from femto_rul.experiments.config import HEALTH_DATASET_PATH, load_experiment_config, load_raw_params
from femto_rul.features.health_indicator import build_health_indicator_samples


def main() -> int:
    cfg = load_experiment_config()
    raw = load_raw_params()
    hcfg = raw.get("health_indicator", {})
    train = pd.read_parquet(TRAIN_FEATURES_PATH)
    health = build_health_indicator_samples(
        train,
        fractions=cfg.prefix_fractions,
        healthy_window=int(hcfg.get("healthy_window_snapshots", 60)),
        recent_window=int(hcfg.get("recent_window_snapshots", 60)),
        robust_z_clip=float(hcfg.get("robust_z_clip", 50.0)),
    )
    HEALTH_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    health.to_parquet(HEALTH_DATASET_PATH, index=False)
    print(f"Wrote {HEALTH_DATASET_PATH} ({len(health):,} rows x {health.shape[1]} columns)")
    print(f"Bearings: {health['bearing'].nunique()}")
    print("Representation: health_v2")
    print("Prefix grid: " + ", ".join(f"{int(v * 100)}%" for v in cfg.prefix_fractions))
    print("Test_set / Validation_Set access: NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
