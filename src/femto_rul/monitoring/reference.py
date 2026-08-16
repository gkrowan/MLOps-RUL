"""Loads the training-time reference distribution for drift monitoring
(Phase 17). Reference = the prefix-feature training dataset
(data/processed/prefix_train_v1.parquet, built by
scripts/build_prefix_dataset.py — see experiments/config.py's
PREFIX_DATASET_PATH), reduced to exactly the 21 Prefix V1 columns the
served model actually consumes — no rul_seconds, no
condition/bearing/cut_file_index/cut_fraction/total_life_seconds (a live
prediction never carries those, so comparing against them would
misrepresent drift)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from femto_rul.experiments.config import PREFIX_DATASET_PATH
from femto_rul.features.prefix import prefix_feature_columns


def load_reference_features(path: Path | None = None) -> pd.DataFrame:
    path = path or PREFIX_DATASET_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Phase 17 needs the prefix training "
            "dataset (data/processed/prefix_train_v1.parquet) as the "
            "monitoring reference distribution — it hasn't been generated "
            "yet (see docs/phase_17_evidently_monitoring.md)."
        )

    df = pd.read_parquet(path)

    feature_columns = prefix_feature_columns()
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing expected Prefix V1 columns: {sorted(missing)}"
        )

    return df[feature_columns].copy()


def load_reference_targets(path: Path | None = None) -> pd.Series:
    """The training rul_seconds label — used only as a rough sanity range
    for production predictions in report.py, not as an Evidently target
    column (a live prediction is not the same quantity as a true label;
    see report.py's docstring)."""
    path = path or PREFIX_DATASET_PATH

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist yet.")

    df = pd.read_parquet(path, columns=["rul_seconds"])
    return df["rul_seconds"]
