"""Loads the training-time reference distribution for drift monitoring
(Phase 17). Reference = Phase 5/6's train_features.parquet, reduced to
exactly the 24 Feature Set V1 columns — no rul_seconds, no
elapsed_time_seconds/file_index/bearing/split (a live prediction never
carries those, so comparing against them would misrepresent drift)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from femto_rul import config
from femto_rul.pipeline import FEATURE_COLUMNS_V1


def load_reference_features(path: Path | None = None) -> pd.DataFrame:
    path = path or config.TRAIN_FEATURES_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Phase 17 needs Phase 5/6's "
            "train_features.parquet as the monitoring reference distribution "
            "— it hasn't been generated yet (see docs/phase_17_evidently_monitoring.md)."
        )

    df = pd.read_parquet(path)

    missing = set(FEATURE_COLUMNS_V1) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing expected Feature Set V1 columns: {sorted(missing)}"
        )

    return df[FEATURE_COLUMNS_V1].copy()


def load_reference_targets(path: Path | None = None) -> pd.Series:
    """The training rul_seconds label — used only as a rough sanity range
    for production predictions in report.py, not as an Evidently target
    column (a live prediction is not the same quantity as a true label;
    see report.py's docstring)."""
    path = path or config.TRAIN_FEATURES_PATH

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist yet.")

    df = pd.read_parquet(path, columns=["rul_seconds"])
    return df["rul_seconds"]
