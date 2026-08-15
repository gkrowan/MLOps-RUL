"""Causal degradation-trend features for FEMTO RUL Feature Set V2.

All temporal features use only the current snapshot and earlier snapshots from
that same bearing. No future rows, RUL labels, elapsed-time predictor, Test
labels, or Validation_Set data are used.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from femto_rul.config import FILE_INTERVAL_SECONDS

# Keep V2 intentionally compact: these six health indicators are the most
# interpretable time-domain degradation signals from V1.
TEMPORAL_SOURCE_FEATURES: Final[list[str]] = [
    "rms_horiz",
    "rms_vert",
    "kurtosis_horiz",
    "kurtosis_vert",
    "crest_factor_horiz",
    "crest_factor_vert",
]

# One vibration snapshot is captured about every 10 seconds.
# 6 / 30 / 60 snapshots ~= 1 / 5 / 10 minutes of causal history.
TEMPORAL_WINDOWS: Final[tuple[int, ...]] = (6, 30, 60)
TEMPORAL_STATS: Final[tuple[str, ...]] = ("mean", "std", "slope")


def temporal_feature_columns() -> list[str]:
    """Return Feature Set V2 temporal columns in deterministic order."""
    return [
        f"{source}_roll_{stat}_{window}"
        for source in TEMPORAL_SOURCE_FEATURES
        for window in TEMPORAL_WINDOWS
        for stat in TEMPORAL_STATS
    ]


def _causal_slope(values: np.ndarray) -> float:
    """Least-squares slope versus time, in feature units per second."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0

    x = np.arange(values.size, dtype=float) * float(FILE_INTERVAL_SECONDS)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0.0:
        return 0.0

    y_centered = values - values.mean()
    return float(np.dot(x_centered, y_centered) / denominator)


def add_causal_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean/std/slope features without looking into the future.

    The input should contain one bearing ordered by ``file_index``. The
    function sorts defensively, applies right-aligned rolling windows, and
    restores a clean integer index. Leading rows use the history available so
    far; undefined single-row std/slope values are filled with zero.
    """
    required = {"file_index", *TEMPORAL_SOURCE_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"temporal feature input missing columns: {missing}")

    result = frame.sort_values("file_index").reset_index(drop=True).copy()

    if result["file_index"].duplicated().any():
        raise ValueError("temporal feature input contains duplicate file_index values")

    for source in TEMPORAL_SOURCE_FEATURES:
        series = result[source].astype(float)

        for window in TEMPORAL_WINDOWS:
            rolling = series.rolling(window=window, min_periods=1)
            result[f"{source}_roll_mean_{window}"] = rolling.mean()
            result[f"{source}_roll_std_{window}"] = (
                rolling.std(ddof=0).fillna(0.0)
            )
            result[f"{source}_roll_slope_{window}"] = (
                rolling.apply(_causal_slope, raw=True).fillna(0.0)
            )

    temporal_cols = temporal_feature_columns()
    values = result[temporal_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("temporal feature generation produced non-finite values")

    return result
