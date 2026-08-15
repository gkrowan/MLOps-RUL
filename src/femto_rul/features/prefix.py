"""Prefix-level features for challenge-aligned FEMTO RUL evaluation.

A prefix sample represents one bearing observed only up to a truncation point.
Features use the visible prefix only. The target RUL is used only for Training_set
pseudo-test samples during model development.
"""

from __future__ import annotations

from typing import Final, Iterable

import numpy as np
import pandas as pd

from femto_rul.config import CONDITIONS, FILE_INTERVAL_SECONDS

PREFIX_SOURCE_FEATURES: Final[list[str]] = [
    "rms_horiz",
    "rms_vert",
    "kurtosis_horiz",
    "kurtosis_vert",
    "crest_factor_horiz",
    "crest_factor_vert",
]

DEFAULT_PREFIX_FRACTIONS: Final[tuple[float, ...]] = (0.55, 0.65, 0.75, 0.85, 0.95)
EARLY_WINDOW: Final[int] = 60
RECENT_WINDOW: Final[int] = 60


def _slope_per_hour(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    x_hours = np.arange(values.size, dtype=float) * float(FILE_INTERVAL_SECONDS) / 3600.0
    x = x_hours - x_hours.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    y = values - values.mean()
    return float(np.dot(x, y) / denom)


def prefix_feature_columns() -> list[str]:
    cols = [
        "observed_age_seconds",
        "rotation_speed_rpm",
        "radial_load_n",
    ]
    for source in PREFIX_SOURCE_FEATURES:
        cols.extend(
            [
                f"{source}_current_over_early",
                f"{source}_recent_mean_over_early",
                f"{source}_recent_slope_per_hour",
            ]
        )
    return cols


def _physical_context(condition: int) -> tuple[float, float]:
    values = CONDITIONS.get(int(condition))
    if values is None:
        raise ValueError(f"unknown operating condition: {condition}")
    return float(values["rotation_speed_rpm"]), float(values["radial_load_n"])


def _one_prefix_row(
    bearing_frame: pd.DataFrame,
    cut_position: int,
    *,
    fraction: float,
) -> dict[str, object]:
    ordered = bearing_frame.sort_values("file_index").reset_index(drop=True)
    prefix = ordered.iloc[: cut_position + 1]
    endpoint = prefix.iloc[-1]

    condition = int(endpoint["condition"])
    bearing = str(endpoint["bearing"])
    speed, load = _physical_context(condition)

    result: dict[str, object] = {
        "condition": condition,
        "bearing": bearing,
        "cut_fraction": float(fraction),
        "cut_file_index": int(endpoint["file_index"]),
        "observed_age_seconds": float(endpoint["file_index"]) * float(FILE_INTERVAL_SECONDS),
        "rotation_speed_rpm": speed,
        "radial_load_n": load,
        "rul_seconds": float(endpoint["rul_seconds"]),
    }

    for source in PREFIX_SOURCE_FEATURES:
        values = prefix[source].to_numpy(dtype=float)
        early = values[: min(EARLY_WINDOW, len(values))]
        recent = values[-min(RECENT_WINDOW, len(values)) :]

        early_level = float(np.median(early))
        scale = max(abs(early_level), 1e-8)
        result[f"{source}_current_over_early"] = float(values[-1] / scale)
        result[f"{source}_recent_mean_over_early"] = float(np.mean(recent) / scale)
        result[f"{source}_recent_slope_per_hour"] = _slope_per_hour(recent)

    return result


def build_prefix_training_samples(
    train_frame: pd.DataFrame,
    *,
    fractions: Iterable[float] = DEFAULT_PREFIX_FRACTIONS,
) -> pd.DataFrame:
    """Create pseudo-test endpoint samples from Training_set trajectories.

    Cut fractions are fixed experiment design points and are never predictors.
    Each feature row sees only samples at or before its cut point.
    """
    required = {
        "condition",
        "bearing",
        "file_index",
        "rul_seconds",
        *PREFIX_SOURCE_FEATURES,
    }
    missing = sorted(required - set(train_frame.columns))
    if missing:
        raise ValueError(f"prefix training input missing columns: {missing}")

    fractions = tuple(float(v) for v in fractions)
    if not fractions or any(v <= 0.0 or v >= 1.0 for v in fractions):
        raise ValueError("prefix fractions must be between 0 and 1")

    rows: list[dict[str, object]] = []
    for bearing, bearing_frame in train_frame.groupby("bearing", sort=True):
        ordered = bearing_frame.sort_values("file_index").reset_index(drop=True)
        if len(ordered) < 4:
            raise ValueError(f"bearing {bearing} has too few snapshots")

        seen_positions: set[int] = set()
        for fraction in fractions:
            cut_position = int(round((len(ordered) - 1) * fraction))
            cut_position = min(max(cut_position, 1), len(ordered) - 2)
            if cut_position in seen_positions:
                continue
            seen_positions.add(cut_position)
            rows.append(_one_prefix_row(ordered, cut_position, fraction=fraction))

    result = pd.DataFrame(rows)
    features = prefix_feature_columns()
    numeric = result[[*features, "rul_seconds"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("prefix feature generation produced non-finite values")
    return result.sort_values(["bearing", "cut_file_index"]).reset_index(drop=True)
