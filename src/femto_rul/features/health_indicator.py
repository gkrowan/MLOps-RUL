"""Causal Health-Indicator V2 features for FEMTO bearing prefixes.

The health baseline is estimated from the first ``healthy_window`` snapshots of
THE SAME bearing. For a pseudo-prefix, only observations at or before the cut
point are used. No future vibration, final lifetime, Test_set, or
Validation_Set information is required.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from femto_rul.config import CONDITIONS, FILE_INTERVAL_SECONDS

BASE_SOURCE_FEATURES = [
    "rms_horiz",
    "rms_vert",
    "kurtosis_horiz",
    "kurtosis_vert",
    "crest_factor_horiz",
    "crest_factor_vert",
]
FFT_HORIZ = [f"fft_band_{i}_horiz" for i in range(8)]
FFT_VERT = [f"fft_band_{i}_vert" for i in range(8)]
HEALTH_SIGNAL_NAMES = [*BASE_SOURCE_FEATURES, "fft_total_horiz", "fft_total_vert"]


def _slope_per_hour(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=float) * float(FILE_INTERVAL_SECONDS) / 3600.0
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    y = values - values.mean()
    return float(np.dot(x, y) / denom)


def _physical_context(condition: int) -> tuple[float, float]:
    values = CONDITIONS.get(int(condition))
    if values is None:
        raise ValueError(f"unknown operating condition: {condition}")
    return float(values["rotation_speed_rpm"]), float(values["radial_load_n"])


def health_indicator_feature_columns() -> list[str]:
    """Model predictors for the Health Indicator V2 representation."""
    cols = [
        "observed_age_seconds",
        "rotation_speed_rpm",
        "radial_load_n",
        "hi_current",
        "hi_recent_mean",
        "hi_recent_std",
        "hi_recent_max",
        "hi_recent_slope_per_hour",
        "hi_full_mean",
        "hi_full_max",
        "hi_full_slope_per_hour",
    ]
    for signal in HEALTH_SIGNAL_NAMES:
        cols.extend([f"{signal}_robust_z_current", f"{signal}_robust_z_recent_mean"])
    return cols


def _signal_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays = {
        name: frame[name].to_numpy(dtype=float)
        for name in BASE_SOURCE_FEATURES
    }
    arrays["fft_total_horiz"] = frame[FFT_HORIZ].sum(axis=1).to_numpy(dtype=float)
    arrays["fft_total_vert"] = frame[FFT_VERT].sum(axis=1).to_numpy(dtype=float)
    return arrays


def _robust_deviation(values: np.ndarray, healthy_window: int, clip: float) -> np.ndarray:
    healthy = np.asarray(values[: min(healthy_window, len(values))], dtype=float)
    median = float(np.median(healthy))
    mad = float(np.median(np.abs(healthy - median)))
    std = float(np.std(healthy))
    # Robust scale with conservative fallbacks for nearly-constant healthy signals.
    scale = max(1.4826 * mad, 0.10 * std, 0.01 * abs(median), 1e-8)
    z = np.abs(np.asarray(values, dtype=float) - median) / scale
    return np.clip(z, 0.0, float(clip))


def _one_health_prefix(
    bearing_frame: pd.DataFrame,
    cut_position: int,
    *,
    fraction: float,
    healthy_window: int,
    recent_window: int,
    robust_z_clip: float,
) -> dict[str, object]:
    ordered = bearing_frame.sort_values("file_index").reset_index(drop=True)
    prefix = ordered.iloc[: cut_position + 1]
    endpoint = prefix.iloc[-1]

    condition = int(endpoint["condition"])
    speed, load = _physical_context(condition)
    observed_age = float(endpoint["file_index"]) * float(FILE_INTERVAL_SECONDS)

    result: dict[str, object] = {
        "condition": condition,
        "bearing": str(endpoint["bearing"]),
        "cut_fraction": float(fraction),
        "cut_file_index": int(endpoint["file_index"]),
        "observed_age_seconds": observed_age,
        "rotation_speed_rpm": speed,
        "radial_load_n": load,
        "rul_seconds": float(endpoint["rul_seconds"]),
    }

    z_series: dict[str, np.ndarray] = {}
    for signal, values in _signal_arrays(prefix).items():
        z = _robust_deviation(values, healthy_window, robust_z_clip)
        z_series[signal] = z
        recent = z[-min(recent_window, len(z)) :]
        result[f"{signal}_robust_z_current"] = float(z[-1])
        result[f"{signal}_robust_z_recent_mean"] = float(np.mean(recent))

    # A single robust degradation trajectory. log1p limits domination by one
    # extreme sensor statistic while preserving increasing degradation evidence.
    stacked = np.column_stack([np.log1p(z_series[name]) for name in HEALTH_SIGNAL_NAMES])
    hi = np.median(stacked, axis=1)
    recent_hi = hi[-min(recent_window, len(hi)) :]

    result.update(
        {
            "hi_current": float(hi[-1]),
            "hi_recent_mean": float(np.mean(recent_hi)),
            "hi_recent_std": float(np.std(recent_hi)),
            "hi_recent_max": float(np.max(recent_hi)),
            "hi_recent_slope_per_hour": _slope_per_hour(recent_hi),
            "hi_full_mean": float(np.mean(hi)),
            "hi_full_max": float(np.max(hi)),
            "hi_full_slope_per_hour": _slope_per_hour(hi),
        }
    )
    return result


def build_health_indicator_samples(
    train_frame: pd.DataFrame,
    *,
    fractions: Iterable[float],
    healthy_window: int = 60,
    recent_window: int = 60,
    robust_z_clip: float = 50.0,
) -> pd.DataFrame:
    """Build causal Health Indicator V2 pseudo-prefix samples."""
    required = {
        "condition",
        "bearing",
        "file_index",
        "rul_seconds",
        *BASE_SOURCE_FEATURES,
        *FFT_HORIZ,
        *FFT_VERT,
    }
    missing = sorted(required - set(train_frame.columns))
    if missing:
        raise ValueError(f"health-indicator input missing columns: {missing}")
    fractions = tuple(float(v) for v in fractions)
    if not fractions or any(v <= 0.0 or v >= 1.0 for v in fractions):
        raise ValueError("prefix fractions must be between 0 and 1")
    if healthy_window < 2 or recent_window < 2:
        raise ValueError("healthy_window and recent_window must be >= 2")

    rows: list[dict[str, object]] = []
    for bearing, bearing_frame in train_frame.groupby("bearing", sort=True):
        ordered = bearing_frame.sort_values("file_index").reset_index(drop=True)
        if len(ordered) <= healthy_window + 2:
            raise ValueError(f"bearing {bearing} is too short for healthy_window={healthy_window}")
        seen: set[int] = set()
        for fraction in fractions:
            cut_position = int(round((len(ordered) - 1) * fraction))
            cut_position = min(max(cut_position, healthy_window), len(ordered) - 2)
            if cut_position in seen:
                continue
            seen.add(cut_position)
            rows.append(
                _one_health_prefix(
                    ordered,
                    cut_position,
                    fraction=fraction,
                    healthy_window=healthy_window,
                    recent_window=recent_window,
                    robust_z_clip=robust_z_clip,
                )
            )

    result = pd.DataFrame(rows).sort_values(["bearing", "cut_file_index"]).reset_index(drop=True)
    numeric = result[[*health_indicator_feature_columns(), "rul_seconds"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("health-indicator generation produced non-finite values")
    return result
