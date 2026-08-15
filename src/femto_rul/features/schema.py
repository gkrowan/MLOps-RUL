"""Feature Set V2 schema for production dataset artifacts.

V2 keeps the original 24 per-snapshot vibration features and adds causal
rolling degradation features. The temporal features use current/past signal
history only; elapsed time, file index, bearing ID, split, and RUL remain
blocked from model input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from femto_rul.features.frequency_domain import fft_band_feature_names
from femto_rul.features.temporal import temporal_feature_columns
from femto_rul.features.time_domain import TIME_DOMAIN_FEATURE_NAMES

FEATURE_SET_VERSION: Final[str] = "v2"
CHANNEL_SPECS: Final[list[tuple[str, str]]] = [
    ("horiz", "horiz_accel_g"),
    ("vert", "vert_accel_g"),
]

METADATA_COLUMNS: Final[list[str]] = [
    "split",
    "condition",
    "bearing",
    "elapsed_time_seconds",
    "file_index",
]

TARGET_COLUMN: Final[str] = "rul_seconds"

# Condition is known at inference time. The default V2 predictor set still
# excludes it so its impact can be measured explicitly in an ablation.
OPTIONAL_CONTEXT_COLUMNS: Final[list[str]] = ["condition"]

BLOCKED_PREDICTOR_COLUMNS: Final[list[str]] = [
    "split",
    "bearing",
    "elapsed_time_seconds",
    "file_index",
    TARGET_COLUMN,
]

GROUND_TRUTH_KEY_COLUMNS: Final[list[str]] = [
    "condition",
    "bearing",
    "file_index",
]


def signal_feature_columns() -> list[str]:
    """Return original V1 per-snapshot vibration features."""
    columns: list[str] = []
    for channel_name, _ in CHANNEL_SPECS:
        columns.extend(f"{name}_{channel_name}" for name in TIME_DOMAIN_FEATURE_NAMES)
        columns.extend(f"{name}_{channel_name}" for name in fft_band_feature_names())
    return columns


def feature_schema() -> dict[str, object]:
    """Return the machine-readable V2 processed-data contract."""
    signal_columns = signal_feature_columns()
    temporal_columns = temporal_feature_columns()
    default_model_columns = [*signal_columns, *temporal_columns]

    return {
        "feature_set_version": FEATURE_SET_VERSION,
        "metadata_columns": METADATA_COLUMNS,
        "signal_feature_columns": signal_columns,
        "temporal_feature_columns": temporal_columns,
        "default_model_feature_columns": default_model_columns,
        "optional_context_columns": OPTIONAL_CONTEXT_COLUMNS,
        "blocked_predictor_columns": BLOCKED_PREDICTOR_COLUMNS,
        "target_column": TARGET_COLUMN,
        "ground_truth_key_columns": GROUND_TRUTH_KEY_COLUMNS,
        "train_columns": [*METADATA_COLUMNS, *default_model_columns, TARGET_COLUMN],
        "test_feature_columns": [*METADATA_COLUMNS, *default_model_columns],
        "test_ground_truth_columns": [*GROUND_TRUTH_KEY_COLUMNS, TARGET_COLUMN],
        "kurtosis_convention": "Fisher/excess (Gaussian approximately 0)",
        "fft_bands_per_channel": len(fft_band_feature_names()),
        "temporal_contract": {
            "causal": True,
            "history_scope": "same bearing, current and past snapshots only",
            "windows_snapshots": [6, 30, 60],
            "approx_windows_minutes": [1, 5, 10],
            "stats": ["mean", "std", "slope"],
            "elapsed_time_predictor_used": False,
        },
    }


def write_feature_schema(path: Path) -> None:
    """Write the Feature Set V2 schema atomically as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(feature_schema(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
