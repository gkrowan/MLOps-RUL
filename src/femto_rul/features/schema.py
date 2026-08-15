"""Feature Set V1 schema for production dataset artifacts.

The feature schema intentionally separates:
- metadata used for grouping/auditing,
- signal features approved for the default model input,
- optional context columns that may be tested explicitly,
- target / leakage columns that must never enter model features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from femto_rul.features.frequency_domain import fft_band_feature_names
from femto_rul.features.time_domain import TIME_DOMAIN_FEATURE_NAMES

FEATURE_SET_VERSION: Final[str] = "v1"
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

# Operating condition is known at inference time and can be evaluated later as
# an explicit modeling choice. It is not part of the default V1 predictor set.
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
    """Return Feature Set V1 signal columns in deterministic order."""
    columns: list[str] = []
    for channel_name, _ in CHANNEL_SPECS:
        columns.extend(f"{name}_{channel_name}" for name in TIME_DOMAIN_FEATURE_NAMES)
        columns.extend(
            f"{name}_{channel_name}" for name in fft_band_feature_names()
        )
    return columns


def feature_schema() -> dict[str, object]:
    """Return the machine-readable schema for processed artifacts."""
    signal_columns = signal_feature_columns()
    return {
        "feature_set_version": FEATURE_SET_VERSION,
        "metadata_columns": METADATA_COLUMNS,
        "signal_feature_columns": signal_columns,
        "default_model_feature_columns": signal_columns,
        "optional_context_columns": OPTIONAL_CONTEXT_COLUMNS,
        "blocked_predictor_columns": BLOCKED_PREDICTOR_COLUMNS,
        "target_column": TARGET_COLUMN,
        "ground_truth_key_columns": GROUND_TRUTH_KEY_COLUMNS,
        "train_columns": [*METADATA_COLUMNS, *signal_columns, TARGET_COLUMN],
        "test_feature_columns": [*METADATA_COLUMNS, *signal_columns],
        "test_ground_truth_columns": [*GROUND_TRUTH_KEY_COLUMNS, TARGET_COLUMN],
        "kurtosis_convention": "Fisher/excess (Gaussian approximately 0)",
        "fft_bands_per_channel": len(fft_band_feature_names()),
    }


def write_feature_schema(path: Path) -> None:
    """Write the Feature Set V1 schema atomically as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(feature_schema(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
