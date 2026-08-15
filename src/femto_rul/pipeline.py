"""Production-safe FEMTO feature/dataset construction.

The key contract is that model-development data and official holdout ground
truth are built through separate APIs:

- ``build_training_dataset`` reads Training_set only and includes ``rul_seconds``.
- ``build_test_feature_dataset`` reads Test_set only and NEVER includes true RUL.
- ``build_test_ground_truth`` is the only production API allowed to combine
  Test_set with Validation_Set to derive official holdout labels.

Legacy combined builders are retained only for backward compatibility with
older notebooks/tests and emit warnings so they are not silently used in the
production training path.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from femto_rul.config import FILE_INTERVAL_SECONDS
from femto_rul.features.frequency_domain import fft_band_energy
from femto_rul.features.schema import CHANNEL_SPECS, GROUND_TRUTH_KEY_COLUMNS
from femto_rul.features.time_domain import time_domain_features
from femto_rul.ingestion.raw_loader import file_index, list_bearing_files, load_acc_file
from femto_rul.labeling.rul import label_full_run_bearing, label_truncated_bearing

BEARING_DIR_RE = re.compile(r"^Bearing(\d+)_(\d+)$")


def _bearing_identity(bearing_dir: Path) -> tuple[int, str]:
    match = BEARING_DIR_RE.match(bearing_dir.name)
    if not match:
        raise ValueError(f"unexpected bearing directory name: {bearing_dir.name}")
    condition = int(match.group(1))
    return condition, bearing_dir.name


def _bearing_directories(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"dataset split directory does not exist: {split_dir}")

    bearings = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
    if not bearings:
        raise ValueError(f"no bearing directories found under {split_dir}")
    return bearings


def _concat(frames: list[pd.DataFrame], *, context: str) -> pd.DataFrame:
    if not frames:
        raise ValueError(f"no frames produced while building {context}")
    return pd.concat(frames, ignore_index=True)


def extract_snapshot_features(acc_path: Path) -> dict[str, float]:
    """Compute Feature Set V1 for one vibration snapshot."""
    df = load_acc_file(acc_path)
    features: dict[str, float] = {}

    for channel_name, column in CHANNEL_SPECS:
        signal = df[column].to_numpy()

        for name, value in time_domain_features(signal).items():
            features[f"{name}_{channel_name}"] = value

        for name, value in fft_band_energy(signal).items():
            features[f"{name}_{channel_name}"] = value

    return features


def extract_bearing_features(bearing_dir: Path) -> pd.DataFrame:
    """Return one Feature Set V1 row per ``acc_*.csv`` snapshot."""
    rows: list[dict[str, float | int]] = []

    for path in list_bearing_files(bearing_dir, "acc"):
        row: dict[str, float | int] = {"file_index": file_index(path)}
        row.update(extract_snapshot_features(path))
        rows.append(row)

    if not rows:
        raise ValueError(f"no acceleration files found under {bearing_dir}")

    return pd.DataFrame(rows)


def build_bearing_feature_dataset(bearing_dir: Path, split_name: str) -> pd.DataFrame:
    """Build metadata + features for one bearing without any RUL label."""
    condition, bearing_name = _bearing_identity(bearing_dir)
    df = extract_bearing_features(bearing_dir)

    df.insert(0, "elapsed_time_seconds", (df["file_index"] - 1) * FILE_INTERVAL_SECONDS)
    df.insert(0, "bearing", bearing_name)
    df.insert(0, "condition", condition)
    df.insert(0, "split", split_name)
    return df


def build_training_bearing_dataset(bearing_dir: Path) -> pd.DataFrame:
    """Build one full-run Training_set bearing with RUL labels."""
    features = build_bearing_feature_dataset(bearing_dir, "Training_set")
    labels = label_full_run_bearing(bearing_dir)
    return features.merge(labels, on="file_index", validate="one_to_one")


def build_training_dataset(training_dir: Path) -> pd.DataFrame:
    """Build the production training table from Training_set only.

    This function intentionally has no Validation_Set argument. It must remain
    runnable when official holdout ground truth is unavailable.
    """
    frames = [
        build_training_bearing_dataset(bearing_dir)
        for bearing_dir in _bearing_directories(training_dir)
    ]
    return _concat(frames, context="Training_set")


def build_test_feature_dataset(test_dir: Path) -> pd.DataFrame:
    """Build official holdout inference features from Test_set only.

    The returned DataFrame never contains ``rul_seconds`` and does not access
    Validation_Set.
    """
    frames = [
        build_bearing_feature_dataset(bearing_dir, "Test_set")
        for bearing_dir in _bearing_directories(test_dir)
    ]
    df = _concat(frames, context="Test_set features")

    if "rul_seconds" in df.columns:  # defensive contract check
        raise AssertionError("test feature dataset must not contain rul_seconds")
    return df


def build_test_ground_truth(test_dir: Path, validation_dir: Path) -> pd.DataFrame:
    """Derive official Test_set RUL ground truth from Validation_Set.

    This is the only production path that is allowed to use Validation_Set.
    It performs no feature extraction and produces only keys + ``rul_seconds``.
    """
    frames: list[pd.DataFrame] = []

    for test_bearing_dir in _bearing_directories(test_dir):
        condition, bearing_name = _bearing_identity(test_bearing_dir)
        validation_bearing_dir = validation_dir / bearing_name

        if not validation_bearing_dir.is_dir():
            raise FileNotFoundError(
                f"missing Validation_Set ground-truth bearing: {validation_bearing_dir}"
            )

        labels = label_truncated_bearing(test_bearing_dir, validation_bearing_dir)
        labels.insert(0, "bearing", bearing_name)
        labels.insert(0, "condition", condition)
        frames.append(labels)

    ground_truth = _concat(frames, context="Test_set ground truth")
    return ground_truth[[*GROUND_TRUTH_KEY_COLUMNS, "rul_seconds"]]


# ---------------------------------------------------------------------------
# Backward-compatible analysis helpers
# ---------------------------------------------------------------------------

def build_bearing_dataset(
    bearing_dir: Path,
    split_name: str,
    validation_bearing_dir: Path | None = None,
) -> pd.DataFrame:
    """Legacy labeled-bearing builder retained for notebooks/tests.

    Production code should use ``build_training_bearing_dataset`` or
    ``build_bearing_feature_dataset`` instead.
    """
    warnings.warn(
        "build_bearing_dataset() is retained for analysis compatibility; "
        "production code should use explicit training/test APIs.",
        FutureWarning,
        stacklevel=2,
    )

    features = build_bearing_feature_dataset(bearing_dir, split_name)
    labels = (
        label_truncated_bearing(bearing_dir, validation_bearing_dir)
        if validation_bearing_dir is not None
        else label_full_run_bearing(bearing_dir)
    )
    return features.merge(labels, on="file_index", validate="one_to_one")


def build_split_dataset(
    split_dir: Path,
    split_name: str,
    validation_dir: Path | None = None,
) -> pd.DataFrame:
    """Legacy labeled split builder retained for exploratory compatibility."""
    frames: list[pd.DataFrame] = []

    for bearing_dir in _bearing_directories(split_dir):
        validation_bearing_dir = (
            validation_dir / bearing_dir.name if validation_dir is not None else None
        )
        frames.append(
            build_bearing_dataset(
                bearing_dir,
                split_name,
                validation_bearing_dir=validation_bearing_dir,
            )
        )

    return _concat(frames, context=split_name)


def build_full_dataset(data_dir: Path) -> pd.DataFrame:
    """Legacy combined labeled artifact for historical analysis only.

    WARNING: this contains Validation_Set and true Test_set RUL. It must never
    be used as a production training/AutoML input.
    """
    warnings.warn(
        "build_full_dataset() creates a leakage-prone combined labeled artifact. "
        "Use build_training_dataset(), build_test_feature_dataset(), and "
        "build_test_ground_truth() for production workflows.",
        FutureWarning,
        stacklevel=2,
    )

    training = build_split_dataset(data_dir / "Training_set", "Training_set")
    validation = build_split_dataset(data_dir / "Validation_Set", "Validation_Set")
    test = build_split_dataset(
        data_dir / "Test_set",
        "Test_set",
        validation_dir=data_dir / "Validation_Set",
    )
    return pd.concat([training, validation, test], ignore_index=True)
