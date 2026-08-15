"""Ties ingestion + feature extraction + RUL labeling together into the
per-snapshot feature table that later orchestrator consumes.

One row per (bearing, file_index) snapshot: metadata columns (split,
condition, bearing, file_index, elapsed_time_seconds), time/frequency
domain features per channel, and the rul_seconds label.
"""

import re
from pathlib import Path

import pandas as pd

from femto_rul.config import FILE_INTERVAL_SECONDS
from femto_rul.features.frequency_domain import fft_band_energy, fft_band_feature_names
from femto_rul.features.time_domain import TIME_DOMAIN_FEATURE_NAMES, time_domain_features
from femto_rul.ingestion.raw_loader import file_index, list_bearing_files, load_acc_file
from femto_rul.labeling.rul import label_full_run_bearing, label_truncated_bearing

BEARING_DIR_RE = re.compile(r"^Bearing(\d+)_(\d+)$")
CHANNELS = [("horiz", "horiz_accel_g"), ("vert", "vert_accel_g")]

# Feature Set V1 column names, in the exact order extract_snapshot_features
# produces them: per channel (horiz then vert), time-domain features then
# FFT band energies. This is the single source of truth other modules
# (serving telemetry, monitoring) should import rather than re-listing.
FEATURE_COLUMNS_V1 = [
    f"{name}_{channel_name}"
    for channel_name, _ in CHANNELS
    for name in (TIME_DOMAIN_FEATURE_NAMES + fft_band_feature_names())
]


def extract_snapshot_features(acc_path: Path) -> dict[str, float]:
    """All time- and frequency-domain features for one acc_*.csv snapshot,
    computed separately per channel (e.g. "rms_horiz", "fft_band_0_vert")."""
    df = load_acc_file(acc_path)
    features: dict[str, float] = {}
    for channel_name, column in CHANNELS:
        signal = df[column].to_numpy()
        for name, value in time_domain_features(signal).items():
            features[f"{name}_{channel_name}"] = value
        for name, value in fft_band_energy(signal).items():
            features[f"{name}_{channel_name}"] = value
    return features


def extract_bearing_features(bearing_dir: Path) -> pd.DataFrame:
    """One feature row per acc_*.csv file in this bearing, indexed by file_index."""
    rows = []
    for path in list_bearing_files(bearing_dir, "acc"):
        row = {"file_index": file_index(path)}
        row.update(extract_snapshot_features(path))
        rows.append(row)
    return pd.DataFrame(rows)


def build_bearing_dataset(
    bearing_dir: Path, split_name: str, validation_bearing_dir: Path | None = None
) -> pd.DataFrame:
    """Feature table for one bearing, with metadata and the rul_seconds label.

    Pass validation_bearing_dir for Test_set bearings (truncated — the true
    total run length comes from the matching Validation_Set bearing).
    Training_set and Validation_Set bearings are full runs, so they're
    self-labeled.
    """
    match = BEARING_DIR_RE.match(bearing_dir.name)
    if not match:
        raise ValueError(f"unexpected bearing directory name: {bearing_dir.name}")
    condition, unit = int(match.group(1)), int(match.group(2))

    features = extract_bearing_features(bearing_dir)
    if validation_bearing_dir is not None:
        labels = label_truncated_bearing(bearing_dir, validation_bearing_dir)
    else:
        labels = label_full_run_bearing(bearing_dir)

    df = features.merge(labels, on="file_index", validate="one_to_one")
    df.insert(0, "elapsed_time_seconds", (df["file_index"] - 1) * FILE_INTERVAL_SECONDS)
    df.insert(0, "bearing", bearing_dir.name)
    df.insert(0, "condition", condition)
    df.insert(0, "split", split_name)
    return df


def build_split_dataset(
    split_dir: Path, split_name: str, validation_dir: Path | None = None
) -> pd.DataFrame:
    """Feature table for every bearing in a split directory."""
    frames = []
    for bearing_dir in sorted(split_dir.iterdir()):
        if not bearing_dir.is_dir():
            continue
        validation_bearing_dir = (
            validation_dir / bearing_dir.name if validation_dir is not None else None
        )
        frames.append(build_bearing_dataset(bearing_dir, split_name, validation_bearing_dir))
    return pd.concat(frames, ignore_index=True)


def build_full_dataset(data_dir: Path) -> pd.DataFrame:
    """Feature table across Training_set, Validation_Set, and Test_set."""
    training = build_split_dataset(data_dir / "Training_set", "Training_set")
    validation = build_split_dataset(data_dir / "Validation_Set", "Validation_Set")
    test = build_split_dataset(
        data_dir / "Test_set", "Test_set", validation_dir=data_dir / "Validation_Set"
    )
    return pd.concat([training, validation, test], ignore_index=True)
