"""RUL (remaining useful life) label construction.

Labeling scheme: raw time-to-failure in seconds, chosen to match the
scoring metrics (RMSE + PHM12 asymmetric scoring), which operate on real
time units rather than a normalized life-percentage — see docs/data_notes.md
for the full rationale.

RUL(file_index) = (total_snapshots_in_full_run - file_index) * FILE_INTERVAL_SECONDS

For Training_set bearings, "total_snapshots_in_full_run" is just that
bearing's own file count, since it's a full run-to-failure. For Test_set
bearings (truncated), the true total comes from the matching Validation_Set
bearing instead — that's the whole point of Validation_Set (see
docs/data_notes.md).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from femto_rul.config import FILE_INTERVAL_SECONDS
from femto_rul.ingestion.raw_loader import file_index, list_bearing_files


def rul_seconds(file_indices: np.ndarray | pd.Series, total_snapshots: int) -> np.ndarray:
    """RUL in seconds for each file_index, given the run's true total length.

    file_index is 1-based (matches the acc_00001.csv naming), so the last
    snapshot (file_index == total_snapshots) gets RUL 0.
    """
    file_indices = np.asarray(file_indices)
    return (total_snapshots - file_indices) * FILE_INTERVAL_SECONDS


def label_full_run_bearing(bearing_dir: Path) -> pd.DataFrame:
    """RUL labels for a bearing whose directory IS the full run-to-failure
    (Training_set bearings, or a Validation_Set bearing considered on its own)."""
    file_indices = [file_index(p) for p in list_bearing_files(bearing_dir, "acc")]
    total_snapshots = max(file_indices)
    return pd.DataFrame(
        {
            "file_index": file_indices,
            "rul_seconds": rul_seconds(file_indices, total_snapshots),
        }
    )


def label_truncated_bearing(test_bearing_dir: Path, validation_bearing_dir: Path) -> pd.DataFrame:
    """RUL labels for a Test_set bearing, using its matching Validation_Set
    bearing to find the true total run length."""
    test_indices = [file_index(p) for p in list_bearing_files(test_bearing_dir, "acc")]
    validation_indices = [file_index(p) for p in list_bearing_files(validation_bearing_dir, "acc")]
    total_snapshots = max(validation_indices)
    return pd.DataFrame(
        {
            "file_index": test_indices,
            "rul_seconds": rul_seconds(test_indices, total_snapshots),
        }
    )
