"""Readers for raw FEMTO acc_*.csv / temp_*.csv files.

The dataset's delimiter is not uniform (see docs/data_notes.md): most files
are comma-delimited, but some bearings ship semicolon-delimited acc and/or
temp files instead, and the choice can differ between the Test_set and
Validation_Set copies of the same bearing. Every reader here detects the
delimiter per file rather than assuming one.
"""

import re
from pathlib import Path

import pandas as pd

from femto_rul.config import ACC_COLUMNS, TEMP_COLUMNS

FILE_INDEX_RE = re.compile(r"_(\d+)\.csv$")


def _file_index(path: Path) -> int:
    match = FILE_INDEX_RE.search(path.name)
    if not match:
        raise ValueError(f"unexpected filename, can't parse index: {path.name}")
    return int(match.group(1))


def _sniff_delimiter(path: Path) -> str:
    with path.open("rb") as f:
        first_line = f.readline()
    return ";" if b";" in first_line else ","


def load_acc_file(path: Path) -> pd.DataFrame:
    """Load a single acc_*.csv into a DataFrame with ACC_COLUMNS."""
    return pd.read_csv(path, header=None, names=ACC_COLUMNS, sep=_sniff_delimiter(path))


def load_temp_file(path: Path) -> pd.DataFrame:
    """Load a single temp_*.csv into a DataFrame with TEMP_COLUMNS."""
    return pd.read_csv(path, header=None, names=TEMP_COLUMNS, sep=_sniff_delimiter(path))


def list_bearing_files(bearing_dir: Path, kind: str) -> list[Path]:
    """List a bearing's acc_*.csv or temp_*.csv files in file-index order.

    kind must be "acc" or "temp".
    """
    if kind not in ("acc", "temp"):
        raise ValueError(f"kind must be 'acc' or 'temp', got {kind!r}")
    return sorted(bearing_dir.glob(f"{kind}_*.csv"), key=_file_index)


def load_bearing_acc(bearing_dir: Path) -> pd.DataFrame:
    """Load and concatenate all acc_*.csv files for one bearing, tagged with
    file_index (the position of that 0.1s snapshot in the run)."""
    frames = []
    for path in list_bearing_files(bearing_dir, "acc"):
        df = load_acc_file(path)
        df.insert(0, "file_index", _file_index(path))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_bearing_temp(bearing_dir: Path) -> pd.DataFrame:
    """Load and concatenate all temp_*.csv files for one bearing, tagged with
    file_index. Returns an empty DataFrame if the bearing has no temp data."""
    frames = []
    for path in list_bearing_files(bearing_dir, "temp"):
        df = load_temp_file(path)
        df.insert(0, "file_index", _file_index(path))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["file_index", *TEMP_COLUMNS])
    return pd.concat(frames, ignore_index=True)
