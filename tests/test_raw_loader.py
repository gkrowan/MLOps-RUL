"""Sanity checks for the raw FEMTO loader against the actual downloaded dataset.

Skips if data/ isn't populated locally (e.g. on a machine that hasn't
downloaded the dataset yet) rather than failing.
"""

import pytest

from femto_rul.config import ACC_COLUMNS, TEMP_COLUMNS, TRAINING_SET_DIR, VALIDATION_SET_DIR
from femto_rul.ingestion.raw_loader import load_bearing_acc, load_bearing_temp

pytestmark = pytest.mark.skipif(
    not TRAINING_SET_DIR.is_dir(), reason="data/ not populated on this machine"
)


def test_load_bearing_acc_comma_delimited():
    df = load_bearing_acc(TRAINING_SET_DIR / "Bearing1_1")
    assert list(df.columns) == ["file_index", *ACC_COLUMNS]
    assert len(df) == 2803 * 2560
    assert df["file_index"].min() == 1
    assert df["file_index"].max() == 2803


def test_load_bearing_acc_semicolon_delimited():
    """Bearing1_4 in Validation_Set ships semicolon-delimited — the loader
    must autodetect this rather than assume a comma."""
    df = load_bearing_acc(VALIDATION_SET_DIR / "Bearing1_4")
    assert list(df.columns) == ["file_index", *ACC_COLUMNS]
    assert len(df) == 1428 * 2560


def test_load_bearing_temp_missing_returns_empty():
    """Bearing2_2 has no temp_*.csv files at all in the raw release."""
    df = load_bearing_temp(TRAINING_SET_DIR / "Bearing2_2")
    assert list(df.columns) == ["file_index", *TEMP_COLUMNS]
    assert len(df) == 0


def test_load_bearing_temp_present():
    df = load_bearing_temp(TRAINING_SET_DIR / "Bearing1_1")
    assert list(df.columns) == ["file_index", *TEMP_COLUMNS]
    assert len(df) > 0
