import numpy as np
import pytest

from femto_rul.config import FILE_INTERVAL_SECONDS, TEST_SET_DIR, TRAINING_SET_DIR, VALIDATION_SET_DIR
from femto_rul.labeling.rul import label_full_run_bearing, label_truncated_bearing, rul_seconds

pytestmark = pytest.mark.skipif(
    not TRAINING_SET_DIR.is_dir(), reason="data/ not populated on this machine"
)


def test_rul_seconds_counts_down_to_zero_at_last_file():
    file_indices = np.array([1, 2, 3, 4, 5])
    result = rul_seconds(file_indices, total_snapshots=5)
    assert result[-1] == 0
    assert list(result) == [40, 30, 20, 10, 0]
    assert result[0] - result[1] == FILE_INTERVAL_SECONDS


def test_label_full_run_bearing_monotonic_decreasing():
    labels = label_full_run_bearing(TRAINING_SET_DIR / "Bearing3_1")
    assert labels["rul_seconds"].min() == 0
    assert labels.sort_values("file_index")["rul_seconds"].is_monotonic_decreasing


def test_label_truncated_bearing_rul_never_reaches_zero():
    # Test_set bearings are truncated before failure, so RUL at the last
    # available snapshot should be > 0 (that's the point of the challenge).
    labels = label_truncated_bearing(
        TEST_SET_DIR / "Bearing1_3", VALIDATION_SET_DIR / "Bearing1_3"
    )
    assert labels["rul_seconds"].min() > 0


def test_label_truncated_bearing_matches_known_verified_truncation():
    # scripts/verify_data.py confirmed Bearing1_3: 1802/2375 acc files,
    # 573 files (5730s) held out as RUL.
    labels = label_truncated_bearing(
        TEST_SET_DIR / "Bearing1_3", VALIDATION_SET_DIR / "Bearing1_3"
    )
    last_row = labels.sort_values("file_index").iloc[-1]
    assert last_row["rul_seconds"] == 573 * FILE_INTERVAL_SECONDS
