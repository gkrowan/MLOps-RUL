import pytest

from femto_rul.config import TEST_SET_DIR, TRAINING_SET_DIR, VALIDATION_SET_DIR
from femto_rul.pipeline import build_bearing_dataset

pytestmark = pytest.mark.skipif(
    not TRAINING_SET_DIR.is_dir(), reason="data/ not populated on this machine"
)


def test_build_bearing_dataset_training_bearing():
    # Bearing3_1 is the smallest training bearing (515 files) — fastest to test against.
    df = build_bearing_dataset(TRAINING_SET_DIR / "Bearing3_1", "Training_set")

    assert len(df) == 515
    assert df["condition"].unique().tolist() == [3]
    assert df["bearing"].unique().tolist() == ["Bearing3_1"]
    assert df["rul_seconds"].min() == 0
    assert not df.isna().any().any()


def test_build_bearing_dataset_truncated_test_bearing():
    # Bearing2_7 is the smallest test bearing (172 files) — fastest to test against.
    df = build_bearing_dataset(
        TEST_SET_DIR / "Bearing2_7",
        "Test_set",
        validation_bearing_dir=VALIDATION_SET_DIR / "Bearing2_7",
    )

    assert len(df) == 172
    assert df["rul_seconds"].min() > 0  # truncated before failure
    assert not df.isna().any().any()
