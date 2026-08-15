import numpy as np
import pandas as pd

from femto_rul.features.temporal import (
    TEMPORAL_SOURCE_FEATURES,
    add_causal_temporal_features,
    temporal_feature_columns,
)


def _frame(n: int = 10) -> pd.DataFrame:
    data = {"file_index": np.arange(1, n + 1)}
    for offset, feature in enumerate(TEMPORAL_SOURCE_FEATURES):
        data[feature] = np.arange(n, dtype=float) + float(offset)
    return pd.DataFrame(data)


def test_temporal_feature_count_and_finiteness():
    out = add_causal_temporal_features(_frame())
    assert len(temporal_feature_columns()) == 54
    assert set(temporal_feature_columns()).issubset(out.columns)
    assert np.isfinite(out[temporal_feature_columns()].to_numpy()).all()


def test_temporal_features_are_causal():
    original = _frame(10)
    changed_future = original.copy()
    changed_future.loc[9, TEMPORAL_SOURCE_FEATURES] = 1_000_000.0

    a = add_causal_temporal_features(original)
    b = add_causal_temporal_features(changed_future)

    # Changing row 10 cannot alter any derived feature for rows 1..9.
    pd.testing.assert_frame_equal(
        a.loc[:8, temporal_feature_columns()],
        b.loc[:8, temporal_feature_columns()],
    )


def test_positive_linear_signal_has_positive_causal_slope():
    out = add_causal_temporal_features(_frame(10))
    assert out.loc[9, "rms_horiz_roll_slope_6"] > 0.0
