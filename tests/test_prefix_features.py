import pandas as pd

from femto_rul.features.prefix import build_prefix_training_samples, prefix_feature_columns


def _bearing(name: str, condition: int, n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "condition": condition,
                "bearing": name,
                "file_index": i,
                "rul_seconds": (n - 1 - i) * 10.0,
                "rms_horiz": 1.0 + i * 0.01,
                "rms_vert": 1.2 + i * 0.01,
                "kurtosis_horiz": 2.0 + i * 0.02,
                "kurtosis_vert": 2.1 + i * 0.02,
                "crest_factor_horiz": 3.0 + i * 0.005,
                "crest_factor_vert": 3.1 + i * 0.005,
            }
        )
    return pd.DataFrame(rows)


def test_prefix_samples_are_one_row_per_bearing_cut():
    frame = pd.concat([_bearing("Bearing1_1", 1, 100), _bearing("Bearing2_1", 2, 80)])
    out = build_prefix_training_samples(frame, fractions=(0.5, 0.8))
    assert len(out) == 4
    assert set(out["bearing"]) == {"Bearing1_1", "Bearing2_1"}
    assert all(col in out.columns for col in prefix_feature_columns())
    assert (out["rul_seconds"] > 0).all()


def test_observed_age_is_derived_from_visible_cut_only():
    frame = _bearing("Bearing1_1", 1, 101)
    out = build_prefix_training_samples(frame, fractions=(0.5,))
    row = out.iloc[0]
    assert row["cut_file_index"] == 50
    assert row["observed_age_seconds"] == 500.0
    assert row["rul_seconds"] == 500.0
