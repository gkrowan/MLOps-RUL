import pandas as pd

from femto_rul.evaluation.prefix_validation import monotonicity_summary


def test_monotonicity_summary_counts_rul_increases():
    frame = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "held_out_bearing": ["B"] * 4,
            "observed_age_seconds": [100.0, 200.0, 300.0, 400.0],
            "prediction_rul_seconds": [900.0, 800.0, 850.0, 700.0],
        }
    )
    out = monotonicity_summary(frame)
    assert int(out.iloc[0]["monotonic_violations"]) == 1
    assert int(out.iloc[0]["monotonic_comparisons"]) == 3
