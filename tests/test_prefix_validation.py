import pandas as pd
from sklearn.linear_model import Ridge

from femto_rul.evaluation.prefix_validation import prefix_lobo_cv


def test_prefix_lobo_holds_out_entire_bearing():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    groups = pd.Series(["A", "A", "B", "B", "C", "C"])
    metrics, predictions = prefix_lobo_cv(Ridge(), X, y, groups, model_name="ridge")
    assert len(metrics) == 3
    assert set(metrics["held_out_bearing"]) == {"A", "B", "C"}
    assert len(predictions) == 6
