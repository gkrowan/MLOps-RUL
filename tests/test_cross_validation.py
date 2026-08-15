import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from femto_rul.evaluation.cross_validation import (
    leave_one_bearing_out_cv,
    summarize_cv_metrics,
)


def test_leave_one_bearing_out_has_one_fold_per_group():
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(2.0 * X["x"] + 5.0)
    groups = pd.Series(["B1"] * 4 + ["B2"] * 4 + ["B3"] * 4)

    folds, predictions = leave_one_bearing_out_cv(
        LinearRegression(), X, y, groups, model_name="linear"
    )

    assert len(folds) == 3
    assert set(folds["held_out_bearing"]) == {"B1", "B2", "B3"}
    assert len(predictions) == len(X)
    assert predictions.groupby("row_index").size().eq(1).all()


def test_summary_orders_by_mean_rmse():
    folds = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "fold": [1, 2, 1, 2],
            "rmse": [10.0, 12.0, 3.0, 5.0],
            "mae": [8.0, 9.0, 2.0, 4.0],
            "r2": [0.1, 0.2, 0.8, 0.7],
            "phm12_snapshot_score": [0.4, 0.5, 0.8, 0.7],
        }
    )
    summary = summarize_cv_metrics(folds)
    assert summary.iloc[0]["model"] == "b"
