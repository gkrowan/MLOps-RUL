"""Regression metrics for FEMTO/PRONOSTIA RUL prediction.

The PHM12 score follows the asymmetric accuracy function from the IEEE PHM
2012 bearing challenge. The original challenge applies the score to one RUL
estimate per test bearing. During LOBO model-development CV we report the same
function averaged over positive-RUL validation snapshots as a *diagnostic*.
The official holdout score should later be computed only at the final observed
snapshot of each of the 11 Test_set bearings.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _as_1d_float(values: object, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def phm12_score(y_true: object, y_pred: object) -> float:
    """Return the IEEE PHM 2012 asymmetric RUL accuracy score.

    Percent error is ``100 * (actual - predicted) / actual``. Overestimating
    RUL (a late prediction) is penalized more strongly than underestimating it.

    Rows with ``actual RUL <= 0`` are excluded because percent error is
    undefined at zero RUL. A ValueError is raised if no positive-RUL rows
    remain.
    """
    actual = _as_1d_float(y_true, name="y_true")
    predicted = _as_1d_float(y_pred, name="y_pred")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    mask = actual > 0
    if not mask.any():
        raise ValueError("PHM12 score requires at least one y_true > 0")

    actual = actual[mask]
    predicted = predicted[mask]
    error_pct = 100.0 * (actual - predicted) / actual

    late = error_pct <= 0
    accuracy = np.empty_like(error_pct, dtype=float)
    accuracy[late] = np.exp(-math.log(0.5) * (error_pct[late] / 5.0))
    accuracy[~late] = np.exp(math.log(0.5) * (error_pct[~late] / 20.0))
    return float(np.mean(accuracy))


def regression_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    """Return the standard row-level regression metrics used in development."""
    actual = _as_1d_float(y_true, name="y_true")
    predicted = _as_1d_float(y_pred, name="y_pred")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "phm12_snapshot_score": phm12_score(actual, predicted),
    }
