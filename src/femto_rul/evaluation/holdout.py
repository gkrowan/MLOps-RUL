"""Official endpoint holdout helpers for the FEMTO Test_set.

This module is intentionally separate from model-development evaluation.  It
uses one prediction at the final observed snapshot of each truncated Test_set
bearing and should only be invoked after model selection has been frozen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from femto_rul.evaluation.metrics import phm12_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def endpoint_ground_truth(ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Return one official RUL label per Test_set bearing at its last snapshot."""
    required = {"condition", "bearing", "file_index", "rul_seconds"}
    missing = sorted(required - set(ground_truth.columns))
    if missing:
        raise ValueError(f"ground truth missing columns: {missing}")

    ordered = ground_truth.sort_values(["bearing", "file_index"])
    endpoint = ordered.groupby("bearing", sort=True, as_index=False).tail(1).copy()
    endpoint = endpoint[["condition", "bearing", "file_index", "rul_seconds"]]
    endpoint = endpoint.rename(columns={"file_index": "cut_file_index"})
    return endpoint.sort_values("bearing").reset_index(drop=True)


def align_endpoint_features_and_truth(
    endpoint_features: pd.DataFrame,
    endpoint_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Validate one-to-one endpoint alignment and return joined evaluation rows."""
    keys = ["condition", "bearing", "cut_file_index"]
    merged = endpoint_features.merge(endpoint_truth, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(endpoint_features) or len(merged) != len(endpoint_truth):
        raise ValueError(
            "official endpoint feature/ground-truth mismatch: "
            f"features={len(endpoint_features)}, truth={len(endpoint_truth)}, joined={len(merged)}"
        )
    return merged.sort_values("bearing").reset_index(drop=True)


def official_endpoint_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    """Metrics for the 11 official Test_set endpoint predictions."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("official holdout arrays must be non-empty and have equal shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("official holdout arrays contain non-finite values")
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "phm12_score": float(phm12_score(actual, predicted)),
    }
