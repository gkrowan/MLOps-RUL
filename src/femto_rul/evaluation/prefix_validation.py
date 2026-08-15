"""Leave-One-Bearing-Out evaluation for pseudo-test bearing prefixes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from femto_rul.evaluation.metrics import phm12_score


def prefix_lobo_cv(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y, and groups must have equal lengths")

    logo = LeaveOneGroupOut()
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold, (train_idx, valid_idx) in enumerate(logo.split(X, y, groups), start=1):
        held = sorted(set(groups.iloc[valid_idx].astype(str)))
        if len(held) != 1:
            raise AssertionError("each prefix LOBO fold must hold out exactly one bearing")
        held_bearing = held[0]

        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = np.maximum(np.asarray(model.predict(X.iloc[valid_idx]), dtype=float), 0.0)
        actual = y.iloc[valid_idx].to_numpy(dtype=float)

        metrics_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "held_out_bearing": held_bearing,
                "train_prefixes": int(len(train_idx)),
                "validation_prefixes": int(len(valid_idx)),
                "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
                "mae": float(mean_absolute_error(actual, pred)),
                "r2": float(r2_score(actual, pred)),
                "phm12_prefix_score": phm12_score(actual, pred),
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "fold": fold,
                    "held_out_bearing": held_bearing,
                    "row_index": X.index[valid_idx].to_numpy(),
                    "actual_rul_seconds": actual,
                    "prediction_rul_seconds": pred,
                }
            )
        )

    return pd.DataFrame(metrics_rows), pd.concat(prediction_frames, ignore_index=True)


def summarize_prefix_cv(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_metrics.groupby("model", as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
            mean_phm12_prefix_score=("phm12_prefix_score", "mean"),
            folds=("fold", "count"),
        )
        .sort_values("mean_rmse")
        .reset_index(drop=True)
    )
