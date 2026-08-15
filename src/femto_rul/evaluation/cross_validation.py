"""Bearing-level cross-validation for RUL model development."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

from femto_rul.evaluation.metrics import regression_metrics


def leave_one_bearing_out_cv(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    model_name: str,
    clip_nonnegative: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one estimator using Leave-One-Bearing-Out cross-validation.

    Each fold trains on every bearing except one and validates on the held-out
    bearing. Group membership is never used as a predictor.
    """
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y, and groups must have the same number of rows")
    if X.empty:
        raise ValueError("X must not be empty")

    unique_groups = pd.Series(groups).astype(str).unique()
    if len(unique_groups) < 2:
        raise ValueError("LOBO CV requires at least two distinct bearings")

    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        logo.split(X, y, groups), start=1
    ):
        train_groups = set(groups.iloc[train_idx].astype(str))
        valid_groups = set(groups.iloc[valid_idx].astype(str))
        if len(valid_groups) != 1:
            raise AssertionError("LOBO fold must contain exactly one held-out bearing")
        if train_groups & valid_groups:
            raise AssertionError("bearing leakage detected between train and validation")

        held_out_bearing = next(iter(valid_groups))
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        raw_pred = np.asarray(model.predict(X.iloc[valid_idx]), dtype=float)
        pred = np.maximum(raw_pred, 0.0) if clip_nonnegative else raw_pred
        actual = y.iloc[valid_idx].to_numpy(dtype=float)
        metrics = regression_metrics(actual, pred)

        fold_rows.append(
            {
                "model": model_name,
                "fold": fold_number,
                "held_out_bearing": held_out_bearing,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(valid_idx)),
                **metrics,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "held_out_bearing": held_out_bearing,
                    "row_index": X.index[valid_idx].to_numpy(),
                    "actual_rul_seconds": actual,
                    "raw_prediction_rul_seconds": raw_pred,
                    "prediction_rul_seconds": pred,
                }
            )
        )

    fold_metrics = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return fold_metrics, predictions


def summarize_cv_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold metrics into one model-comparison row per model."""
    required = {
        "model",
        "rmse",
        "mae",
        "r2",
        "phm12_snapshot_score",
    }
    missing = required - set(fold_metrics.columns)
    if missing:
        raise ValueError(f"fold metrics missing required columns: {sorted(missing)}")

    summary = (
        fold_metrics.groupby("model", as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_phm12_snapshot_score=("phm12_snapshot_score", "mean"),
            std_phm12_snapshot_score=("phm12_snapshot_score", "std"),
            folds=("fold", "count"),
        )
        .sort_values("mean_rmse", ascending=True)
        .reset_index(drop=True)
    )
    return summary
