"""Reusable helpers for training RUL regressors from the feature table."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)


MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 12,
        "random_state": 42,
        "n_jobs": -1,
    },
    "extra_trees": {
        "n_estimators": 200,
        "max_depth": 12,
        "random_state": 42,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state": 42,
    },
}

MODEL_CLASSES = {
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}

NON_FEATURE_COLUMNS = {
    "dataset_version",
    "split",
    "bearing",
    "file_index",
    "elapsed_time_seconds",
    "rul_seconds",
}


def build_regressor(
    model_type: str, hyperparameter_overrides: dict[str, Any] | None = None
) -> tuple[RegressorMixin, dict[str, Any]]:
    """Construct a supported regressor and return its effective parameters."""
    if model_type not in MODEL_CLASSES:
        supported = ", ".join(sorted(MODEL_CLASSES))
        raise ValueError(f"Unsupported model_type {model_type!r}; choose from {supported}")

    parameters = {**MODEL_DEFAULTS[model_type], **(hyperparameter_overrides or {})}
    model = MODEL_CLASSES[model_type](**parameters)
    return model, parameters


def prepare_feature_splits(
    dataset: pd.DataFrame,
    *,
    train_split: str,
    evaluation_split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    """Select finite numeric features and return train/evaluation matrices."""
    required = {"split", "rul_seconds"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Feature table is missing required columns: {', '.join(missing)}")

    numeric_columns = set(dataset.select_dtypes(include="number").columns)
    feature_columns = sorted(numeric_columns - NON_FEATURE_COLUMNS)
    if not feature_columns:
        raise ValueError("Feature table has no numeric model features")

    selected = dataset[["split", "rul_seconds", *feature_columns]].copy()
    selected[feature_columns] = selected[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    before = len(selected)
    selected = selected.dropna(subset=["rul_seconds", *feature_columns])
    dropped_rows = before - len(selected)

    training = selected[selected["split"] == train_split]
    evaluation = selected[selected["split"] == evaluation_split]
    if training.empty:
        raise ValueError(f"No usable rows found for train split {train_split!r}")
    if evaluation.empty:
        raise ValueError(
            f"No usable rows found for evaluation split {evaluation_split!r}"
        )

    return (
        training[feature_columns],
        evaluation[feature_columns],
        training["rul_seconds"],
        evaluation["rul_seconds"],
        dropped_rows,
    )
