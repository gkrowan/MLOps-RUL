"""Baseline regressors and the production-safe training data contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MedianRULRegressor(RegressorMixin, BaseEstimator):
    """A no-feature baseline that always predicts the training median RUL."""

    def fit(self, X: Any, y: Any) -> "MedianRULRegressor":
        target = np.asarray(y, dtype=float)
        if target.size == 0:
            raise ValueError("cannot fit median baseline on an empty target")
        self.median_rul_ = float(np.median(target))
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "median_rul_"):
            raise RuntimeError("MedianRULRegressor must be fitted before predict")
        return np.full(len(X), self.median_rul_, dtype=float)


def baseline_estimators(random_state: int = 42) -> dict[str, Any]:
    """Return deterministic baseline models using only scikit-learn."""
    return {
        "median": MedianRULRegressor(),
        "ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }


def validate_training_contract(
    frame: pd.DataFrame,
    schema: dict[str, Any],
) -> tuple[list[str], str, str]:
    """Validate schema-driven model inputs and return feature/target/group names."""
    feature_columns = list(schema.get("default_model_feature_columns", []))
    blocked_columns = set(schema.get("blocked_predictor_columns", []))
    target_column = str(schema.get("target_column", "rul_seconds"))
    group_column = "bearing"

    if not feature_columns:
        raise ValueError("feature schema has no default_model_feature_columns")

    leaked = blocked_columns.intersection(feature_columns)
    if leaked:
        raise ValueError(f"blocked predictors present in model features: {sorted(leaked)}")

    required = set(feature_columns) | {target_column, group_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"training data missing required columns: {sorted(missing)}")

    if frame[feature_columns].isna().any().any():
        raise ValueError("training model features contain missing values")
    if frame[target_column].isna().any():
        raise ValueError("training target contains missing values")
    if frame[group_column].isna().any():
        raise ValueError("training bearing/group column contains missing values")

    if not np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("training model features contain non-finite values")
    if not np.isfinite(frame[target_column].to_numpy(dtype=float)).all():
        raise ValueError("training target contains non-finite values")

    if frame[group_column].nunique() < 2:
        raise ValueError("LOBO training requires at least two bearings")

    return feature_columns, target_column, group_column


def load_training_data(
    train_features_path: Path,
    feature_schema_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Load train_features + schema and return safe X/y/groups objects."""
    if not train_features_path.is_file():
        raise FileNotFoundError(f"missing training features: {train_features_path}")
    if not feature_schema_path.is_file():
        raise FileNotFoundError(f"missing feature schema: {feature_schema_path}")

    frame = pd.read_parquet(train_features_path)
    schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
    feature_columns, target_column, group_column = validate_training_contract(frame, schema)

    X = frame[feature_columns].copy()
    y = frame[target_column].astype(float).copy()
    groups = frame[group_column].astype(str).copy()
    return X, y, groups, schema
