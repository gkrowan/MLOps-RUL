"""Compact estimators for prefix-level FEMTO RUL experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ConditionLifePriorRegressor(RegressorMixin, BaseEstimator):
    """Predict RUL from observed age and training total-life priors."""

    def fit(self, X: Any, y: Any) -> "ConditionLifePriorRegressor":
        condition = np.asarray(X["condition"], dtype=int)
        age = np.asarray(X["observed_age_seconds"], dtype=float)
        target = np.asarray(y, dtype=float)
        total_life = age + target

        self.global_total_life_ = float(np.median(total_life))
        self.condition_total_life_ = {
            int(c): float(np.median(total_life[condition == c]))
            for c in np.unique(condition)
        }
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "global_total_life_"):
            raise RuntimeError("ConditionLifePriorRegressor must be fitted before predict")
        condition = np.asarray(X["condition"], dtype=int)
        age = np.asarray(X["observed_age_seconds"], dtype=float)
        life = np.array(
            [self.condition_total_life_.get(int(c), self.global_total_life_) for c in condition],
            dtype=float,
        )
        return np.maximum(life - age, 0.0)


class TotalLifeToRULRegressor(RegressorMixin, BaseEstimator):
    """Fit an estimator to total bearing life, then convert back to RUL.

    The public fit target remains RUL. During fit, total life is computed only
    from Training_set prefix age + Training_set RUL. At prediction time the
    estimator predicts total life and known observed age is subtracted.
    """

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> "TotalLifeToRULRegressor":
        if "observed_age_seconds" not in X:
            raise ValueError("TotalLifeToRULRegressor requires observed_age_seconds")
        age = np.asarray(X["observed_age_seconds"], dtype=float)
        rul = np.asarray(y, dtype=float)
        total_life = age + rul
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, total_life)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("TotalLifeToRULRegressor must be fitted before predict")
        age = np.asarray(X["observed_age_seconds"], dtype=float)
        predicted_total_life = np.asarray(self.estimator_.predict(X), dtype=float)
        return np.maximum(predicted_total_life - age, 0.0)


def _direct_rf(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def prefix_target_estimators(random_state: int = 42) -> dict[str, Any]:
    """Models for direct-RUL vs total-life target ablation.

    This is still a controlled baseline comparison, not hyperparameter tuning.
    """
    return {
        "condition_life_prior": ConditionLifePriorRegressor(),
        "rf_direct_rul": _direct_rf(random_state),
        "ridge_total_life": TotalLifeToRULRegressor(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=10.0)),
                ]
            )
        ),
        "knn_total_life": TotalLifeToRULRegressor(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors=3, weights="distance")),
                ]
            )
        ),
        "rf_total_life": TotalLifeToRULRegressor(_direct_rf(random_state)),
        "extra_trees_total_life": TotalLifeToRULRegressor(
            ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            )
        ),
    }


# Backward-compatible Phase 9 estimator set.
def prefix_estimators(random_state: int = 42) -> dict[str, Any]:
    return {
        "condition_life_prior": ConditionLifePriorRegressor(),
        "ridge_prefix": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]
        ),
        "knn_prefix": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=3, weights="distance")),
            ]
        ),
        "random_forest_prefix": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
