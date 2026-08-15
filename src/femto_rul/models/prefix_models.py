"""Compact estimators for prefix-level FEMTO RUL experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ConditionLifePriorRegressor(RegressorMixin, BaseEstimator):
    """Predict RUL from observed age and training total-life priors.

    Total life for a training pseudo-prefix is observed_age + target RUL.
    The model stores medians per operating condition using training bearings only.
    """

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


def prefix_estimators(random_state: int = 42) -> dict[str, Any]:
    """Return deliberately simple estimators for the small prefix sample set."""
    return {
        "condition_life_prior": ConditionLifePriorRegressor(),
        "ridge_prefix": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
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
