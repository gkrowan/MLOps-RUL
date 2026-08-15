"""Canonical direct-RUL model factory.

Every model in this module is evaluated on the same prefix dataset and LOBO
protocol. Optional third-party estimators are imported lazily so the core test
suite remains usable before the modeling extras are installed.
"""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from femto_rul.models.prefix_models import ConditionLifePriorRegressor


class MissingModelDependency(RuntimeError):
    pass


def _xgboost(params: dict[str, Any], random_state: int) -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise MissingModelDependency(
            "xgboost is not installed; run: pip install -r requirements-modeling.txt"
        ) from exc
    return XGBRegressor(random_state=random_state, n_jobs=-1, **params)


def _lightgbm(params: dict[str, Any], random_state: int) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise MissingModelDependency(
            "lightgbm is not installed; run: pip install -r requirements-modeling.txt"
        ) from exc
    return LGBMRegressor(random_state=random_state, n_jobs=-1, **params)


def make_model(
    model_name: str,
    *,
    defaults: dict[str, dict[str, Any]],
    random_state: int,
    overrides: dict[str, Any] | None = None,
) -> Any:
    overrides = dict(overrides or {})
    base = dict(defaults.get(model_name, {}))
    base.update(overrides)

    if model_name == "condition_life_prior":
        return ConditionLifePriorRegressor()
    if model_name == "median":
        return DummyRegressor(strategy="median")
    if model_name == "ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(**base)),
            ]
        )
    if model_name == "knn":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(**base)),
            ]
        )
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **base)
    if model_name == "extra_trees":
        return ExtraTreesRegressor(random_state=random_state, n_jobs=-1, **base)
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=random_state, **base)
    if model_name == "xgboost":
        return _xgboost(base, random_state)
    if model_name == "lightgbm":
        return _lightgbm(base, random_state)
    raise KeyError(f"unsupported model: {model_name}")


def effective_model_params(model_name: str, defaults: dict[str, dict[str, Any]], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    values = dict(defaults.get(model_name, {}))
    values.update(overrides or {})
    return values
