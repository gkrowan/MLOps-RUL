import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from femto_rul.modeling.training import build_regressor, prepare_feature_splits


def test_build_regressor_merges_hyperparameter_overrides() -> None:
    model, parameters = build_regressor(
        "random_forest", {"n_estimators": 17, "max_depth": 4}
    )

    assert isinstance(model, RandomForestRegressor)
    assert parameters["n_estimators"] == 17
    assert parameters["max_depth"] == 4
    assert parameters["random_state"] == 42


def test_prepare_feature_splits_excludes_metadata_and_drops_invalid_rows() -> None:
    dataset = pd.DataFrame(
        {
            "split": ["Training_set", "Training_set", "Validation_Set"],
            "bearing": ["Bearing1_1", "Bearing1_1", "Bearing1_3"],
            "file_index": [1, 2, 1],
            "elapsed_time_seconds": [0, 10, 0],
            "condition": [1, 1, 1],
            "rms_horiz": [0.1, None, 0.3],
            "rul_seconds": [20, 10, 30],
        }
    )

    x_train, x_eval, y_train, y_eval, dropped = prepare_feature_splits(
        dataset,
        train_split="Training_set",
        evaluation_split="Validation_Set",
    )

    assert list(x_train.columns) == ["condition", "rms_horiz"]
    assert len(x_train) == len(y_train) == 1
    assert len(x_eval) == len(y_eval) == 1
    assert dropped == 1


def test_prepare_feature_splits_requires_requested_splits() -> None:
    dataset = pd.DataFrame(
        {"split": ["Training_set"], "rms_horiz": [0.1], "rul_seconds": [20]}
    )

    with pytest.raises(ValueError, match="evaluation split"):
        prepare_feature_splits(
            dataset,
            train_split="Training_set",
            evaluation_split="Validation_Set",
        )
