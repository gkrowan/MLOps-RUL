import pandas as pd
import pytest

from femto_rul.models.baselines import validate_training_contract


SCHEMA = {
    "default_model_feature_columns": ["f1", "f2"],
    "blocked_predictor_columns": [
        "split",
        "bearing",
        "elapsed_time_seconds",
        "file_index",
        "rul_seconds",
    ],
    "target_column": "rul_seconds",
}


def _frame():
    return pd.DataFrame(
        {
            "bearing": ["B1", "B1", "B2", "B2"],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [4.0, 3.0, 2.0, 1.0],
            "rul_seconds": [30.0, 20.0, 10.0, 0.0],
        }
    )


def test_training_contract_accepts_schema_features_only():
    features, target, group = validate_training_contract(_frame(), SCHEMA)
    assert features == ["f1", "f2"]
    assert target == "rul_seconds"
    assert group == "bearing"


def test_training_contract_rejects_leakage_column_in_features():
    schema = dict(SCHEMA)
    schema["default_model_feature_columns"] = ["f1", "rul_seconds"]
    with pytest.raises(ValueError, match="blocked predictors"):
        validate_training_contract(_frame(), schema)
