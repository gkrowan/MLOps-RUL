from femto_rul.features.schema import feature_schema


def test_v2_schema_keeps_leakage_out_of_default_model_features():
    schema = feature_schema()
    model_features = set(schema["default_model_feature_columns"])
    blocked = set(schema["blocked_predictor_columns"])

    assert schema["feature_set_version"] == "v2"
    assert len(schema["signal_feature_columns"]) == 24
    assert len(schema["temporal_feature_columns"]) == 54
    assert len(schema["default_model_feature_columns"]) == 78
    assert not model_features.intersection(blocked)
    assert "elapsed_time_seconds" not in model_features
    assert "file_index" not in model_features
    assert "rul_seconds" not in model_features
