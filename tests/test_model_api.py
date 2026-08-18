import pytest

from api.model_loader import ModelService


class FakeModel:
    def predict(self, frame):
        assert list(frame.columns) == ["condition", "rms_horiz"]
        return [frame.iloc[0]["condition"] + frame.iloc[0]["rms_horiz"]]


def _service() -> ModelService:
    return ModelService(
        model=FakeModel(),
        model_name="femto-rul-model",
        reference="latest",
        version="3",
        run_id="run-123",
        model_uri="models:/femto-rul-model/3",
        input_names=("condition", "rms_horiz"),
    )


def test_predict_orders_features_by_model_signature() -> None:
    prediction = _service().predict({"rms_horiz": 0.25, "condition": 1.0})

    assert prediction == pytest.approx(1.25)


def test_predict_rejects_schema_mismatch() -> None:
    with pytest.raises(ValueError, match="missing features: rms_horiz"):
        _service().predict({"condition": 1.0, "unexpected": 3.0})
