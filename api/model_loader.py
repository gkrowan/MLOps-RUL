"""Load a registered MLflow model and enforce its named input signature."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from api.model_registry import resolve_model_version


@dataclass(frozen=True)
class ModelService:
    model: Any
    model_name: str
    reference: str
    version: str
    run_id: str
    model_uri: str
    input_names: tuple[str, ...]

    @classmethod
    def load(
        cls, *, tracking_uri: str, model_name: str, reference: str
    ) -> "ModelService":
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        version = resolve_model_version(MlflowClient(), model_name, reference)
        model_uri = f"models:/{model_name}/{version.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        schema = model.metadata.get_input_schema()
        input_names = tuple(schema.input_names()) if schema is not None else ()
        if not input_names:
            raise RuntimeError(
                f"Model {model_uri} has no named input signature; refusing to serve it"
            )
        return cls(
            model=model,
            model_name=model_name,
            reference=reference,
            version=str(version.version),
            run_id=str(version.run_id),
            model_uri=model_uri,
            input_names=input_names,
        )

    def predict(self, features: dict[str, float]) -> float:
        expected = set(self.input_names)
        received = set(features)
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing features: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected features: {', '.join(extra)}")
            raise ValueError("; ".join(details))

        invalid = [name for name, value in features.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"non-finite feature values: {', '.join(sorted(invalid))}")

        frame = pd.DataFrame(
            [[features[name] for name in self.input_names]], columns=self.input_names
        )
        prediction = float(self.model.predict(frame)[0])
        if not math.isfinite(prediction):
            raise RuntimeError("Model returned a non-finite prediction")
        return max(0.0, prediction)

    def info(self) -> dict[str, str | list[str]]:
        return {
            "model_name": self.model_name,
            "reference": self.reference,
            "version": self.version,
            "run_id": self.run_id,
            "model_uri": self.model_uri,
            "input_features": list(self.input_names),
        }
