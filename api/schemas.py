"""HTTP request and response contracts for model inference."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        min_length=1,
        description="Feature names and values matching the registered model signature.",
    )


class PredictionResponse(BaseModel):
    predicted_rul_seconds: float
    model_name: str
    model_version: str
    model_reference: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: str | None = None
