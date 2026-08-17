"""
FastAPI service for serving the FEMTO bearing RUL champion model.

The service loads the current MLflow Model Registry alias (default: champion)
once at startup and exposes real-time RUL inference.

Input:
    Prefix V1 pre-extracted features used by the registered model.
    Raw vibration signals are NOT accepted by this endpoint.

Run locally from the repository root:
    uvicorn api.main:app --reload --port 8000

Existing environment variables reused:
    MLFLOW_TRACKING_URI
    MLFLOW_MODEL_NAME
    MODEL_NAME
    MODEL_ALIAS
    MINIO_ENDPOINT_URL
    MINIO_ROOT_USER
    MINIO_ROOT_PASSWORD

No .env change is required for the current local setup.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, create_model

# Importing project config also preserves the project's existing .env loading.
from femto_rul.config import MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI
from femto_rul.features.prefix import prefix_feature_columns
from femto_rul.serving.telemetry import log_prediction


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("femto-rul-api")


# -----------------------------------------------------------------------------
# Runtime configuration
# -----------------------------------------------------------------------------

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    os.getenv("MLFLOW_MODEL_NAME", MLFLOW_MODEL_NAME),
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion",
)

# Reuse the project's existing MinIO variables for MLflow's S3-compatible
# artifact client. setdefault() preserves explicitly supplied AWS/MLflow vars.
MINIO_ENDPOINT_URL = os.getenv(
    "MINIO_ENDPOINT_URL",
    "http://localhost:9000",
)

os.environ.setdefault(
    "MLFLOW_S3_ENDPOINT_URL",
    MINIO_ENDPOINT_URL,
)

if os.getenv("MINIO_ROOT_USER"):
    os.environ.setdefault(
        "AWS_ACCESS_KEY_ID",
        os.environ["MINIO_ROOT_USER"],
    )

if os.getenv("MINIO_ROOT_PASSWORD"):
    os.environ.setdefault(
        "AWS_SECRET_ACCESS_KEY",
        os.environ["MINIO_ROOT_PASSWORD"],
    )


# -----------------------------------------------------------------------------
# Model state
# -----------------------------------------------------------------------------

model_state = {
    "model": None,
    "version": None,
    "run_id": None,
    "model_uri": None,
}


# -----------------------------------------------------------------------------
# Model lifecycle
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the registered champion once when the API starts."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    logger.info("Loading model from %s", model_uri)

    try:
        model = mlflow.pyfunc.load_model(model_uri)

        model_version = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS,
        )

        model_state["model"] = model
        model_state["version"] = model_version.version
        model_state["run_id"] = model_version.run_id
        model_state["model_uri"] = model_uri

        logger.info(
            "Loaded model=%s version=%s alias=%s",
            MODEL_NAME,
            model_version.version,
            MODEL_ALIAS,
        )

    except Exception:
        logger.exception(
            "Failed to load MLflow model %s",
            model_uri,
        )
        raise

    yield

    model_state["model"] = None


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="FEMTO Bearing RUL Prediction API",
    description=(
        "Serves Remaining Useful Life predictions using the "
        "registered MLflow champion model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# -----------------------------------------------------------------------------
# Request / response schemas
# -----------------------------------------------------------------------------

# The API contract comes directly from the production Prefix V1 feature
# definition used by the registered model. This prevents a second hard-coded
# feature list from drifting away from training.
EXPECTED_FEATURES = prefix_feature_columns()

BearingFeatures = create_model(
    "BearingFeatures",
    __config__=ConfigDict(extra="forbid"),
    **{
        feature_name: (float, ...)
        for feature_name in EXPECTED_FEATURES
    },
)


class RULPrediction(BaseModel):
    predicted_rul_seconds: float
    model_name: str
    model_alias: str
    model_version: Optional[str]
    model_run_id: Optional[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness/readiness endpoint."""

    loaded = model_state["model"] is not None

    return HealthResponse(
        status="ok" if loaded else "not_ready",
        model_loaded=loaded,
    )


@app.get("/model-info")
def model_info():
    """Return metadata for the registered model currently being served."""

    if model_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": model_state["version"],
        "run_id": model_state["run_id"],
        "model_uri": model_state["model_uri"],
        "feature_count": len(EXPECTED_FEATURES),
        "features": EXPECTED_FEATURES,
    }


def _log_prediction_safely(**kwargs) -> None:
    """Best-effort telemetry write (Phase 16) — never let a logging failure
    fail the prediction response."""
    try:
        log_prediction(
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
            model_version=str(model_state["version"]),
            feature_set_version="prefix_v1",
            **kwargs,
        )
    except Exception:
        logger.exception("prediction logging failed")


@app.post("/predict", response_model=RULPrediction)
def predict(features: BearingFeatures):
    """
    Predict Remaining Useful Life in seconds.

    The request must contain the exact Prefix V1 feature set expected by the
    registered model. Missing or additional fields are rejected by Pydantic.
    """

    if model_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    request_id = str(uuid.uuid4())
    payload = features.model_dump()
    start = time.perf_counter()

    try:
        # Explicit feature ordering protects inference from request-key order.
        input_df = pd.DataFrame(
            [[payload[name] for name in EXPECTED_FEATURES]],
            columns=EXPECTED_FEATURES,
        )

        prediction = model_state["model"].predict(input_df)

        # RUL cannot be negative.
        predicted_value = max(
            0.0,
            float(prediction[0]),
        )

    except Exception as exc:
        logger.exception("Prediction failed")

        _log_prediction_safely(
            request_id=request_id,
            features=None,
            predicted_rul_seconds=None,
            latency_ms=(time.perf_counter() - start) * 1000,
            status="error",
            error_message=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc

    _log_prediction_safely(
        request_id=request_id,
        features=payload,
        predicted_rul_seconds=predicted_value,
        latency_ms=(time.perf_counter() - start) * 1000,
        status="ok",
        error_message=None,
    )

    return RULPrediction(
        predicted_rul_seconds=predicted_value,
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        model_version=model_state["version"],
        model_run_id=model_state["run_id"],
    )
