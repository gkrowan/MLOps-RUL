"""
FastAPI service for serving the FEMTO bearing RUL model.

Loads the current "Production" (or aliased) model version from the MLflow
Model Registry and exposes it for real-time inference. Input is a vector
of pre-extracted time-domain and frequency-domain features, NOT raw
vibration signal.

Run locally (from the repo root, with this file at api/main.py):
    uvicorn api.main:app --reload --port 8000

Env vars expected:
    MLFLOW_TRACKING_URI   e.g. http://127.0.0.1:5000 (local or tunneled remote)
    MODEL_NAME            e.g. "femto-rul"
    MODEL_STAGE           e.g. "Production" (or use an alias like "champion")

If MLflow's artifact store is S3/MinIO, also make sure these are set so
the model file itself (not just the registry metadata) can be fetched:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    MLFLOW_S3_ENDPOINT_URL   e.g. http://127.0.0.1:9000 (MinIO's S3 API port,
                              NOT 9001, which is the MinIO web console)
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("femto-rul-api")

MODEL_NAME = os.getenv("MODEL_NAME", "femto-rul")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Holds the loaded model + metadata so /predict and /model-info can share it
model_state = {"model": None, "version": None, "run_id": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the model once, not on every request
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info(f"Loading model from {model_uri}")

    model_state["model"] = mlflow.pyfunc.load_model(model_uri)

    # Grab version metadata for /model-info and monitoring provenance
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if versions:
        model_state["version"] = versions[0].version
        model_state["run_id"] = versions[0].run_id

    logger.info(f"Loaded {MODEL_NAME} v{model_state['version']}")
    yield
    # Shutdown: nothing to clean up currently
    model_state["model"] = None


app = FastAPI(
    title="FEMTO Bearing RUL Prediction API",
    description="Serves Remaining Useful Life predictions from vibration-derived features",
    version="0.1.0",
    lifespan=lifespan,
)


# ---- Schemas -----------------------------------------------------------
# Matches the real output of src/femto_rul/pipeline.py's build_bearing_dataset
# (confirmed against data/processed/features.parquet, 2026-08-14):
# 5 metadata cols + 12 features per channel (horiz, vert) + rul_seconds label.
# Metadata columns (split, condition, bearing, elapsed_time_seconds,
# file_index) and the rul_seconds label are NOT part of the request body —
# only the 24 feature columns are, since those are what the model was
# trained on. If the feature pipeline changes, update this class to match.

class BearingFeatures(BaseModel):
    rms_horiz: float = Field(..., description="RMS of horizontal channel")
    kurtosis_horiz: float = Field(..., description="Kurtosis of horizontal channel")
    skewness_horiz: float = Field(..., description="Skewness of horizontal channel")
    crest_factor_horiz: float = Field(..., description="Peak amplitude / RMS, horizontal channel")
    fft_band_0_horiz: float = Field(..., description="FFT band 0 energy, horizontal channel")
    fft_band_1_horiz: float = Field(..., description="FFT band 1 energy, horizontal channel")
    fft_band_2_horiz: float = Field(..., description="FFT band 2 energy, horizontal channel")
    fft_band_3_horiz: float = Field(..., description="FFT band 3 energy, horizontal channel")
    fft_band_4_horiz: float = Field(..., description="FFT band 4 energy, horizontal channel")
    fft_band_5_horiz: float = Field(..., description="FFT band 5 energy, horizontal channel")
    fft_band_6_horiz: float = Field(..., description="FFT band 6 energy, horizontal channel")
    fft_band_7_horiz: float = Field(..., description="FFT band 7 energy, horizontal channel")

    rms_vert: float = Field(..., description="RMS of vertical channel")
    kurtosis_vert: float = Field(..., description="Kurtosis of vertical channel")
    skewness_vert: float = Field(..., description="Skewness of vertical channel")
    crest_factor_vert: float = Field(..., description="Peak amplitude / RMS, vertical channel")
    fft_band_0_vert: float = Field(..., description="FFT band 0 energy, vertical channel")
    fft_band_1_vert: float = Field(..., description="FFT band 1 energy, vertical channel")
    fft_band_2_vert: float = Field(..., description="FFT band 2 energy, vertical channel")
    fft_band_3_vert: float = Field(..., description="FFT band 3 energy, vertical channel")
    fft_band_4_vert: float = Field(..., description="FFT band 4 energy, vertical channel")
    fft_band_5_vert: float = Field(..., description="FFT band 5 energy, vertical channel")
    fft_band_6_vert: float = Field(..., description="FFT band 6 energy, vertical channel")
    fft_band_7_vert: float = Field(..., description="FFT band 7 energy, vertical channel")

    class Config:
        json_schema_extra = {
            "example": {
                "rms_horiz": 0.42, "kurtosis_horiz": 3.1, "skewness_horiz": 0.05,
                "crest_factor_horiz": 4.7,
                "fft_band_0_horiz": 12.3, "fft_band_1_horiz": 8.9, "fft_band_2_horiz": 3.4,
                "fft_band_3_horiz": 2.1, "fft_band_4_horiz": 1.5, "fft_band_5_horiz": 0.9,
                "fft_band_6_horiz": 0.6, "fft_band_7_horiz": 0.3,
                "rms_vert": 0.38, "kurtosis_vert": 2.9, "skewness_vert": -0.02,
                "crest_factor_vert": 4.3,
                "fft_band_0_vert": 11.1, "fft_band_1_vert": 7.6, "fft_band_2_vert": 3.0,
                "fft_band_3_vert": 1.9, "fft_band_4_vert": 1.3, "fft_band_5_vert": 0.8,
                "fft_band_6_vert": 0.5, "fft_band_7_vert": 0.25,
            }
        }


class RULPrediction(BaseModel):
    predicted_rul_seconds: float
    model_version: Optional[str]
    model_run_id: Optional[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---- Routes -------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness/readiness check — also useful for your monitoring dashboard."""
    return HealthResponse(
        status="ok",
        model_loaded=model_state["model"] is not None,
    )


@app.get("/model-info")
def model_info():
    """Confirms which registered model version is currently serving."""
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": MODEL_NAME,
        "stage": MODEL_STAGE,
        "version": model_state["version"],
        "run_id": model_state["run_id"],
    }


@app.post("/predict", response_model=RULPrediction)
def predict(features: BearingFeatures):
    """
    Accepts a single feature vector and returns predicted RUL.

    Pydantic validation here is your first line of defense for the
    drift-simulation deliverable: malformed schema (missing/extra fields,
    wrong types) gets rejected with a 422 before it ever reaches the model
    or your monitoring layer. Values that are structurally valid but
    out-of-distribution (e.g. absurd RMS spikes) will pass through here
    and should be caught downstream by Evidently instead.
    """
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([features.model_dump()])
        prediction = model_state["model"].predict(input_df)
        predicted_value = float(prediction[0])
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return RULPrediction(
        predicted_rul_seconds=predicted_value,
        model_version=model_state["version"],
        model_run_id=model_state["run_id"],
    )