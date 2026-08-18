"""FastAPI application serving the registered FEMTO RUL model."""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from api.model_loader import ModelService
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse


def _load_service() -> ModelService:
    return ModelService.load(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        model_name=os.getenv("API_MODEL_NAME", "femto-rul-model"),
        reference=os.getenv("API_MODEL_REFERENCE", "latest"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_service = _load_service()
        app.state.model_load_error = None
    except Exception as error:  # keep diagnostics reachable when MLflow is unavailable
        app.state.model_service = None
        app.state.model_load_error = f"{type(error).__name__}: {error}"
    yield


app = FastAPI(
    title="FEMTO RUL Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


def require_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    configured_key = os.getenv("API_KEY", "")
    if configured_key and (
        x_api_key is None or not secrets.compare_digest(x_api_key, configured_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )


def get_model_service(request: Request) -> ModelService:
    service = request.app.state.model_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=request.app.state.model_load_error or "Model is not loaded",
        )
    return service


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = request.app.state.model_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=HealthResponse(
                status="unhealthy",
                model_loaded=False,
                detail=request.app.state.model_load_error,
            ).model_dump(),
        )
    return HealthResponse(status="healthy", model_loaded=True)


@app.get("/model-info", dependencies=[Depends(require_api_key)])
def model_info(service: Annotated[ModelService, Depends(get_model_service)]):
    return service.info()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
def predict(
    payload: PredictionRequest,
    service: Annotated[ModelService, Depends(get_model_service)],
) -> PredictionResponse:
    try:
        prediction = service.predict(payload.features)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error

    return PredictionResponse(
        predicted_rul_seconds=prediction,
        model_name=service.model_name,
        model_version=service.version,
        model_reference=service.reference,
    )
