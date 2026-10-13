"""Sentiment endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, status

from ..schemas.request import PredictRequest
from ..schemas.response import HealthResponse, ModelMetaResponse, PredictResponse
from ..services.model_registry import registry
from ..services.predictor import PredictorService, get_predictor

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["sentiment"])

#: Metric keys surfaced by /model. The full report stays in the artifact --
#: the per-example predictions in metrics.json are not something to serve.
EVALUATION_KEYS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "n",
    "evaluation_split",
    "per_class",
    "confusion_matrix",
)


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify the sentiment expressed about one aspect",
    responses={
        400: {"description": "The text or aspect could not be encoded."},
        422: {"description": "The request body failed validation."},
        503: {"description": "The model is not loaded."},
    },
)
def predict(
    payload: PredictRequest,
    predictor: PredictorService = Depends(get_predictor),
) -> PredictResponse:
    """Predict the sentiment of ``payload.aspect`` within ``payload.text``."""
    return predictor.predict(payload.text, payload.aspect)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and model readiness",
)
def health() -> HealthResponse:
    """Report whether the service is up and whether the model is loaded.

    Always 200. A caller needs to tell "process up, model still loading" apart
    from "process unreachable", and answering 503 here would merge the two.
    """
    loaded = registry.is_loaded()
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@router.get(
    "/model",
    response_model=ModelMetaResponse,
    summary="Which model is being served, and how it scored",
    responses={503: {"description": "The model is not loaded."}},
)
def model_metadata() -> ModelMetaResponse:
    """Describe the loaded artifact, including its evaluation metrics."""
    registry.get_config()  # raises ModelNotReadyError -> 503 when absent

    metrics = registry.read_metrics()
    evaluation: dict[str, Any] | None = None
    if metrics:
        evaluation = {key: metrics[key] for key in EVALUATION_KEYS if key in metrics}

    return ModelMetaResponse(
        version=registry.model_version,
        base_model=registry.base_model,
        labels=registry.labels,
        evaluation=evaluation or None,
    )
