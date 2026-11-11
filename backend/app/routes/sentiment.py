"""Sentiment endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..core.config import get_settings
from ..core.errors import HTTP_422
from ..schemas.request import PredictRequest
from ..schemas.response import HealthResponse, ModelMetaResponse, PredictResponse
from ..services.aspect_match import aspect_in_text
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


def require_aspect_presence(request: Request) -> bool:
    """Whether this app refuses an aspect that is absent from the review.

    Reads the ``Settings`` the application factory was given, falling back to
    the process-wide ones for an app built some other way, so a test can flip
    the switch by constructing an app rather than patching a global.
    """
    settings = getattr(request.app.state, "settings", None) or get_settings()
    return bool(settings.REQUIRE_ASPECT_IN_TEXT)


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify the sentiment expressed about one aspect",
    responses={
        400: {"description": "The text or aspect could not be encoded."},
        422: {
            "description": (
                "The request body failed validation, or the aspect does not "
                "occur in the review text."
            )
        },
        503: {"description": "The model is not loaded."},
    },
)
def predict(
    payload: PredictRequest,
    predictor: PredictorService = Depends(get_predictor),
    enforce_presence: bool = Depends(require_aspect_presence),
) -> PredictResponse:
    """Predict the sentiment of ``payload.aspect`` within ``payload.text``.

    Refuses an aspect the review never mentions. The model is a softmax over
    three polarities with no way to abstain, so it answers such a request as
    confidently as any other -- and the answer is the sentence's overall
    sentiment wearing the aspect's name. A caller cannot tell the two apart
    from the response, so the question is rejected instead of answered.

    Checked before the model is touched: the gate is a string comparison, it
    needs no artifact, and failing here keeps a plainly bad request from
    turning into a 503 when the model happens to be unloaded.
    """
    if enforce_presence and not aspect_in_text(payload.text, payload.aspect):
        LOGGER.info("refused absent aspect %r", payload.aspect)
        raise HTTPException(
            status_code=HTTP_422,
            detail=(
                f"aspect {payload.aspect!r} does not occur in the review text; "
                "the model can only judge an aspect the review mentions"
            ),
        )
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
