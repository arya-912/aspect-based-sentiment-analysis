"""Response bodies."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """The sentiment expressed about one aspect."""

    aspect: str = Field(..., description="The aspect that was judged.")
    sentiment: str = Field(..., description="negative | neutral | positive")
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Softmax probability of the chosen label."
    )
    model_version: str = Field(..., description="Artifact version that served this.")


class HealthResponse(BaseModel):
    """Liveness and model readiness.

    Always returns 200: a caller needs to distinguish "the process is up but
    the model is still loading" from "the process is unreachable", and a 503
    here would collapse those two cases.
    """

    status: Literal["ok", "degraded"]
    model_loaded: bool


class ModelMetaResponse(BaseModel):
    """What model is being served, and how well it scored."""

    version: str
    base_model: str
    task: str = "aspect-based-sentiment-analysis"
    labels: list[str]
    evaluation: dict[str, Any] | None = Field(
        None, description="Headline metrics from the artifact's metrics.json."
    )
