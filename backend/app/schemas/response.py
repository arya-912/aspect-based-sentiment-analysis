"""Response bodies."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TokenAttention(BaseModel):
    """One review token and the share of attention it received.

    WordPiece continuation pieces are folded back into their parent word, so
    ``token`` is a whole word rather than ``un`` / ``##responsive``.
    """

    token: str = Field(..., description="A token from the review segment.")
    weight: float = Field(
        ..., ge=0.0, le=1.0, description="Attention weight, summed over the word's pieces."
    )


class PredictResponse(BaseModel):
    """The sentiment expressed about one aspect."""

    aspect: str = Field(..., description="The aspect that was judged.")
    sentiment: str = Field(..., description="negative | neutral | positive")
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Softmax probability of the chosen label."
    )
    scores: dict[str, float] | None = Field(
        None, description="Softmax probability of every label, keyed by label name."
    )
    attention: list[TokenAttention] | None = Field(
        None,
        description=(
            "The most-attended review tokens, highest weight first. Null when "
            "the loaded model cannot report attention."
        ),
    )
    aspect_in_text: bool | None = Field(
        None,
        description=(
            "Whether the aspect's tokens occur verbatim in the review the "
            "model saw. False means the prediction is outside the regime the "
            "model was trained on -- every training aspect was a literal span "
            "of its sentence -- so the result is a guess worth distrusting. "
            "Null when it could not be determined. An aspect missing from the "
            "review altogether is refused with a 422 instead of reaching "
            "here, so False now narrows to the subtler cases: a stem match "
            "(\"batteries\" against \"battery\"), an occurrence truncation "
            "dropped, or REQUIRE_ASPECT_IN_TEXT turned off."
        ),
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
    corpus: str | None = Field(
        None,
        description=(
            "Stem of the training file the artifact was fitted on, e.g. "
            "'Laptop_Train_v2' or 'Restaurants_Train_v2'. Taken from the "
            "artifact's recorded paths, not from its version name, so a UI can "
            "label the domain and offer examples that belong to it. Null when "
            "the artifact records no training path."
        ),
    )
    evaluation: dict[str, Any] | None = Field(
        None, description="Headline metrics from the artifact's metrics.json."
    )
