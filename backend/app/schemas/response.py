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


class AspectExample(BaseModel):
    """One sentence, and the model's verdict on one aspect of it."""

    sentence: str
    surface: str = Field(
        ...,
        description=(
            "The aspect exactly as this sentence spells it. May differ in "
            "inflection from the aspect's label -- 'batteries' grouped under "
            "'battery' -- because the model is asked about the text that is "
            "really in the sentence."
        ),
    )
    sentiment: str
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    review_index: int = Field(..., description="Which review in the request this came from.")


class AspectSummaryResponse(BaseModel):
    """What a review set says about one aspect."""

    aspect: str
    source: str = Field(
        ...,
        description=(
            "Where the aspect came from, never from the model: 'corpus' (a "
            "SemEval gold term annotated 3+ times), 'corpus-rare' (annotated "
            "once or twice), or 'given' (named by the caller)."
        ),
    )
    total: int = Field(..., description="Sentences judged for this aspect.")
    counts: dict[str, int] = Field(..., description="Sentence count per label.")
    shares: dict[str, float] = Field(..., description="Share of sentences per label.")
    dominant: str
    mean_confidence: float = Field(..., ge=0.0, le=1.0)
    mixed: bool = Field(
        ..., description="True when this aspect drew more than one polarity."
    )
    examples: list[AspectExample]


class InsightsResponse(BaseModel):
    """Per-aspect sentiment over a set of reviews."""

    reviews: int
    sentences: int
    predictions: int = Field(..., description="Forward passes actually made.")
    aspects: list[AspectSummaryResponse]
    mixed_sentences: int = Field(
        ...,
        description=(
            "Sentences where two aspects drew different polarities. Each one "
            "is a sentence a single-label classifier is necessarily wrong "
            "about, which is the case for doing this per aspect."
        ),
    )
    mixed_aspects: int
    truncated: bool = Field(
        ...,
        description="True when the request hit the forward-pass ceiling and some occurrences were not judged.",
    )
    model_version: str


class ProductSummary(BaseModel):
    """A catalogue entry, without its reviews."""

    slug: str
    name: str
    category: str = Field(..., description="'laptop' or 'restaurant'.")
    subtitle: str
    price: str
    review_count: int


class ProductDetail(ProductSummary):
    """A catalogue entry with its reviews and their aspect breakdown."""

    reviews: list[str]
    insights: InsightsResponse


class ProductAspectScore(BaseModel):
    """One product's standing on one aspect, for sorting the catalogue."""

    aspect: str
    positive_share: float = Field(..., ge=0.0, le=1.0)
    negative_share: float = Field(..., ge=0.0, le=1.0)
    total: int


class CatalogueInsights(BaseModel):
    """Every product scored on every aspect its category defines."""

    aspects: list[str] = Field(..., description="The aspects available to sort by.")
    products: dict[str, list[ProductAspectScore]] = Field(
        ..., description="Product slug -> its score on each aspect it is discussed on."
    )
    model_version: str
