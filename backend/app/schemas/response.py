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
    rating: float | None = Field(
        None,
        ge=0.0,
        le=5.0,
        description=(
            "Mean star rating from the uploaded CSV, when it carried one. "
            "This is the reviewers' own overall score, not the model's: it is "
            "shown beside the aspect breakdown precisely so the two can be "
            "compared. Null for the built-in demo catalogue, whose SemEval "
            "sentences have no star ratings."
        ),
    )
    rating_count: int = Field(
        0, description="How many reviews the star rating was averaged over."
    )
    reviews_seen: int = Field(
        0,
        description=(
            "Reviews this product had in the uploaded file before the "
            "per-product cap, so a shelf can say it is showing 15 of 187."
        ),
    )


class ProductDetail(ProductSummary):
    """A catalogue entry with its reviews and their aspect breakdown."""

    reviews: list[str]
    insights: InsightsResponse


class ProductAspectScore(BaseModel):
    """One product's standing on one aspect, for sorting the catalogue."""

    aspect: str
    positive_share: float = Field(..., ge=0.0, le=1.0)
    negative_share: float = Field(..., ge=0.0, le=1.0)
    total: int = Field(..., description="Sentences judged for this aspect.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Wilson 95% lower bound on the positive share -- the number to "
            "rank by. Ranking on positive_share alone puts one lucky review "
            "above fifty good ones: 1/1 and 45/50 are both 1.00 and 0.90, "
            "but their lower bounds are 0.21 and 0.79. Computed here rather "
            "than in the client so the shelf and any weighted ranking agree "
            "on one definition."
        ),
    )


class CatalogueInsights(BaseModel):
    """Every product scored on every aspect its category defines."""

    aspects: list[str] = Field(..., description="The aspects available to sort by.")
    products: dict[str, list[ProductAspectScore]] = Field(
        ...,
        description=(
            "Product slug -> its score on each aspect it is discussed on. A "
            "product absent from this map has not been scored yet; scoring "
            "runs in the background, so poll /catalogue/status for progress "
            "rather than assuming an empty map means no data."
        ),
    )
    model_version: str
    scored: int = Field(..., description="Products scored so far.")
    total: int = Field(..., description="Products in the active catalogue.")
    scoring: bool = Field(..., description="True while a background pass is running.")


class CatalogueStatus(BaseModel):
    """Which catalogue is active, and how far its scoring has got."""

    source: str = Field(..., description="'demo' or 'upload'.")
    source_name: str = Field(
        ...,
        description=(
            "The uploaded file's name, or the built-in catalogue's label. The "
            "UI states this, because a page showing an uploaded file's numbers "
            "under the demo's honesty note would be lying about its own data."
        ),
    )
    products: int
    scored: int = Field(..., description="Products whose breakdown is cached.")
    scoring: bool = Field(..., description="True while a background pass is running.")
    model_version: str
    rows_used: int = Field(
        0, description="Review rows kept from the upload; 0 for the demo catalogue."
    )
    products_found: int = Field(
        0,
        description=(
            "Products in the uploaded file before the cap, so the UI can say "
            "it is showing 24 of 2,526."
        ),
    )
    truncated_products: bool = False
    truncated_reviews: bool = False
    note: str = Field("", description="Any caveat worth printing verbatim.")


class CatalogueUploadResult(CatalogueStatus):
    """What an upload produced. Same shape as the status, plus the warnings."""

    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal observations: caps applied, ratings missing, and so on.",
    )
