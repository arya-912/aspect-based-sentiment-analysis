"""Aspect-extraction, review-set insights, and the demo storefront.

Everything here is built on one endpoint's worth of model: a batch of
(sentence, aspect) pairs. What is new is the step *before* the model -- 
deciding what the aspects are -- which the model does not do and which these
routes are explicit about, reporting a ``source`` for every aspect so a
client can say where it came from.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..core.errors import HTTP_422
from ..schemas.request import InsightsRequest
from ..schemas.response import (
    CatalogueInsights,
    InsightsResponse,
    ProductDetail,
    ProductSummary,
)
from ..services import catalogue
from ..services.insights import summarise
from ..services.predictor import PredictorService, get_predictor

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["insights"])


@router.post(
    "/insights",
    response_model=InsightsResponse,
    status_code=status.HTTP_200_OK,
    summary="Break a set of reviews down by aspect",
    responses={
        422: {"description": "The request body failed validation."},
        503: {"description": "The model is not loaded."},
    },
)
def review_insights(
    payload: InsightsRequest,
    predictor: PredictorService = Depends(get_predictor),
) -> InsightsResponse:
    """Per-aspect sentiment across the reviews in the body.

    This is the thing a sentence-level classifier cannot produce.
    *"Battery life is amazing but the screen is dim"* has no single correct
    label; it has two aspects with opposite ones. ``mixed_sentences`` counts
    how many sentences in the caller's own data are like that.

    Aspects come from the corpora's gold vocabulary unless the caller names
    them, and every aspect reports its ``source``. The model contributes no
    part of that choice -- it only judges pairs it is handed.

    An empty ``aspects`` list is a valid answer, not an error: reviews about
    something SemEval never annotated genuinely have no aspects this can
    find.
    """
    summary = summarise(
        payload.reviews,
        predictor=predictor,
        aspects=payload.aspects,
        domain=payload.domain,
        max_aspects=payload.max_aspects,
    )
    return catalogue.to_response(summary)


@router.get(
    "/products",
    response_model=list[ProductSummary],
    summary="The demo storefront's catalogue",
)
def list_products() -> list[ProductSummary]:
    """Catalogue metadata only.

    Deliberately free of predictions, so the shelf renders immediately; the
    aspect scores are a second request the page makes alongside it.
    """
    return catalogue.summaries()


# Declared before /products/{slug}: FastAPI matches in order, and a dynamic
# segment would otherwise swallow "insights" as a product slug.
@router.get(
    "/products/insights",
    response_model=CatalogueInsights,
    summary="Every product's score on every aspect, for sorting the shelf",
    responses={503: {"description": "The model is not loaded."}},
)
def all_product_insights(
    predictor: PredictorService = Depends(get_predictor),
) -> CatalogueInsights:
    """Aspect scores for the whole catalogue.

    The first call computes every product's breakdown -- a couple of hundred
    forward passes, measured at ~8s on CPU -- and the result is cached for
    the life of the process. Clients should render the shelf from
    ``/products`` first and treat this as a slow follow-up.
    """
    return catalogue.catalogue_insights(predictor)


@router.get(
    "/products/{slug}",
    response_model=ProductDetail,
    summary="One product, its reviews, and its aspect breakdown",
    responses={
        404: {"description": "No such product."},
        503: {"description": "The model is not loaded."},
    },
)
def product_detail(
    slug: str,
    predictor: PredictorService = Depends(get_predictor),
) -> ProductDetail:
    """One product page's worth of data."""
    product = catalogue.find(slug)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no such product: {slug!r}"
        )
    return catalogue.detail(product, predictor)


@router.get(
    "/aspects",
    response_model=list[str],
    summary="Candidate aspects in a single text, without judging them",
)
def candidate_aspects(
    text: str = Query(..., min_length=1, max_length=5000),
    domain: str | None = Query(None, description="'laptop' or 'restaurant'."),
    limit: int = Query(12, ge=1, le=24),
) -> list[str]:
    """Extract aspects and return only their names.

    Needs no model at all -- it is lexicon matching on strings -- so it
    answers while the artifact is still loading, and lets a UI show the
    chips it is about to analyse before paying for any predictions.
    """
    if not text.strip():
        raise HTTPException(status_code=HTTP_422, detail="text must not be blank")

    from ..services.aspects import extract_aspects

    return [
        candidate.aspect
        for candidate in extract_aspects([text], domain=domain, limit=limit)
    ]
