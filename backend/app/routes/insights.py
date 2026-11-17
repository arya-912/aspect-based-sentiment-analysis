"""Aspect-extraction, review-set insights, and the demo storefront.

Everything here is built on one endpoint's worth of model: a batch of
(sentence, aspect) pairs. What is new is the step *before* the model -- 
deciding what the aspects are -- which the model does not do and which these
routes are explicit about, reporting a ``source`` for every aspect so a
client can say where it came from.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import PlainTextResponse

from ..core.errors import HTTP_422
from ..schemas.request import InsightsRequest
from ..schemas.response import (
    CatalogueInsights,
    CatalogueStatus,
    CatalogueUploadResult,
    InsightsResponse,
    ProductDetail,
    ProductSummary,
)
from ..services import catalogue
from ..services.catalogue_csv import (
    MAX_BYTES,
    MAX_PRODUCTS,
    MAX_REVIEWS_PER_PRODUCT,
    VALID_CATEGORIES,
    CatalogueCsvError,
    parse_catalogue_csv,
)
from ..services.insights import summarise
from ..services.predictor import PredictorService, get_predictor

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["insights"])

#: The sample file the UI offers for download. Three products across both
#: categories, so someone can see the shape and the category values without
#: reading the README.
SAMPLE_CSV = """product_name,category,price,review_text,rating
Aurora 14,laptop,$1099,Battery life is amazing but the screen is dim.,4
Aurora 14,laptop,$1099,The keyboard feels great for long typing sessions.,5
Aurora 14,laptop,$1099,Screen is washed out in sunlight.,2
Meridian Pro 16,laptop,$2399,Gorgeous screen and the performance is superb.,5
Meridian Pro 16,laptop,$2399,The price is far too high for what you get.,2
Olive & Ash,restaurant,$$$,Great food but the service was dreadful.,3
Olive & Ash,restaurant,$$$,The atmosphere is wonderful and the prices are reasonable.,5
"""



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


# ---- the active catalogue ---------------------------------------------- #


@router.get(
    "/catalogue/status",
    response_model=CatalogueStatus,
    summary="Which catalogue is active, and how far its scoring has got",
)
def catalogue_status(
    predictor: PredictorService = Depends(get_predictor),
) -> CatalogueStatus:
    """Report the active catalogue and the background scoring's progress.

    Always 200, and deliberately works with no model loaded: a store page
    that could not say which file it is showing until the artifact finished
    loading would be worse than one reporting "scored 0 of 24". The UI polls
    this while `scoring` is true.
    """
    return catalogue.status(predictor)


@router.get(
    "/catalogue/sample",
    response_class=PlainTextResponse,
    summary="A sample CSV in the upload format",
)
def catalogue_sample() -> PlainTextResponse:
    """The upload format, as a file.

    Served rather than shipped as a frontend asset so the format has exactly
    one definition, next to the parser that enforces it.
    """
    return PlainTextResponse(
        SAMPLE_CSV,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="absa-catalogue-sample.csv"'},
    )


@router.post(
    "/catalogue/upload",
    response_model=CatalogueUploadResult,
    summary="Replace the active catalogue with an uploaded CSV",
    responses={
        422: {"description": "The file could not be used; the message says why."},
    },
)
async def catalogue_upload(
    file: UploadFile = File(..., description="CSV in the documented upload format."),
    default_category: str | None = Form(
        None,
        description=(
            "Category for rows whose file has no `category` column: "
            "'laptop' or 'restaurant'."
        ),
    ),
    predictor: PredictorService = Depends(get_predictor),
) -> CatalogueUploadResult:
    """Serve a catalogue built from the uploaded CSV instead of the demo one.

    In memory only. A restart returns to the built-in catalogue, which is the
    right default for a demo surface and avoids a server quietly serving a
    file somebody uploaded weeks ago.

    Needs no model: parsing is CSV work, and the upload succeeds while the
    artifact is still loading. The *scoring* needs one, and the status
    endpoint reports that separately.

    **The limit worth knowing before uploading anything:** aspects come from
    the SemEval gold vocabulary, which covers laptops and restaurants only.
    Reviews of anything else parse fine and then produce few or no aspects.
    """
    raw = await file.read()
    try:
        parsed = parse_catalogue_csv(raw, default_category=default_category)
    except CatalogueCsvError as exc:
        # The caller can fix every one of these, so the message is the whole
        # point of the response; it names the column, the row or the cap.
        LOGGER.info("rejected catalogue upload %r: %s", file.filename, exc)
        raise HTTPException(status_code=HTTP_422, detail=str(exc)) from exc

    state = catalogue.state_from_upload(parsed, file.filename or "uploaded.csv")
    catalogue.active.replace(state)

    result = catalogue.status(predictor)
    # Scoring starts now so the shelf fills in while the user reads the
    # summary, rather than on the first sort.
    try:
        catalogue.breakdowns.score_all_in_background(predictor)
    except Exception:  # noqa: BLE001 - no model yet is not an upload failure
        LOGGER.info("catalogue uploaded but not scored: no model loaded")

    return CatalogueUploadResult(**result.model_dump(), warnings=state.warnings)


@router.post(
    "/catalogue/reset",
    response_model=CatalogueStatus,
    summary="Go back to the built-in demo catalogue",
)
def catalogue_reset(
    predictor: PredictorService = Depends(get_predictor),
) -> CatalogueStatus:
    """Discard an uploaded catalogue and serve the demo one again."""
    catalogue.active.reset()
    return catalogue.status(predictor)
