"""The demo storefront's products, and their aspect breakdowns.

A mock catalogue whose point is that each product page shows *per-aspect*
sentiment, and that the shelf can be sorted by one aspect -- "best battery" --
which is a query no sentence-level sentiment score can answer. Sorting by an
overall star rating ranks products; sorting by an aspect ranks them
differently per aspect, and the difference is the demo.

The products are fictional and the reviews are not: each is a real
SemEval-2014 sentence chosen by its gold polarity for the aspect in question
(see ``training/export_demo_catalogue.py``).

Breakdowns are computed on demand and cached. 106 sentences across 8
products is a couple of hundred forward passes: measured on this CPU, one
product page is ~1s and the catalogue-wide scoring behind the aspect sort is
~8s. Both are far too slow to repeat per page load, and too slow to run at
startup, where they would delay readiness for a feature most callers never
open. So the first request for a product pays for that product, the first
catalogue-wide sort pays for the rest, and everything after is free until the
process restarts. The 8s is a visible loading state in the UI, not a hidden
stall.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..schemas.response import (
    AspectExample,
    AspectSummaryResponse,
    CatalogueInsights,
    CatalogueStatus,
    InsightsResponse,
    ProductAspectScore,
    ProductDetail,
    ProductSummary,
)
from .insights import InsightsSummary, summarise
from .predictor import PredictorService

LOGGER = logging.getLogger(__name__)

CATALOGUE_PATH = Path(__file__).resolve().parents[1] / "data" / "products.json"


@dataclass(frozen=True)
class Product:
    slug: str
    name: str
    category: str
    subtitle: str
    price: str
    reviews: tuple[str, ...]
    provenance: str
    #: Mean star rating from an uploaded CSV. None for the demo catalogue,
    #: whose SemEval sentences carry no stars.
    rating: float | None = None
    rating_count: int = 0
    #: Reviews the upload held for this product before the per-product cap.
    reviews_seen: int = 0


def wilson_lower_bound(positives: int, total: int, z: float = 1.96) -> float:
    """Wilson 95% lower bound on a proportion.

    The shelf used to rank on the raw positive share, which puts one lucky
    review above fifty good ones -- 1/1 is 1.00 and 45/50 is 0.90, so the
    single review wins. The lower bound asks instead "how good is this
    *at least*, given how little we have seen": 1/1 -> 0.21, 45/50 -> 0.79.

    Chosen over a shrunk mean because it needs no prior to be argued about
    and is the standard answer for ranking by a proportion. ``z`` is fixed at
    95%; the exact confidence level matters far less than having one at all.
    """
    if total <= 0:
        return 0.0
    phat = positives / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    return max(0.0, (centre - margin) / denominator)


@dataclass
class CatalogueState:
    """The catalogue being served, and where it came from."""

    products: tuple[Product, ...]
    category_aspects: dict[str, tuple[str, ...]]
    #: "demo" or "upload".
    source: str = "demo"
    source_name: str = "built-in demo catalogue"
    #: Bumped on every replace, and used as part of the breakdown cache key,
    #: so a new upload can never read a previous catalogue's cached scores
    #: even if a product slug happens to repeat.
    token: int = 0
    rows_used: int = 0
    products_found: int = 0
    truncated_products: bool = False
    truncated_reviews: bool = False
    note: str = ""
    warnings: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def load_catalogue(path: str | None = None) -> tuple[tuple[Product, ...], dict[str, tuple[str, ...]]]:
    """The product list and the per-category sort aspects.

    Returns an empty catalogue rather than raising when the file is absent:
    the store is a demo surface, and a backend that otherwise works should
    not fail to start because a generated data file was not committed.
    """
    target = Path(path) if path else CATALOGUE_PATH
    if not target.is_file():
        LOGGER.warning("no demo catalogue at %s; the store will be empty", target)
        return (), {}

    payload = json.loads(target.read_text(encoding="utf-8"))
    products = tuple(
        Product(
            slug=str(raw["slug"]),
            name=str(raw["name"]),
            category=str(raw["category"]),
            subtitle=str(raw.get("subtitle", "")),
            price=str(raw.get("price", "")),
            reviews=tuple(str(review) for review in raw.get("reviews", ())),
            provenance=str(raw.get("provenance", "")),
        )
        for raw in payload.get("products", [])
    )
    aspects = {
        category: tuple(terms)
        for category, terms in payload.get("category_aspects", {}).items()
    }
    LOGGER.info("loaded %d demo products from %s", len(products), target)
    return products, aspects


class ActiveCatalogue:
    """Which catalogue the store serves: the built-in demo, or an upload.

    An upload replaces it in memory only. Nothing is written to disk, so a
    restart returns to the demo catalogue -- which is the right default for a
    demo surface and avoids a server that quietly serves a file someone
    uploaded weeks ago.
    """

    def __init__(self) -> None:
        self._state: CatalogueState | None = None
        self._lock = threading.Lock()

    def get(self) -> CatalogueState:
        with self._lock:
            if self._state is None:
                products, aspects = load_catalogue()
                self._state = CatalogueState(products=products, category_aspects=aspects)
            return self._state

    def replace(self, state: CatalogueState) -> CatalogueState:
        """Serve ``state`` from now on, discarding any cached breakdowns."""
        with self._lock:
            previous = self._state.token if self._state else 0
            state.token = previous + 1
            self._state = state
        breakdowns.clear()
        LOGGER.info(
            "catalogue replaced: %s (%d products, token %d)",
            state.source_name,
            len(state.products),
            state.token,
        )
        return state

    def reset(self) -> CatalogueState:
        """Back to the built-in demo catalogue."""
        with self._lock:
            self._state = None
        breakdowns.clear()
        return self.get()


def find(slug: str) -> Product | None:
    state = active.get()
    return next((product for product in state.products if product.slug == slug), None)


def summaries() -> list[ProductSummary]:
    """The catalogue, without reviews or predictions. No model needed."""
    return [
        ProductSummary(
            slug=product.slug,
            name=product.name,
            category=product.category,
            subtitle=product.subtitle,
            price=product.price,
            review_count=len(product.reviews),
            rating=product.rating,
            rating_count=product.rating_count,
            reviews_seen=product.reviews_seen,
        )
        for product in active.get().products
    ]


class BreakdownCache:
    """Per-product aspect breakdowns, computed once per catalogue.

    Keyed by ``(catalogue token, slug)``, so an upload cannot be served a
    previous catalogue's numbers even when a slug repeats.

    Two locks, not one. ``_lock`` guards the dict and is held only briefly;
    ``_work`` serialises the scoring itself. Holding one lock across a
    multi-second forward pass would block a request for an *already cached*
    product behind an unrelated computation -- which is exactly the stall the
    background scorer exists to avoid.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int, str], InsightsSummary] = {}
        self._lock = threading.Lock()
        self._work = threading.Lock()
        self._scoring = False

    def peek(self, token: int, slug: str, model_version: str) -> InsightsSummary | None:
        with self._lock:
            cached = self._cache.get((token, slug))
            if cached is not None and cached.model_version == model_version:
                return cached
            return None

    def get(self, product: Product, predictor: PredictorService) -> InsightsSummary:
        state = active.get()
        version = predictor.registry.model_version
        cached = self.peek(state.token, product.slug, version)
        if cached is not None:
            return cached

        with self._work:
            # Re-checked under the work lock: another thread may have scored
            # this product while this one waited.
            cached = self.peek(state.token, product.slug, version)
            if cached is not None:
                return cached

            # Named explicitly rather than extracted: a storefront's sort keys
            # have to be the same on every product, or the shelf cannot be
            # ordered by one of them. Extraction would give each product
            # whichever aspects its own reviews happened to mention.
            aspects = list(state.category_aspects.get(product.category, ()))
            summary = summarise(
                list(product.reviews),
                predictor=predictor,
                aspects=aspects or None,
                domain=product.category,
                max_aspects=max(len(aspects), 1),
            )
            with self._lock:
                self._cache[(state.token, product.slug)] = summary
            return summary

    def scored_count(self, token: int) -> int:
        with self._lock:
            return sum(1 for (cached_token, _) in self._cache if cached_token == token)

    @property
    def scoring(self) -> bool:
        with self._lock:
            return self._scoring

    def wait_for_idle(self, timeout: float = 30.0) -> bool:
        """Block until no pass is running. For tests and shutdown.

        The worker is a daemon thread, so without this a test's scoring pass
        outlives the test and the next one's guard refuses to start a pass of
        its own -- which showed up as a flaky timeout rather than as the
        cross-test leak it was.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.scoring:
                return True
            time.sleep(0.01)
        return False

    def score_all_in_background(self, predictor: PredictorService) -> None:
        """Score every product, one at a time, off the request thread.

        Measured on this CPU: ~1.5s per product, ~36s for a full 24-product
        upload. Doing that inside a request would hold the connection open
        for half a minute; doing it at startup would delay readiness for a
        feature most callers never open. So it runs here and the UI polls
        ``/catalogue/status``, showing scores as they land.

        A single worker, not a pool: the model is one CPU-bound object and
        four threads contending on it would finish no sooner while making the
        per-product timings meaningless.
        """
        state = active.get()

        # Claimed synchronously, under the lock, *before* the thread starts.
        # Setting it inside the worker left a window in which a pass was
        # pending but `scoring` still read False -- so a second caller could
        # start a duplicate pass, and a status poll could report "idle, 0
        # scored" for a catalogue that was about to be scored.
        with self._lock:
            if self._scoring:
                return
            self._scoring = True

        def worker() -> None:
            try:
                for product in state.products:
                    # The catalogue can be replaced mid-pass; abandon the old
                    # one rather than filling the cache with dead entries.
                    if active.get().token != state.token:
                        LOGGER.info("background scoring abandoned: catalogue replaced")
                        return  # the finally below releases the claim
                    try:
                        self.get(product, predictor)
                    except Exception:  # noqa: BLE001 - one bad product must not stop the pass
                        LOGGER.exception("could not score %s", product.slug)
            finally:
                with self._lock:
                    self._scoring = False

        threading.Thread(target=worker, name="catalogue-scorer", daemon=True).start()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


#: Process-wide, like the model registry: the breakdowns describe the loaded
#: artifact, and a per-request cache would never be warm.
breakdowns = BreakdownCache()

#: Which catalogue is being served. Declared after the cache it clears.
active = ActiveCatalogue()


def to_response(summary: InsightsSummary) -> InsightsResponse:
    """Map the service dataclass onto the wire schema."""
    return InsightsResponse(
        reviews=summary.reviews,
        sentences=summary.sentences,
        predictions=summary.predictions,
        mixed_sentences=summary.mixed_sentences,
        mixed_aspects=summary.mixed_aspects,
        truncated=summary.truncated,
        model_version=summary.model_version,
        aspects=[
            AspectSummaryResponse(
                aspect=aspect.aspect,
                source=aspect.source,
                total=aspect.total,
                counts=aspect.counts,
                shares=aspect.shares,
                dominant=aspect.dominant,
                mean_confidence=aspect.mean_confidence,
                mixed=aspect.mixed,
                examples=[
                    AspectExample(
                        sentence=example.sentence,
                        surface=example.surface,
                        sentiment=example.sentiment,
                        confidence=example.confidence,
                        review_index=example.review_index,
                    )
                    for example in aspect.examples
                ],
            )
            for aspect in summary.aspects
        ],
    )


def detail(product: Product, predictor: PredictorService) -> ProductDetail:
    summary = breakdowns.get(product, predictor)
    return ProductDetail(
        slug=product.slug,
        name=product.name,
        category=product.category,
        subtitle=product.subtitle,
        price=product.price,
        review_count=len(product.reviews),
        rating=product.rating,
        rating_count=product.rating_count,
        reviews_seen=product.reviews_seen,
        reviews=list(product.reviews),
        insights=to_response(summary),
    )


def _scores_for(summary: InsightsSummary) -> list[ProductAspectScore]:
    """One product's per-aspect scores, ranked on the Wilson lower bound."""
    return [
        ProductAspectScore(
            aspect=aspect.aspect,
            positive_share=round(aspect.shares.get("positive", 0.0), 4),
            negative_share=round(aspect.shares.get("negative", 0.0), 4),
            total=aspect.total,
            score=round(
                wilson_lower_bound(aspect.counts.get("positive", 0), aspect.total), 4
            ),
        )
        for aspect in summary.aspects
    ]


def catalogue_insights(
    predictor: PredictorService, *, background: bool = True
) -> CatalogueInsights:
    """Whatever is scored so far, and a background pass for the rest.

    Returns immediately with the products already in the cache rather than
    blocking until all of them are done: a full 24-product upload is ~36s on
    CPU. ``background=False`` scores everything inline instead, which is what
    the tests want -- a fake model is instant and a thread would make them
    race.
    """
    # Asked for up front so a caller with no model loaded gets a 503 rather
    # than an empty map and a background pass that can never produce
    # anything. model_version alone would not do it: it falls back to the
    # artifact directory's name and answers happily with nothing loaded.
    predictor.registry.get_model()

    state = active.get()
    version = predictor.registry.model_version
    scores: dict[str, list[ProductAspectScore]] = {}

    for product in state.products:
        summary = (
            breakdowns.get(product, predictor)
            if not background
            else breakdowns.peek(state.token, product.slug, version)
        )
        if summary is not None:
            scores[product.slug] = _scores_for(summary)

    if background and len(scores) < len(state.products):
        breakdowns.score_all_in_background(predictor)

    # Every aspect any category offers, in the order the data file lists them.
    ordered: list[str] = []
    for terms in state.category_aspects.values():
        for term in terms:
            if term not in ordered:
                ordered.append(term)

    return CatalogueInsights(
        aspects=ordered,
        products=scores,
        model_version=version,
        scored=len(scores),
        total=len(state.products),
        scoring=breakdowns.scoring,
    )


def status(predictor: PredictorService | None = None) -> CatalogueStatus:
    """Which catalogue is active and how far its scoring has got.

    Takes no model when none is loaded: the source, the product count and the
    caveats are all known without one, and a store page that could not even
    say which file it was showing until the artifact loaded would be worse
    than one that says "scoring: 0 of 24".
    """
    state = active.get()
    version = ""
    scored = 0
    if predictor is not None:
        try:
            version = predictor.registry.model_version
            scored = breakdowns.scored_count(state.token)
        except Exception:  # noqa: BLE001 - no model loaded is not an error here
            version = ""

    return CatalogueStatus(
        source=state.source,
        source_name=state.source_name,
        products=len(state.products),
        scored=scored,
        scoring=breakdowns.scoring,
        model_version=version,
        rows_used=state.rows_used,
        products_found=state.products_found,
        truncated_products=state.truncated_products,
        truncated_reviews=state.truncated_reviews,
        note=state.note,
    )


def state_from_upload(parsed, filename: str) -> CatalogueState:
    """Turn a parsed CSV into the catalogue state the store serves.

    The aspect sort keys come from the **demo** catalogue's per-category list
    rather than from the upload, so an uploaded restaurant catalogue ranks on
    the same five aspects the built-in one does. Letting the file choose them
    would mean every upload ranked on a different axis.
    """
    _, demo_aspects = load_catalogue()
    warnings: list[str] = []

    if parsed.truncated_products:
        warnings.append(
            f"The file held {parsed.products_found:,} products; the "
            f"{len(parsed.products)} most-reviewed are shown."
        )
    if parsed.truncated_reviews:
        warnings.append(
            "Products with more reviews than the per-product cap were sampled "
            "down, so a share is over the reviews shown, not all of them."
        )
    if not any(product.rating is not None for product in parsed.products):
        warnings.append("No usable star ratings in the file, so none are shown.")

    products = tuple(
        Product(
            slug=parsed_product.slug,
            name=parsed_product.name,
            category=parsed_product.category,
            subtitle=(
                f"{parsed_product.reviews_seen} reviews in the file"
                if parsed_product.reviews_seen > len(parsed_product.reviews)
                else f"{len(parsed_product.reviews)} reviews"
            ),
            price=parsed_product.price,
            reviews=parsed_product.reviews,
            provenance=f"Uploaded from {filename}.",
            rating=parsed_product.rating,
            rating_count=parsed_product.rating_count,
            reviews_seen=parsed_product.reviews_seen,
        )
        for parsed_product in parsed.products
    )

    categories = {product.category for product in products}
    aspects = {
        category: tuple(demo_aspects.get(category, ())) for category in categories
    }

    return CatalogueState(
        products=products,
        category_aspects=aspects,
        source="upload",
        source_name=filename,
        rows_used=parsed.rows_used,
        products_found=parsed.products_found,
        truncated_products=parsed.truncated_products,
        truncated_reviews=parsed.truncated_reviews,
        note=(
            "These numbers come from an uploaded file, not the built-in demo "
            "catalogue."
        ),
        warnings=warnings,
    )
