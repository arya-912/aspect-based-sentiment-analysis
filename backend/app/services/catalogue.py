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
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..schemas.response import (
    AspectExample,
    AspectSummaryResponse,
    CatalogueInsights,
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


def find(slug: str) -> Product | None:
    products, _ = load_catalogue()
    return next((product for product in products if product.slug == slug), None)


def summaries() -> list[ProductSummary]:
    """The catalogue, without reviews or predictions. No model needed."""
    products, _ = load_catalogue()
    return [
        ProductSummary(
            slug=product.slug,
            name=product.name,
            category=product.category,
            subtitle=product.subtitle,
            price=product.price,
            review_count=len(product.reviews),
        )
        for product in products
    ]


class BreakdownCache:
    """Per-product aspect breakdowns, computed once per process.

    Locked, because two requests for the same product arriving together would
    otherwise both run several hundred forward passes. The lock is held across
    the computation rather than only around the dict write: the point is to do
    the work once, not to do it twice safely.
    """

    def __init__(self) -> None:
        self._cache: dict[str, InsightsSummary] = {}
        self._lock = threading.Lock()

    def get(self, product: Product, predictor: PredictorService) -> InsightsSummary:
        with self._lock:
            cached = self._cache.get(product.slug)
            if cached is not None and cached.model_version == predictor.registry.model_version:
                return cached

            _, category_aspects = load_catalogue()
            # Named explicitly rather than extracted: a storefront's sort keys
            # have to be the same on every product, or the shelf cannot be
            # ordered by one of them. Extraction would give each product
            # whichever aspects its own reviews happened to mention.
            aspects = list(category_aspects.get(product.category, ()))
            summary = summarise(
                list(product.reviews),
                predictor=predictor,
                aspects=aspects or None,
                domain=product.category,
                max_aspects=max(len(aspects), 1),
            )
            self._cache[product.slug] = summary
            return summary

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


#: Process-wide, like the model registry: the breakdowns describe the loaded
#: artifact, and a per-request cache would never be warm.
breakdowns = BreakdownCache()


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
        reviews=list(product.reviews),
        insights=to_response(summary),
    )


def catalogue_insights(predictor: PredictorService) -> CatalogueInsights:
    """Every product's standing on every aspect, for sorting the shelf."""
    products, category_aspects = load_catalogue()
    scores: dict[str, list[ProductAspectScore]] = {}

    for product in products:
        summary = breakdowns.get(product, predictor)
        scores[product.slug] = [
            ProductAspectScore(
                aspect=aspect.aspect,
                positive_share=round(aspect.shares.get("positive", 0.0), 4),
                negative_share=round(aspect.shares.get("negative", 0.0), 4),
                total=aspect.total,
            )
            for aspect in summary.aspects
        ]

    # Every aspect any category offers, in the order the data file lists them.
    ordered: list[str] = []
    for terms in category_aspects.values():
        for term in terms:
            if term not in ordered:
                ordered.append(term)

    return CatalogueInsights(
        aspects=ordered,
        products=scores,
        model_version=predictor.registry.model_version,
    )
