"""Building a catalogue out of an uploaded CSV.

The storefront ships with a fixed demo catalogue; this turns a real review
export into the same shape, so the Store tab can be pointed at someone's own
data. Nothing here touches the model -- it produces the same ``Product``
objects the demo catalogue is parsed into, and the existing scoring and
caching run unchanged on top.

**Canonical schema** (the format documented in the README and offered as a
sample download):

    product_name, category, review_text        -- required
    price, rating                              -- optional

Real exports never use those names, so a small table of **explicit aliases**
is accepted as well (``name`` for ``product_name``, ``text`` for
``review_text``, and so on). Aliases are a fixed list rather than fuzzy
matching: guessing which column holds the review would fail silently and
produce a catalogue of nonsense, where an unrecognised header produces a 422
naming exactly what is missing.

``category`` may be absent from the file and supplied once for the whole
upload instead -- a single-domain export has no reason to repeat it on every
row. It must be ``laptop`` or ``restaurant``, because those are the only
domains the aspect vocabulary covers.

Everything is capped. The caps are not politeness: each review is split into
sentences and every aspect occurrence is a BERT forward pass, so an
uncapped upload is a denial of service against a CPU-bound server.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

#: Categories the aspect vocabulary covers. Anything else would extract
#: nothing and look like a broken page rather than an unsupported domain.
VALID_CATEGORIES = ("laptop", "restaurant")

#: Largest upload accepted, in bytes. The Yelp restaurant extract this was
#: built against is 9.3MB of CSV for 14k reviews, so 20MB is roughly twice
#: the largest realistic input while still bounded.
MAX_BYTES = 20 * 1024 * 1024

#: Row ceiling, checked while parsing so a huge file is abandoned early
#: rather than read into memory first.
MAX_ROWS = 50_000

#: Products kept, most-reviewed first. A shelf is browsed, not paged through,
#: and the ceiling is set by what the scoring costs, measured rather than
#: guessed: on this CPU, with the cap below, a product averages ~1.5s and a
#: whole catalogue ~36s. The scoring runs in the background with progress
#: reported, so that is a wait the page shows, not one it hides -- but it is
#: still the reason this is 24 and not 400.
MAX_PRODUCTS = 24

#: Reviews kept per product. 15 reviews of real Yelp prose is ~14 aspect
#: judgements, which is enough for a share to mean something; past this the
#: shares stop moving and the only thing another review buys is CPU.
MAX_REVIEWS_PER_PRODUCT = 15

#: Reviews shorter than this are dropped: a two-word review carries no
#: aspect and would only pad the denominator.
MIN_REVIEW_CHARS = 12

#: Matches the per-review ceiling the insights request schema enforces.
MAX_REVIEW_CHARS = 5_000

#: Canonical column -> the header names accepted for it, lower-cased.
#: The canonical name is always accepted; aliases are the headers real
#: exports use, including this project's reference dataset (Yelp restaurant
#: extract: `name`, `text`, `stars_review`).
ALIASES: dict[str, tuple[str, ...]] = {
    "product_name": ("product_name", "name", "business_name", "product", "title", "item"),
    "review_text": ("review_text", "text", "review", "reviews.text", "content", "body"),
    "category": ("category", "domain", "product_category"),
    "price": ("price", "prices", "cost"),
    "rating": ("rating", "stars", "stars_review", "reviews.rating", "score", "review_stars"),
}

REQUIRED = ("product_name", "review_text")


class CatalogueCsvError(ValueError):
    """An upload the caller has to fix. The route turns this into a 422."""


@dataclass(frozen=True)
class ParsedProduct:
    """One product gathered from the CSV, before it becomes a ``Product``."""

    slug: str
    name: str
    category: str
    price: str
    reviews: tuple[str, ...]
    #: Mean of the per-review star ratings, when the file carried any.
    rating: float | None
    #: How many reviews the rating was averaged over.
    rating_count: int
    #: Reviews seen for this product before the per-product cap.
    reviews_seen: int


@dataclass(frozen=True)
class ParsedCatalogue:
    products: tuple[ParsedProduct, ...]
    #: Per-category aspect sort keys, same shape as the demo catalogue's.
    category_aspects: dict[str, tuple[str, ...]]
    #: Counts for the honesty note and the upload summary.
    rows_read: int
    rows_used: int
    products_found: int
    truncated_products: bool
    truncated_reviews: bool


def slugify(name: str) -> str:
    """A URL-safe id for a product name.

    Collisions are resolved by the caller, which appends a counter; two
    restaurants really can share a name.
    """
    kept = [character.lower() if character.isalnum() else "-" for character in name]
    slug = "".join(kept).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:60] or "product"


def _resolve_headers(fieldnames: list[str] | None) -> dict[str, str]:
    """Map canonical names onto the file's actual headers.

    Raises:
        CatalogueCsvError: when a required column has no recognised header.
    """
    if not fieldnames:
        raise CatalogueCsvError("the file has no header row")

    seen = {(name or "").strip().lower(): (name or "") for name in fieldnames}
    resolved: dict[str, str] = {}
    for canonical, accepted in ALIASES.items():
        for alias in accepted:
            if alias in seen:
                resolved[canonical] = seen[alias]
                break

    missing = [name for name in REQUIRED if name not in resolved]
    if missing:
        raise CatalogueCsvError(
            "missing required column(s): "
            + ", ".join(missing)
            + ". Accepted headers: "
            + "; ".join(
                f"{name} ({', '.join(ALIASES[name])})" for name in missing
            )
            + f". Found: {', '.join(sorted(seen.values()))}"
        )
    return resolved


def _coerce_rating(raw: str) -> float | None:
    """A star rating, or None when the cell is blank or not a number."""
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    # Ratings outside 0-5 are a different scale (or a mis-mapped column);
    # dropping one is better than displaying 87 stars.
    return value if 0.0 <= value <= 5.0 else None


def parse_catalogue_csv(
    data: bytes,
    *,
    default_category: str | None = None,
    aspects_by_category: dict[str, tuple[str, ...]] | None = None,
    max_products: int = MAX_PRODUCTS,
    max_reviews: int = MAX_REVIEWS_PER_PRODUCT,
) -> ParsedCatalogue:
    """Group an uploaded CSV into products.

    Args:
        data: the raw file.
        default_category: used for rows whose file has no ``category`` column.
        aspects_by_category: the sort keys each category offers; defaults to
            the demo catalogue's, so an upload ranks on the same aspects.
        max_products: product ceiling, most-reviewed first.
        max_reviews: per-product review ceiling.

    Returns:
        A :class:`ParsedCatalogue`.

    Raises:
        CatalogueCsvError: for anything the caller must fix -- too large, not
            decodable, no header, a missing column, an unknown category, or
            no usable rows at all.
    """
    if not data:
        raise CatalogueCsvError("the file is empty")
    if len(data) > MAX_BYTES:
        raise CatalogueCsvError(
            f"the file is {len(data) / 1_048_576:.1f}MB; the limit is "
            f"{MAX_BYTES // 1_048_576}MB. Upload a sample of it instead."
        )

    try:
        # utf-8-sig: a CSV exported from a spreadsheet usually carries a BOM,
        # which would otherwise become part of the first header's name and
        # make `product_name` unrecognisable.
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CatalogueCsvError(
            "the file is not valid UTF-8 text; re-export it as UTF-8 CSV"
        ) from exc

    if default_category is not None and default_category not in VALID_CATEGORIES:
        raise CatalogueCsvError(
            f"category must be one of {', '.join(VALID_CATEGORIES)}, got "
            f"{default_category!r}"
        )

    reader = csv.DictReader(io.StringIO(text))
    headers = _resolve_headers(reader.fieldnames)

    # name -> accumulating state. Insertion-ordered, so an export already
    # grouped by product keeps its order among equally-reviewed products.
    gathered: dict[str, dict] = {}
    rows_read = 0
    rows_used = 0

    for row in reader:
        rows_read += 1
        if rows_read > MAX_ROWS:
            raise CatalogueCsvError(
                f"the file has more than {MAX_ROWS:,} rows. Upload a sample of it instead."
            )

        name = (row.get(headers["product_name"]) or "").strip()
        review = (row.get(headers["review_text"]) or "").strip()
        if not name or len(review) < MIN_REVIEW_CHARS:
            continue

        if "category" in headers:
            category = (row.get(headers["category"]) or "").strip().lower()
            if not category:
                category = default_category or ""
        else:
            category = default_category or ""

        if category not in VALID_CATEGORIES:
            raise CatalogueCsvError(
                f"row {rows_read}: category {category or '(blank)'!r} is not supported. "
                f"Use one of {', '.join(VALID_CATEGORIES)} -- the aspect vocabulary "
                "only covers those domains. Supply it as a column or pick one for "
                "the whole file."
            )

        state = gathered.setdefault(
            name,
            {
                "name": name,
                "category": category,
                "price": "",
                "reviews": [],
                "ratings": [],
                "seen": 0,
            },
        )
        state["seen"] += 1
        rows_used += 1

        if "price" in headers and not state["price"]:
            state["price"] = (row.get(headers["price"]) or "").strip()[:24]

        if "rating" in headers:
            rating = _coerce_rating(row.get(headers["rating"]) or "")
            if rating is not None:
                state["ratings"].append(rating)

        if len(state["reviews"]) < max_reviews:
            state["reviews"].append(review[:MAX_REVIEW_CHARS])

    if not gathered:
        raise CatalogueCsvError(
            "no usable rows: every row was missing a product name or had a review "
            f"shorter than {MIN_REVIEW_CHARS} characters"
        )

    products_found = len(gathered)
    # Most-reviewed first: a product with 40 reviews says more than one with
    # a single review, and the cap should keep the informative ones.
    ranked = sorted(gathered.values(), key=lambda state: -state["seen"])
    truncated_products = products_found > max_products
    truncated_reviews = any(state["seen"] > max_reviews for state in ranked[:max_products])

    products: list[ParsedProduct] = []
    used_slugs: set[str] = set()
    for state in ranked[:max_products]:
        slug = slugify(state["name"])
        if slug in used_slugs:
            # Two businesses can share a name; the slug is a URL id and has
            # to stay unique or one product shadows the other.
            suffix = 2
            while f"{slug}-{suffix}" in used_slugs:
                suffix += 1
            slug = f"{slug}-{suffix}"
        used_slugs.add(slug)

        ratings = state["ratings"]
        products.append(
            ParsedProduct(
                slug=slug,
                name=state["name"],
                category=state["category"],
                price=state["price"],
                reviews=tuple(state["reviews"]),
                rating=round(sum(ratings) / len(ratings), 2) if ratings else None,
                rating_count=len(ratings),
                reviews_seen=state["seen"],
            )
        )

    categories = {product.category for product in products}
    aspects = {
        category: tuple(aspects_by_category.get(category, ()) if aspects_by_category else ())
        for category in categories
    }

    LOGGER.info(
        "parsed upload: %d rows -> %d products (%d found, %d rows used)",
        rows_read,
        len(products),
        products_found,
        rows_used,
    )
    return ParsedCatalogue(
        products=tuple(products),
        category_aspects=aspects,
        rows_read=rows_read,
        rows_used=rows_used,
        products_found=products_found,
        truncated_products=truncated_products,
        truncated_reviews=truncated_reviews,
    )
