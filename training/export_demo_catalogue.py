"""Build the demo store's catalogue out of real, gold-labelled sentences.

The storefront needs products whose aspects genuinely differ -- one laptop
with good battery and a poor screen, another the other way round -- or
"sort by best battery" ranks nothing. The tempting shortcut is to write the
reviews by hand, which would make every number in the demo a number about
sentences chosen to produce it.

Instead each product is assembled from **real SemEval-2014 sentences whose
gold polarity for that aspect is known**, so a product advertised as strong
on battery is built from sentences human annotators labelled positive about
battery. The products are fictional; not one sentence is.

Two honesty constraints, both surfaced in the API and the UI:

* Held-out sentences are preferred. A sentence from the *training* split is
  one the model was fitted on, so predicting it is not a fair demo. The
  laptop test split is too thin for contrast (6 "battery life" sentences,
  all positive), so laptops fall back to train sentences; each product
  records how many of its reviews are held out in ``provenance``.
* No sentence is reused across products, so two products never share
  evidence.

Run after changing the profiles below, and commit the result:

    cd training && python export_demo_catalogue.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "backend" / "app" / "data" / "products.json"

SEED = 42

#: Corpus files per category: (held-out gold, training split).
CORPORA: dict[str, tuple[str, str]] = {
    "laptop": (
        "training/data/raw/Laptops_Test_Gold.csv",
        "training/data/raw/Laptop_Train_v2.csv",
    ),
    "restaurant": (
        "training/data/raw/Restaurants_Test_Gold.csv",
        "training/data/raw/Restaurants_Train_v2.csv",
    ),
}

#: The aspects each category's storefront offers as sort keys. Kept short:
#: these are the aspects with enough gold sentences to rank products on.
CATEGORY_ASPECTS: dict[str, list[str]] = {
    "laptop": ["battery life", "screen", "keyboard", "price", "performance"],
    "restaurant": ["food", "service", "atmosphere", "prices", "menu"],
}

#: Each product: the (aspect, gold polarity, how many sentences) it is built
#: from. The point is contrast -- every category has products that disagree
#: on the same aspect, so sorting by that aspect reorders the shelf.
PROFILES: list[dict[str, object]] = [
    {
        "slug": "aurora-14",
        "name": "Aurora 14",
        "category": "laptop",
        "subtitle": "Ultraportable, 14-inch",
        "price": "$1,099",
        "mix": [("battery life", "positive", 4), ("screen", "negative", 4),
                ("keyboard", "positive", 3), ("price", "negative", 2)],
    },
    {
        "slug": "meridian-pro-16",
        "name": "Meridian Pro 16",
        "category": "laptop",
        "subtitle": "Workstation, 16-inch",
        "price": "$2,399",
        "mix": [("screen", "positive", 4), ("performance", "positive", 3),
                ("price", "negative", 3), ("battery life", "negative", 3)],
    },
    {
        "slug": "vertex-air",
        "name": "Vertex Air",
        "category": "laptop",
        "subtitle": "Thin and light, 13-inch",
        "price": "$849",
        "mix": [("keyboard", "negative", 4), ("battery life", "positive", 4),
                ("price", "positive", 3), ("screen", "positive", 2)],
    },
    {
        "slug": "nimbus-15",
        "name": "Nimbus 15",
        "category": "laptop",
        "subtitle": "Budget all-rounder, 15-inch",
        "price": "$579",
        "mix": [("price", "positive", 5), ("performance", "negative", 3),
                ("hard drive", "negative", 2), ("keyboard", "positive", 2)],
    },
    {
        "slug": "olive-and-ash",
        "name": "Olive & Ash",
        "category": "restaurant",
        "subtitle": "Mediterranean, mid-town",
        "price": "$$$",
        "mix": [("food", "positive", 5), ("service", "negative", 5),
                ("atmosphere", "positive", 3), ("prices", "negative", 2)],
    },
    {
        "slug": "the-copper-pot",
        "name": "The Copper Pot",
        "category": "restaurant",
        "subtitle": "Neighbourhood bistro",
        "price": "$$",
        "mix": [("service", "positive", 5), ("food", "positive", 4),
                ("prices", "negative", 3), ("menu", "negative", 2)],
    },
    {
        "slug": "saffron-lane",
        "name": "Saffron Lane",
        "category": "restaurant",
        "subtitle": "Indian, river quarter",
        "price": "$$",
        "mix": [("atmosphere", "positive", 4), ("food", "negative", 4),
                ("menu", "neutral", 3), ("service", "positive", 2)],
    },
    {
        "slug": "rua-nova",
        "name": "Rua Nova",
        "category": "restaurant",
        "subtitle": "Portuguese small plates",
        "price": "$",
        "mix": [("prices", "positive", 5), ("food", "positive", 3),
                ("service", "negative", 3), ("menu", "positive", 2)],
    },
]


def load_pool(relative: str) -> pd.DataFrame:
    path = PROJECT_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"missing corpus: {path}\n"
            "The raw CSVs are gitignored; see PROJECT_CONTEXT.md section 15."
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = frame.rename(columns={"Sentence": "text", "Aspect Term": "aspect"})
    frame["aspect"] = frame["aspect"].str.strip().str.lower()
    frame["text"] = frame["text"].str.strip()
    return frame[["text", "aspect", "polarity"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    rng = random.Random(SEED)
    pools: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        category: (load_pool(heldout), load_pool(train))
        for category, (heldout, train) in CORPORA.items()
    }

    used: set[str] = set()  # no sentence appears under two products
    products = []
    warnings: list[str] = []

    for profile in PROFILES:
        category = str(profile["category"])
        heldout_pool, train_pool = pools[category]
        reviews: list[str] = []
        heldout_count = 0

        for aspect, polarity, wanted in profile["mix"]:  # type: ignore[union-attr]
            taken = 0
            # Held-out first; the training split only makes up a shortfall.
            for pool, is_heldout in ((heldout_pool, True), (train_pool, False)):
                if taken >= wanted:
                    break
                matches = pool[(pool.aspect == aspect) & (pool.polarity == polarity)]
                candidates = [t for t in dict.fromkeys(matches["text"]) if t not in used]
                rng.shuffle(candidates)
                for text in candidates[: wanted - taken]:
                    used.add(text)
                    reviews.append(text)
                    taken += 1
                    heldout_count += int(is_heldout)
            if taken < wanted:
                warnings.append(
                    f"{profile['slug']}: wanted {wanted} {polarity} {aspect!r}, found {taken}"
                )

        rng.shuffle(reviews)
        products.append(
            {
                "slug": profile["slug"],
                "name": profile["name"],
                "category": category,
                "subtitle": profile["subtitle"],
                "price": profile["price"],
                "reviews": reviews,
                "provenance": (
                    f"{heldout_count} of {len(reviews)} sentences are from the held-out "
                    "gold test split; the rest are from the training split, which the "
                    "model was fitted on."
                ),
            }
        )

    payload = {
        "_comment": (
            "Generated by training/export_demo_catalogue.py. Products are "
            "fictional; every review sentence is a real SemEval-2014 sentence, "
            "selected by its gold polarity for the aspect in question. Do not "
            "edit by hand."
        ),
        "category_aspects": CATEGORY_ASPECTS,
        "products": products,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"  {len(products)} products, {sum(len(p['reviews']) for p in products)} review sentences")
    print(f"  by category: {dict(Counter(p['category'] for p in products))}")
    for product in products:
        held = product["provenance"].split()[0]
        print(f"    {product['slug']:18} {len(product['reviews']):3} reviews, {held} held out")
    for warning in warnings:
        print(f"  WARNING {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
