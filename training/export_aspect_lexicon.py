"""Export the corpora's own aspect vocabulary for the backend to serve.

The classifier takes an aspect as *input*: it judges a (sentence, aspect)
pair and has no aspect-extraction step at all. Any demo that accepts a raw
review and reports "battery: 80% positive" has to decide what the aspects
are before the model can say anything, and that decision is not the model's.

The obvious reach is a POS tagger (spaCy noun chunks). This takes a different
route: SemEval-2014 Task 4 ships **human-annotated gold aspect terms**, and
they are sitting in the training CSVs already. Mining them gives an aspect
vocabulary that is, by construction, the vocabulary the model was fitted on
-- so an extracted aspect is in regime rather than merely plausible -- at the
cost of no new dependency, no download, and nothing to keep offline-safe in
CI. What it cannot do is find an aspect the corpora never annotated; that
limitation is real, stated in the UI, and the reason `source` is reported per
aspect.

Surface forms are written out verbatim, not stemmed. Matching is the
backend's job and it already owns the stemmer that the 422 gate uses
(`app/services/aspect_match.stem`); stemming here would fork that rule into
a second place, which is exactly the drift the parity fixture exists to stop.

Run after changing the corpora, and commit the result:

    cd training && python export_aspect_lexicon.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Corpus file -> the domain tag its aspects get. The demo store filters on
#: these, so a laptop product is not offered "atmosphere" as an aspect.
SOURCES: dict[str, str] = {
    "training/data/raw/Laptop_Train_v2.csv": "laptop",
    "training/data/raw/Restaurants_Train_v2.csv": "restaurant",
}

#: Keep every gold term, and let the backend tier them by `count`.
#:
#: The whole annotated vocabulary is 2142 terms. The 312 annotated three or
#: more times are a vocabulary in the useful sense -- reusable aspects people
#: write about. The long tail is mostly one-off phrases ("the very reasonably
#: priced wine") that are an annotation of a single sentence. Both are worth
#: shipping: the common tier drives extraction, and the rare tier is a second
#: pass for a review whose aspects the common tier misses entirely, which
#: still beats guessing from word frequency because every entry is a phrase a
#: human marked as an aspect term.
MIN_COUNT = 1

#: Drop anything longer than this many words. The corpus contains annotations
#: up to 19 words long ("the very reasonably priced wine that we ordered with
#: dinner"), which are a label on one sentence rather than an aspect anyone
#: would match against another review. Keeping them only inflates the file.
MAX_WORDS = 4

OUT_PATH = PROJECT_ROOT / "backend" / "app" / "data" / "aspect_lexicon.json"


def collect(min_count: int) -> list[dict[str, object]]:
    """Gold aspect terms, pooled across corpora, with counts and domains."""
    counts: Counter[str] = Counter()
    domains: dict[str, set[str]] = {}

    for relative, domain in SOURCES.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"missing corpus: {path}\n"
                "The raw CSVs are gitignored; see PROJECT_CONTEXT.md section 15."
            )
        # keep_default_na=False for the same reason the loaders use it: an
        # aspect term "nan" is the bread, not missing data.
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        column = next(c for c in frame.columns if "Aspect" in c)
        for raw in frame[column]:
            term = " ".join(raw.split()).lower()
            if not term:
                continue
            counts[term] += 1
            domains.setdefault(term, set()).add(domain)

    terms = [
        {
            "term": term,
            "count": count,
            "domains": sorted(domains[term]),
        }
        for term, count in counts.items()
        if count >= min_count and len(term.split()) <= MAX_WORDS
    ]
    # Longest first, then most frequent: the extractor wants to try "battery
    # life" before "battery", so that a review saying "battery life is short"
    # yields one aspect rather than two overlapping ones.
    terms.sort(key=lambda entry: (-len(str(entry["term"]).split()), -int(entry["count"]), entry["term"]))
    return terms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-count", type=int, default=MIN_COUNT)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    terms = collect(args.min_count)
    payload = {
        "_comment": (
            "Gold aspect terms from SemEval-2014 Task 4, exported by "
            "training/export_aspect_lexicon.py. Do not edit by hand."
        ),
        "min_count": args.min_count,
        "terms": terms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    by_domain = Counter(d for entry in terms for d in entry["domains"])  # type: ignore[union-attr]
    print(f"wrote {args.out}")
    print(f"  {len(terms)} terms at min_count={args.min_count}")
    print(f"  by domain: {dict(by_domain)}")
    print(f"  longest: {max(len(str(e['term']).split()) for e in terms)} words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
