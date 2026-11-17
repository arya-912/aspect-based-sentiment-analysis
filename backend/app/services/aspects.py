"""Finding the aspects in a review, which the model itself cannot do.

The classifier judges a (sentence, aspect) pair: the aspect is an *input*.
Nothing in this project extracts aspects, because SemEval-2014 Task 4 splits
aspect extraction and aspect sentiment into separate subtasks and only the
second one is modelled here. So anything that accepts a raw review and
reports "battery: 80% positive" has to decide the aspects first, outside the
model, and should be honest that it is doing so.

This module decides them by **matching the corpora's own gold aspect terms**,
exported to ``app/data/aspect_lexicon.json`` by
``training/export_aspect_lexicon.py``. Two tiers:

``corpus``
    Annotated three or more times -- a reusable aspect vocabulary.
``corpus-rare``
    Annotated once or twice. Tried only when the common tier finds nothing in
    a review, because the tail is mostly one-off phrasing.

Why not a POS tagger. spaCy noun chunks would find aspects the corpora never
annotated, which is a real advantage, at the cost of a dependency and a model
download in a project whose tests block network access on purpose. More to
the point, a noun chunk is only a *plausible* aspect, while a gold term is
one the model was actually fitted on -- and the model is confidently wrong
off-distribution, which is the failure the whole 422 gate exists to prevent.

The limitation that remains, and which the UI states: an aspect no SemEval
annotator marked is invisible here. Reviews of a product category outside
laptops and restaurants will extract little or nothing.

Every aspect this returns is a verbatim span of the sentence it was found in
(``Occurrence.surface``), so a prediction about it can never hit the
presence gate in ``aspect_match``. That is not a coincidence to rely on
loosely -- there is a test for it.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .aspect_match import WORD_PATTERN, stem

LOGGER = logging.getLogger(__name__)

LEXICON_PATH = Path(__file__).resolve().parents[1] / "data" / "aspect_lexicon.json"

#: Minimum annotation count for the primary tier.
COMMON_MIN_COUNT = 3

#: Sentence boundary: terminal punctuation followed by whitespace. Crude on
#: purpose -- it splits "e.g." and "Dr." wrongly -- because the model was
#: trained on single sentences and over-splitting costs a shorter context,
#: while under-splitting lets one aspect's sentiment be read off another
#: aspect's clause. Reviews in both corpora are single sentences; this exists
#: for pasted text, which is not.
SENTENCE_BREAK = re.compile(r"(?<=[.!?])[\s ]+")

#: Longest sentence worth sending to the model, in characters. The encoder
#: truncates at 128 wordpieces anyway; this keeps a wall of unpunctuated text
#: from becoming one enormous "sentence" whose tail is silently dropped.
MAX_SENTENCE_CHARS = 600


@dataclass(frozen=True)
class Word:
    """One word of a sentence, with its offsets and its stem."""

    surface: str
    stem: str
    start: int
    end: int


@dataclass(frozen=True)
class Occurrence:
    """One place an aspect was found."""

    review_index: int
    sentence_index: int
    sentence: str
    #: The aspect exactly as it appears in this sentence. Predictions use
    #: this, not the canonical label: the lexicon matches on stems, so a
    #: sentence saying "batteries" is grouped under "battery" but must be
    #: *asked* about "batteries", or the model is handed a phrase its own
    #: input does not contain.
    surface: str


@dataclass(frozen=True)
class AspectCandidate:
    """An extracted aspect and everywhere it turned up."""

    #: Canonical label: the surface form seen most often across the reviews.
    aspect: str
    occurrences: tuple[Occurrence, ...]
    #: "corpus", "corpus-rare", or "given" when the caller supplied it.
    source: str
    #: How many times SemEval annotated this term, when it came from there.
    lexicon_count: int | None = None

    @property
    def total(self) -> int:
        return len(self.occurrences)


@dataclass(frozen=True)
class LexiconEntry:
    term: str
    count: int
    domains: tuple[str, ...]
    #: The term's words, stemmed, for matching.
    stems: tuple[str, ...]


@lru_cache(maxsize=1)
def load_lexicon(path: str | None = None) -> tuple[LexiconEntry, ...]:
    """The exported gold aspect vocabulary, stemmed and longest-first.

    Cached: it is read once per process and never changes under a running
    server. Stemming happens here rather than in the exporter so that this
    service and the 422 gate share one stemmer -- forking that rule is the
    drift the parity fixture exists to catch.
    """
    target = Path(path) if path else LEXICON_PATH
    if not target.is_file():
        LOGGER.warning("no aspect lexicon at %s; extraction will find nothing", target)
        return ()

    payload = json.loads(target.read_text(encoding="utf-8"))
    entries = []
    for raw in payload.get("terms", []):
        term = str(raw["term"])
        stems = tuple(stem(match.group()) for match in WORD_PATTERN.finditer(term))
        if not stems:
            continue
        entries.append(
            LexiconEntry(
                term=term,
                count=int(raw.get("count", 1)),
                domains=tuple(raw.get("domains", ())),
                stems=stems,
            )
        )
    # Longest first so "battery life" is matched and consumed before
    # "battery" can claim half of it.
    entries.sort(key=lambda entry: (-len(entry.stems), -entry.count, entry.term))
    LOGGER.info("loaded %d aspect terms from %s", len(entries), target)
    return tuple(entries)


def split_sentences(text: str) -> list[str]:
    """Break a review into sentences, dropping empties."""
    parts = [part.strip() for part in SENTENCE_BREAK.split(text.strip())]
    return [part[:MAX_SENTENCE_CHARS] for part in parts if part]


def words_of(sentence: str) -> list[Word]:
    """The sentence's words, with offsets and stems.

    Uses ``aspect_match``'s own pattern, so "what counts as a word" is the
    same question here as it is at the gate.
    """
    return [
        Word(
            surface=match.group(),
            stem=stem(match.group()),
            start=match.start(),
            end=match.end(),
        )
        for match in WORD_PATTERN.finditer(sentence)
    ]


def _match_terms(
    sentence: str, entries: tuple[LexiconEntry, ...]
) -> list[tuple[LexiconEntry, str]]:
    """Every lexicon term in ``sentence``, non-overlapping, longest first.

    Returns (entry, verbatim span). Non-overlapping because "battery life"
    and "battery" would otherwise both fire on one phrase and the dashboard
    would show two aspects where the review discussed one.
    """
    words = words_of(sentence)
    if not words:
        return []

    stems = [word.stem for word in words]
    taken = [False] * len(words)
    found: list[tuple[LexiconEntry, str]] = []

    for entry in entries:  # already longest-first
        span = len(entry.stems)
        for start in range(len(stems) - span + 1):
            if any(taken[start : start + span]):
                continue
            if stems[start : start + span] != list(entry.stems):
                continue
            for index in range(start, start + span):
                taken[index] = True
            # The verbatim slice, so the model is asked about text that is
            # really in this sentence.
            found.append((entry, sentence[words[start].start : words[start + span - 1].end]))
    return found


def extract_aspects(
    reviews: list[str],
    *,
    domain: str | None = None,
    limit: int = 12,
    lexicon_path: str | None = None,
) -> list[AspectCandidate]:
    """The aspects discussed across ``reviews``, most-discussed first.

    Args:
        reviews: raw review texts, each possibly several sentences.
        domain: ``"laptop"`` or ``"restaurant"`` to restrict the vocabulary,
            so a laptop product is not offered "atmosphere". ``None`` uses all
            of it.
        limit: how many aspects to return.
        lexicon_path: override, for tests.

    Returns:
        Candidates sorted by how many sentences mention them, then by how
        often SemEval annotated the term.
    """
    entries = load_lexicon(lexicon_path)
    if domain:
        entries = tuple(e for e in entries if domain in e.domains)

    common = tuple(e for e in entries if e.count >= COMMON_MIN_COUNT)
    rare = tuple(e for e in entries if e.count < COMMON_MIN_COUNT)

    # term -> occurrences, and term -> surface forms seen, for the label.
    hits: dict[str, list[Occurrence]] = {}
    surfaces: dict[str, Counter[str]] = {}
    counts: dict[str, int] = {}

    for review_index, review in enumerate(reviews):
        for sentence_index, sentence in enumerate(split_sentences(review)):
            matched = _match_terms(sentence, common)
            if not matched:
                # Second pass only where the common tier found nothing at
                # all: the rare tier is one-off phrasing and would otherwise
                # bury the aspects people actually write about.
                matched = _match_terms(sentence, rare)
            for entry, surface in matched:
                hits.setdefault(entry.term, []).append(
                    Occurrence(
                        review_index=review_index,
                        sentence_index=sentence_index,
                        sentence=sentence,
                        surface=surface,
                    )
                )
                surfaces.setdefault(entry.term, Counter())[surface] += 1
                counts[entry.term] = entry.count

    candidates = [
        AspectCandidate(
            # The commonest surface form reads better than the lexicon's
            # lower-cased key: "Battery life" over "battery life" when that
            # is how the reviews write it.
            aspect=surfaces[term].most_common(1)[0][0],
            occurrences=tuple(occurrences),
            source="corpus" if counts[term] >= COMMON_MIN_COUNT else "corpus-rare",
            lexicon_count=counts[term],
        )
        for term, occurrences in hits.items()
    ]
    candidates.sort(key=lambda c: (-c.total, -(c.lexicon_count or 0), c.aspect.lower()))
    return candidates[:limit]


def locate_aspects(reviews: list[str], aspects: list[str]) -> list[AspectCandidate]:
    """Where caller-supplied aspects occur, as ``extract_aspects`` would report.

    Used when the client names the aspects itself -- the demo store does, from
    a fixed per-category list, and the dashboard does when the user edits the
    chips. Matching is the same stem-based, whole-word, contiguous rule as the
    gate, so an aspect the user typed in another inflection still lines up.
    """
    found: list[AspectCandidate] = []

    for aspect in aspects:
        term_stems = tuple(stem(m.group()) for m in WORD_PATTERN.finditer(aspect))
        if not term_stems:
            continue
        entry = LexiconEntry(term=aspect, count=0, domains=(), stems=term_stems)

        occurrences: list[Occurrence] = []
        surfaces: Counter[str] = Counter()
        for review_index, review in enumerate(reviews):
            for sentence_index, sentence in enumerate(split_sentences(review)):
                for _, surface in _match_terms(sentence, (entry,)):
                    occurrences.append(
                        Occurrence(
                            review_index=review_index,
                            sentence_index=sentence_index,
                            sentence=sentence,
                            surface=surface,
                        )
                    )
                    surfaces[surface] += 1
        if occurrences:
            found.append(
                AspectCandidate(
                    aspect=aspect,
                    occurrences=tuple(occurrences),
                    source="given",
                    lexicon_count=None,
                )
            )
    found.sort(key=lambda c: (-c.total, c.aspect.lower()))
    return found
