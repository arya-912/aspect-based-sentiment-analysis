"""Aggregating per-aspect sentiment over a set of reviews.

This is the thing sentence-level sentiment cannot produce. A sentence
classifier reading *"Battery life is amazing but the screen is dim"* has to
pick one label for the whole sentence and is therefore wrong about one of the
two aspects, whichever it picks. Here the same sentence contributes
``battery life: positive`` and ``screen: negative``, and ``mixed_sentences``
counts how often that happens in the caller's own data -- which is the
argument for aspect-based analysis made with their reviews rather than with a
claim.

The pipeline:

1. Split each review into sentences (the model was trained on single
   sentences, one aspect at a time).
2. Decide the aspects -- ``aspects.extract_aspects`` from the corpora's gold
   vocabulary, or the caller's own list.
3. Pair every aspect with the sentences that actually mention it, using the
   verbatim span from *that* sentence.
4. One batched set of forward passes.
5. Aggregate per aspect, keeping the highest-confidence example of each
   polarity.

Step 3 is why nothing here can trip the 422 presence gate: a pair exists only
because the aspect was found in that sentence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .aspects import AspectCandidate, Occurrence, extract_aspects, locate_aspects, split_sentences
from .predictor import PredictorService

LOGGER = logging.getLogger(__name__)

#: Hard ceiling on forward passes for one request. Each is a BERT pass on
#: CPU; 400 is roughly 3 seconds on the machine this was built on, which is
#: the edge of what a UI can wait for without a progress story. Over the cap
#: the aspects are kept and their occurrences truncated, so the shares stay
#: representative instead of the last aspects vanishing.
MAX_PAIRS = 400

#: Examples kept per aspect. Enough to show one of each polarity.
MAX_EXAMPLES = 3

#: Tie-break order for ``dominant`` when two labels draw the same number of
#: sentences. Earlier wins, so a 50/50 aspect is headlined "negative".
#:
#: A tie is not a verdict, and this is a product dashboard: resolving it
#: alphabetically made a screen that half the reviews dislike read as
#: "positive". Leaning to the complaint is the honest default -- the full
#: distribution and the `mixed` flag are right beside it, so nothing is
#: hidden, and the headline no longer flatters.
TIE_BREAK = ("negative", "neutral", "positive")


def _dominant(counts: dict[str, int]) -> str:
    """The label with the most sentences; ties go to the worse news."""

    def rank(label: str) -> int:
        try:
            return TIE_BREAK.index(label)
        except ValueError:  # an artifact with other label names
            return len(TIE_BREAK)

    return min(counts.items(), key=lambda item: (-item[1], rank(item[0])))[0]


@dataclass(frozen=True)
class Example:
    """One sentence, and what the model said about this aspect in it."""

    sentence: str
    #: The aspect exactly as this sentence spells it; may differ in
    #: inflection from the aspect's canonical label.
    surface: str
    sentiment: str
    confidence: float | None
    review_index: int


@dataclass
class AspectSummary:
    """What a review set says about one aspect."""

    aspect: str
    #: "corpus", "corpus-rare" or "given" -- where the aspect came from, since
    #: none of them came from the model.
    source: str
    total: int
    counts: dict[str, int]
    shares: dict[str, float]
    dominant: str
    mean_confidence: float
    #: True when this aspect drew more than one polarity across the reviews.
    mixed: bool
    examples: list[Example] = field(default_factory=list)


@dataclass
class InsightsSummary:
    """The whole aggregate for one review set."""

    reviews: int
    sentences: int
    predictions: int
    aspects: list[AspectSummary]
    #: Sentences where two aspects drew *different* polarities. The headline
    #: number: every one of these is a sentence a single-label classifier is
    #: necessarily wrong about.
    mixed_sentences: int
    #: Aspects that drew more than one polarity across the set.
    mixed_aspects: int
    truncated: bool
    model_version: str


def _pairs_for(
    candidates: list[AspectCandidate], limit: int
) -> tuple[list[tuple[str, str]], list[tuple[str, Occurrence]], bool]:
    """Flatten candidates into prediction pairs, round-robin under the cap.

    Round-robin rather than aspect-by-aspect: taking the first 400 pairs in
    order would spend the whole budget on the most-mentioned aspect and leave
    the rest with nothing, which silently changes *which* aspects the answer
    is about. Cycling keeps every aspect represented and only thins the
    evidence behind each.
    """
    queues = [list(candidate.occurrences) for candidate in candidates]
    pairs: list[tuple[str, str]] = []
    keyed: list[tuple[str, Occurrence]] = []
    truncated = False

    position = 0
    while any(queues):
        progressed = False
        for index, queue in enumerate(queues):
            if position >= len(queue):
                continue
            progressed = True
            if len(pairs) >= limit:
                truncated = True
                return pairs, keyed, truncated
            occurrence = queue[position]
            pairs.append((occurrence.sentence, occurrence.surface))
            keyed.append((candidates[index].aspect, occurrence))
        if not progressed:
            break
        position += 1

    return pairs, keyed, truncated


def summarise(
    reviews: list[str],
    *,
    predictor: PredictorService,
    aspects: list[str] | None = None,
    domain: str | None = None,
    max_aspects: int = 12,
    max_pairs: int = MAX_PAIRS,
) -> InsightsSummary:
    """Per-aspect sentiment across ``reviews``.

    Args:
        reviews: raw review texts.
        predictor: the model wrapper; injected so tests can fake it.
        aspects: name the aspects explicitly instead of extracting them.
        domain: restrict the extraction vocabulary ("laptop"/"restaurant").
        max_aspects: how many aspects to report.
        max_pairs: ceiling on forward passes.

    Returns:
        An :class:`InsightsSummary`. Empty of aspects -- not an error -- when
        nothing was found: a review set about a product category the corpora
        never annotated is a real and expected outcome.
    """
    texts = [review for review in reviews if review and review.strip()]
    sentence_count = sum(len(split_sentences(review)) for review in texts)

    candidates = (
        locate_aspects(texts, aspects)
        if aspects
        else extract_aspects(texts, domain=domain, limit=max_aspects)
    )
    candidates = candidates[:max_aspects]

    if not candidates:
        return InsightsSummary(
            reviews=len(texts),
            sentences=sentence_count,
            predictions=0,
            aspects=[],
            mixed_sentences=0,
            mixed_aspects=0,
            truncated=False,
            model_version=predictor.registry.model_version,
        )

    pairs, keyed, truncated = _pairs_for(candidates, max_pairs)
    predictions = predictor.predict_many(pairs)

    if len(predictions) != len(keyed):
        # predict_many drops pairs the encoder rejects, so the lists can
        # diverge. Re-align on the aspect text, which is all that identifies
        # a result; a mismatch here would silently attribute one aspect's
        # sentiment to another.
        LOGGER.info(
            "%d of %d pairs were unencodable and have been dropped",
            len(keyed) - len(predictions),
            len(keyed),
        )
        keyed = _realign(keyed, predictions)

    # aspect -> label -> count, and the examples worth keeping.
    buckets: dict[str, dict[str, int]] = {}
    confidences: dict[str, list[float]] = {}
    examples: dict[str, list[Example]] = {}
    # (review, sentence) -> the set of labels its aspects drew.
    per_sentence: dict[tuple[int, int], set[str]] = {}

    for (aspect, occurrence), prediction in zip(keyed, predictions):
        label = prediction.sentiment
        buckets.setdefault(aspect, {})[label] = buckets.setdefault(aspect, {}).get(label, 0) + 1
        if prediction.confidence is not None:
            confidences.setdefault(aspect, []).append(prediction.confidence)
        examples.setdefault(aspect, []).append(
            Example(
                sentence=occurrence.sentence,
                surface=occurrence.surface,
                sentiment=label,
                confidence=prediction.confidence,
                review_index=occurrence.review_index,
            )
        )
        key = (occurrence.review_index, occurrence.sentence_index)
        per_sentence.setdefault(key, set()).add(label)

    summaries: list[AspectSummary] = []
    by_aspect = {candidate.aspect: candidate for candidate in candidates}
    for aspect, counts in buckets.items():
        total = sum(counts.values())
        scores = confidences.get(aspect, [])
        summaries.append(
            AspectSummary(
                aspect=aspect,
                source=by_aspect[aspect].source if aspect in by_aspect else "given",
                total=total,
                counts=counts,
                shares={label: round(n / total, 4) for label, n in counts.items()},
                dominant=_dominant(counts),
                mean_confidence=round(sum(scores) / len(scores), 4) if scores else 0.0,
                mixed=len(counts) > 1,
                examples=_pick_examples(examples.get(aspect, [])),
            )
        )
    # Most-discussed first, as the extractor ordered them.
    summaries.sort(key=lambda summary: (-summary.total, summary.aspect.lower()))

    return InsightsSummary(
        reviews=len(texts),
        sentences=sentence_count,
        predictions=len(predictions),
        aspects=summaries,
        mixed_sentences=sum(1 for labels in per_sentence.values() if len(labels) > 1),
        mixed_aspects=sum(1 for summary in summaries if summary.mixed),
        truncated=truncated,
        model_version=predictor.registry.model_version,
    )


def _realign(
    keyed: list[tuple[str, Occurrence]], predictions: list
) -> list[tuple[str, Occurrence]]:
    """Drop the keys whose pair the encoder rejected.

    Walks both lists in order matching on the aspect surface, which is what
    ``predict_many`` echoes back. Order is preserved on both sides, so a
    linear walk is enough.
    """
    aligned: list[tuple[str, Occurrence]] = []
    cursor = 0
    for aspect, occurrence in keyed:
        if cursor >= len(predictions):
            break
        if predictions[cursor].aspect == occurrence.surface:
            aligned.append((aspect, occurrence))
            cursor += 1
    return aligned


def _pick_examples(found: list[Example]) -> list[Example]:
    """Up to ``MAX_EXAMPLES``, preferring one of each polarity.

    One per polarity first, highest confidence each, then fill from what is
    left. An aspect that drew both positive and negative is the interesting
    case, and showing three positives for it would hide exactly that.
    """
    by_label: dict[str, Example] = {}
    for example in sorted(found, key=lambda e: -(e.confidence or 0.0)):
        by_label.setdefault(example.sentiment, example)

    chosen = list(by_label.values())[:MAX_EXAMPLES]
    chosen_ids = {(e.sentence, e.surface, e.sentiment) for e in chosen}
    for example in sorted(found, key=lambda e: -(e.confidence or 0.0)):
        if len(chosen) >= MAX_EXAMPLES:
            break
        if (example.sentence, example.surface, example.sentiment) not in chosen_ids:
            chosen.append(example)
            chosen_ids.add((example.sentence, example.surface, example.sentiment))
    return chosen
