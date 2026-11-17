"""Aggregating per-aspect sentiment, with the model stubbed out.

The arithmetic is what matters here -- shares, the dominant label, the
mixed-sentence count, which examples survive -- so the predictor is a stub
that returns whatever labels a test dictates. Using the real model would make
these tests about the model's opinions instead of about the aggregation, and
they would change whenever the artifact did.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.response import PredictResponse
from app.services.aspects import load_lexicon
from app.services.insights import MAX_EXAMPLES, _dominant, summarise


class StubRegistry:
    model_version = "absa-stub"


class StubPredictor:
    """Returns a scripted label per call, in order.

    ``script`` is consumed one entry per pair, so a test spells out exactly
    what the model "said" about each (sentence, aspect) pair.
    """

    def __init__(self, script: list[str], confidences: list[float] | None = None) -> None:
        self.script = list(script)
        self.confidences = list(confidences) if confidences else None
        self.registry = StubRegistry()
        self.batches: list[list[tuple[str, str]]] = []

    def predict_many(self, pairs, *, include_attention=False, batch_size=32):
        self.batches.append(list(pairs))
        out = []
        for index, (_, aspect) in enumerate(pairs):
            label = self.script[index % len(self.script)]
            confidence = (
                self.confidences[index % len(self.confidences)]
                if self.confidences
                else 0.9
            )
            out.append(
                PredictResponse(
                    aspect=aspect,
                    sentiment=label,
                    confidence=confidence,
                    scores=None,
                    attention=None,
                    aspect_in_text=True,
                    model_version="absa-stub",
                )
            )
        return out


FIXTURE_TERMS = [
    {"term": "screen", "count": 63, "domains": ["laptop"]},
    {"term": "keyboard", "count": 51, "domains": ["laptop"]},
    {"term": "battery", "count": 48, "domains": ["laptop"]},
]


@pytest.fixture(autouse=True)
def fixture_lexicon(tmp_path: Path, monkeypatch):
    """Point the extractor at a tiny vocabulary for every test in this file."""
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps({"terms": FIXTURE_TERMS}), encoding="utf-8")
    load_lexicon.cache_clear()
    monkeypatch.setattr("app.services.aspects.LEXICON_PATH", path)
    yield
    load_lexicon.cache_clear()


class TestShapeOfTheSummary:
    def test_counts_reviews_and_sentences(self):
        predictor = StubPredictor(["positive"])
        summary = summarise(
            ["The screen is dim. The keyboard is fine.", "The battery died."],
            predictor=predictor,
        )
        assert summary.reviews == 2
        assert summary.sentences == 3
        assert summary.model_version == "absa-stub"

    def test_ignores_blank_reviews_without_counting_them(self):
        summary = summarise(
            ["The screen is dim.", "   ", ""], predictor=StubPredictor(["positive"])
        )
        assert summary.reviews == 1

    def test_no_aspects_found_is_an_empty_summary_not_an_error(self):
        # Reviews about something the vocabulary never saw.
        summary = summarise(
            ["The noise cancelling is superb."], predictor=StubPredictor(["positive"])
        )
        assert summary.aspects == []
        assert summary.predictions == 0
        assert summary.mixed_sentences == 0

    def test_makes_one_prediction_per_aspect_occurrence(self):
        predictor = StubPredictor(["positive"])
        summary = summarise(
            ["The screen is dim and the keyboard is fine."], predictor=predictor
        )
        assert summary.predictions == 2
        assert len(predictor.batches) == 1


class TestAggregation:
    def test_shares_sum_to_one_per_aspect(self):
        predictor = StubPredictor(["positive", "negative", "positive"])
        summary = summarise(
            ["The screen is good.", "The screen is bad.", "The screen is good."],
            predictor=predictor,
        )
        screen = summary.aspects[0]
        assert screen.total == 3
        assert sum(screen.shares.values()) == pytest.approx(1.0)
        assert screen.counts == {"positive": 2, "negative": 1}
        assert screen.shares["positive"] == pytest.approx(2 / 3, abs=1e-4)

    def test_the_dominant_label_is_the_commonest(self):
        predictor = StubPredictor(["negative", "negative", "positive"])
        summary = summarise(
            ["The screen is bad.", "The screen is bad.", "The screen is good."],
            predictor=predictor,
        )
        assert summary.aspects[0].dominant == "negative"

    def test_mixed_is_true_only_when_polarities_differ(self):
        one = summarise(
            ["The screen is good.", "The screen is great."],
            predictor=StubPredictor(["positive"]),
        )
        assert one.aspects[0].mixed is False
        assert one.mixed_aspects == 0

        two = summarise(
            ["The screen is good.", "The screen is bad."],
            predictor=StubPredictor(["positive", "negative"]),
        )
        assert two.aspects[0].mixed is True
        assert two.mixed_aspects == 1

    def test_mean_confidence_is_averaged_over_the_aspect(self):
        predictor = StubPredictor(["positive"], confidences=[0.8, 1.0])
        summary = summarise(
            ["The screen is good.", "The screen is great."], predictor=predictor
        )
        assert summary.aspects[0].mean_confidence == pytest.approx(0.9)

    def test_aspects_are_ordered_by_how_much_they_are_discussed(self):
        predictor = StubPredictor(["positive"])
        summary = summarise(
            ["The screen is dim.", "The screen is dim.", "The keyboard is fine."],
            predictor=predictor,
        )
        assert [a.aspect for a in summary.aspects] == ["screen", "keyboard"]

    def test_honours_max_aspects(self):
        predictor = StubPredictor(["positive"])
        summary = summarise(
            ["screen keyboard battery"], predictor=predictor, max_aspects=2
        )
        assert len(summary.aspects) == 2


class TestDominantTieBreak:
    """A tie is not a verdict, and must not flatter the product."""

    def test_a_fifty_fifty_aspect_is_headlined_negative(self):
        predictor = StubPredictor(["positive", "negative"])
        summary = summarise(
            ["The screen is good.", "The screen is bad."], predictor=predictor
        )
        assert summary.aspects[0].dominant == "negative"
        assert summary.aspects[0].mixed is True

    @pytest.mark.parametrize(
        "counts, expected",
        [
            ({"negative": 2, "positive": 2}, "negative"),
            ({"neutral": 2, "positive": 2}, "neutral"),
            ({"negative": 1, "neutral": 1, "positive": 1}, "negative"),
            ({"positive": 3, "negative": 1}, "positive"),
            ({"neutral": 5}, "neutral"),
        ],
    )
    def test_the_rule_directly(self, counts, expected):
        assert _dominant(counts) == expected

    def test_an_unexpected_label_name_does_not_crash(self):
        # An artifact could ship other label names; they rank last.
        assert _dominant({"mixed": 2, "positive": 2}) == "positive"


class TestMixedSentences:
    """The headline statistic: sentences a single label cannot describe."""

    def test_counts_a_sentence_whose_aspects_disagree(self):
        predictor = StubPredictor(["positive", "negative"])
        summary = summarise(
            ["The screen is dim but the keyboard is fine."], predictor=predictor
        )
        assert summary.mixed_sentences == 1

    def test_does_not_count_a_sentence_whose_aspects_agree(self):
        predictor = StubPredictor(["positive"])
        summary = summarise(
            ["The screen is great and the keyboard is fine."], predictor=predictor
        )
        assert summary.mixed_sentences == 0

    def test_does_not_count_a_sentence_with_one_aspect(self):
        predictor = StubPredictor(["positive", "negative"])
        summary = summarise(["The screen is dim."], predictor=predictor)
        assert summary.mixed_sentences == 0

    def test_counts_per_sentence_not_per_review(self):
        # Two sentences, each internally mixed, in one review.
        predictor = StubPredictor(["positive", "negative"])
        summary = summarise(
            [
                "The screen is dim but the keyboard is fine. "
                "The battery is good but the screen is bad."
            ],
            predictor=predictor,
        )
        assert summary.mixed_sentences == 2


class TestExamples:
    def test_keeps_one_example_of_each_polarity_first(self):
        # Three positives and one negative: the negative is the interesting
        # one and must not be crowded out.
        predictor = StubPredictor(
            ["positive", "positive", "positive", "negative"],
            confidences=[0.99, 0.98, 0.97, 0.5],
        )
        summary = summarise(
            [
                "The screen is good.",
                "The screen is great.",
                "The screen is fine.",
                "The screen is bad.",
            ],
            predictor=predictor,
        )
        labels = {example.sentiment for example in summary.aspects[0].examples}
        assert "negative" in labels
        assert "positive" in labels

    def test_caps_the_example_count(self):
        predictor = StubPredictor(["positive"])
        summary = summarise(
            [f"The screen is good number {i}." for i in range(10)], predictor=predictor
        )
        assert len(summary.aspects[0].examples) <= MAX_EXAMPLES

    def test_an_example_carries_its_sentence_and_review_index(self):
        predictor = StubPredictor(["negative"])
        summary = summarise(
            ["The keyboard is fine.", "The screen is dim."], predictor=predictor
        )
        screen = next(a for a in summary.aspects if a.aspect == "screen")
        example = screen.examples[0]
        assert example.sentence == "The screen is dim."
        assert example.surface == "screen"
        assert example.review_index == 1


class TestCallerSuppliedAspects:
    def test_judges_the_named_aspects_instead_of_extracting(self):
        predictor = StubPredictor(["positive"])
        summary = summarise(
            ["The noise cancelling is superb and the screen is dim."],
            predictor=predictor,
            aspects=["noise cancelling"],
        )
        assert [a.aspect for a in summary.aspects] == ["noise cancelling"]
        assert summary.aspects[0].source == "given"

    def test_skips_a_named_aspect_no_review_mentions(self):
        predictor = StubPredictor(["positive"])
        summary = summarise(
            ["The screen is dim."], predictor=predictor, aspects=["battery", "screen"]
        )
        assert [a.aspect for a in summary.aspects] == ["screen"]

    def test_asks_the_model_about_the_sentence_s_own_wording(self):
        # Grouped under "battery", but the pair sent to the model says
        # "batteries" -- the text that is really in the sentence.
        predictor = StubPredictor(["negative"])
        summarise(
            ["The batteries died."], predictor=predictor, aspects=["battery"]
        )
        assert predictor.batches[0] == [("The batteries died.", "batteries")]


class TestTheForwardPassCeiling:
    def test_truncates_and_says_so(self):
        predictor = StubPredictor(["positive"])
        reviews = [f"The screen is dim number {i}." for i in range(20)]
        summary = summarise(reviews, predictor=predictor, max_pairs=5)
        assert summary.truncated is True
        assert summary.predictions == 5

    def test_does_not_claim_truncation_when_it_fitted(self):
        predictor = StubPredictor(["positive"])
        summary = summarise(["The screen is dim."], predictor=predictor, max_pairs=5)
        assert summary.truncated is False

    def test_keeps_every_aspect_represented_under_the_ceiling(self):
        # Spending the budget on the most-mentioned aspect would silently
        # change which aspects the answer is about.
        predictor = StubPredictor(["positive"])
        reviews = [f"The screen is dim number {i}." for i in range(10)]
        reviews.append("The keyboard is fine.")
        summary = summarise(reviews, predictor=predictor, max_pairs=4)
        assert summary.truncated is True
        assert {a.aspect for a in summary.aspects} == {"screen", "keyboard"}


class TestUnencodablePairs:
    def test_a_dropped_pair_does_not_misattribute_the_rest(self):
        """``predict_many`` skips what the encoder rejects.

        The results then no longer line up with the pairs by index, and a
        naive zip would file one aspect's sentiment under another's name.
        """

        class DroppingPredictor(StubPredictor):
            def predict_many(self, pairs, *, include_attention=False, batch_size=32):
                results = super().predict_many(pairs, include_attention=include_attention)
                return results[1:]  # the first pair "failed to encode"

        predictor = DroppingPredictor(["negative"])
        summary = summarise(
            ["The screen is dim.", "The keyboard is fine."], predictor=predictor
        )
        # One prediction survived, and it is attributed to a real aspect
        # whose own surface it carries.
        assert summary.predictions == 1
        assert len(summary.aspects) == 1
        aspect = summary.aspects[0]
        assert aspect.examples[0].surface in aspect.examples[0].sentence
