"""The aspect extractor: the step the model does not do.

These tests need no model and no artifact -- extraction is lexicon matching
on strings. They use a small fixture lexicon rather than the real 2085-term
export, so a case is readable and a corpus change cannot silently alter what
is under test.

The invariant that matters most is the last one: every aspect this returns is
a verbatim span of the sentence it was found in, so a prediction about it can
never be refused by the 422 presence gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.aspect_match import aspect_in_text
from app.services.aspects import (
    extract_aspects,
    load_lexicon,
    locate_aspects,
    split_sentences,
    words_of,
)

#: A readable stand-in for the exported vocabulary. Counts straddle the
#: common/rare threshold of 3 on purpose.
FIXTURE_TERMS = [
    {"term": "battery life", "count": 55, "domains": ["laptop"]},
    {"term": "battery", "count": 48, "domains": ["laptop"]},
    {"term": "screen", "count": 63, "domains": ["laptop"]},
    {"term": "keyboard", "count": 51, "domains": ["laptop"]},
    {"term": "hard drive", "count": 29, "domains": ["laptop"]},
    {"term": "price", "count": 100, "domains": ["laptop", "restaurant"]},
    {"term": "food", "count": 376, "domains": ["restaurant"]},
    {"term": "service", "count": 238, "domains": ["restaurant"]},
    {"term": "os", "count": 12, "domains": ["laptop"]},
    # The rare tier: tried only when the common tier finds nothing.
    {"term": "fan noise", "count": 2, "domains": ["laptop"]},
    {"term": "bezel", "count": 1, "domains": ["laptop"]},
]


@pytest.fixture
def lexicon(tmp_path: Path) -> str:
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps({"terms": FIXTURE_TERMS}), encoding="utf-8")
    load_lexicon.cache_clear()
    yield str(path)
    load_lexicon.cache_clear()


def extract(reviews, lexicon, **kwargs):
    return extract_aspects(reviews, lexicon_path=lexicon, **kwargs)


# ---- sentence splitting ----------------------------------------------- #


class TestSplitSentences:
    def test_splits_on_terminal_punctuation(self):
        assert split_sentences("The screen is dim. The keyboard is fine!") == [
            "The screen is dim.",
            "The keyboard is fine!",
        ]

    def test_keeps_a_single_sentence_whole(self):
        assert split_sentences("Battery life is amazing but the screen is dim.") == [
            "Battery life is amazing but the screen is dim."
        ]

    def test_text_with_no_terminal_punctuation_is_one_sentence(self):
        assert split_sentences("no full stop here") == ["no full stop here"]

    def test_drops_empty_fragments(self):
        assert split_sentences("  One.   Two.   ") == ["One.", "Two."]

    def test_returns_nothing_for_blank(self):
        assert split_sentences("   ") == []
        assert split_sentences("") == []

    def test_caps_a_runaway_sentence(self):
        # An unpunctuated wall of text would otherwise become one "sentence"
        # whose tail the encoder silently truncates anyway.
        long = "word " * 400
        assert len(split_sentences(long)[0]) <= 600


class TestWordsOf:
    def test_reports_offsets_that_slice_the_word_back_out(self):
        sentence = "The hard-drive is loud."
        for word in words_of(sentence):
            assert sentence[word.start : word.end] == word.surface

    def test_stems_each_word(self):
        stems = {word.surface: word.stem for word in words_of("the batteries died")}
        assert stems["batteries"] == "battery"


# ---- extraction -------------------------------------------------------- #


class TestExtractAspects:
    def test_finds_a_term_from_the_vocabulary(self, lexicon):
        found = extract(["The screen is dim."], lexicon)
        assert [c.aspect for c in found] == ["screen"]
        assert found[0].source == "corpus"
        assert found[0].total == 1

    def test_prefers_the_longer_overlapping_term(self, lexicon):
        # "battery life" and "battery" both match; reporting both would show
        # two aspects where the review discussed one.
        found = extract(["Battery life is amazing."], lexicon)
        assert [c.aspect for c in found] == ["Battery life"]

    def test_the_shorter_term_still_matches_on_its_own(self, lexicon):
        found = extract(["The battery died."], lexicon)
        assert [c.aspect for c in found] == ["battery"]

    def test_counts_occurrences_across_reviews(self, lexicon):
        found = extract(["The screen is dim.", "The screen is bright."], lexicon)
        assert found[0].aspect == "screen"
        assert found[0].total == 2

    def test_orders_by_how_much_they_are_discussed(self, lexicon):
        found = extract(
            ["The screen is dim.", "The screen is bright.", "The keyboard is fine."],
            lexicon,
        )
        assert [c.aspect for c in found] == ["screen", "keyboard"]

    def test_matches_an_inflected_mention(self, lexicon):
        # Stem-based, like the gate: "batteries" is a mention of "battery".
        found = extract(["The batteries died."], lexicon)
        assert found[0].total == 1
        assert found[0].occurrences[0].surface == "batteries"

    def test_labels_the_aspect_with_its_commonest_surface_form(self, lexicon):
        found = extract(["Screen is dim.", "The screen is dim.", "screen again."], lexicon)
        # Lower case wins 2-1, and reads better than the lexicon's own key.
        assert found[0].aspect == "screen"

    def test_matches_whole_words_only(self, lexicon):
        # "os" lives inside "most"; a substring test would find it.
        found = extract(["most of it is fine"], lexicon)
        assert found == []

    def test_requires_a_phrase_to_be_contiguous(self, lexicon):
        assert extract(["a hard case and a test drive"], lexicon) == []
        assert [c.aspect for c in extract(["it has a hard drive"], lexicon)] == ["hard drive"]

    def test_restricts_the_vocabulary_by_domain(self, lexicon):
        reviews = ["The food was great and the screen is dim."]
        assert [c.aspect for c in extract(reviews, lexicon, domain="restaurant")] == ["food"]
        assert [c.aspect for c in extract(reviews, lexicon, domain="laptop")] == ["screen"]

    def test_honours_the_limit(self, lexicon):
        reviews = ["screen keyboard price food service"]
        assert len(extract(reviews, lexicon, limit=2)) == 2

    def test_finds_nothing_in_an_unknown_domain(self, lexicon):
        # The stated limitation: an aspect no annotator marked is invisible.
        assert extract(["The noise cancelling is superb."], lexicon) == []

    def test_an_empty_review_set_is_not_an_error(self, lexicon):
        assert extract([], lexicon) == []
        assert extract(["   "], lexicon) == []

    def test_records_which_review_and_sentence_each_mention_came_from(self, lexicon):
        found = extract(["First. The screen is dim.", "The keyboard is fine."], lexicon)
        screen = next(c for c in found if c.aspect == "screen")
        assert screen.occurrences[0].review_index == 0
        assert screen.occurrences[0].sentence_index == 1
        keyboard = next(c for c in found if c.aspect == "keyboard")
        assert keyboard.occurrences[0].review_index == 1


class TestTiers:
    def test_the_rare_tier_is_skipped_when_the_common_tier_matches(self, lexicon):
        # "fan noise" is rare; "screen" is common and in the same sentence.
        found = extract(["The screen is dim and the fan noise is awful."], lexicon)
        assert [c.aspect for c in found] == ["screen"]

    def test_the_rare_tier_rescues_a_sentence_the_common_tier_misses(self, lexicon):
        found = extract(["The fan noise is awful."], lexicon)
        assert [c.aspect for c in found] == ["fan noise"]
        assert found[0].source == "corpus-rare"

    def test_a_rare_term_is_reported_as_rare(self, lexicon):
        found = extract(["The bezel is huge."], lexicon)
        assert found[0].source == "corpus-rare"
        assert found[0].lexicon_count == 1


class TestLocateAspects:
    def test_finds_a_caller_supplied_aspect(self):
        found = locate_aspects(["The screen is dim."], ["screen"])
        assert [c.aspect for c in found] == ["screen"]
        assert found[0].source == "given"

    def test_reports_nothing_for_an_aspect_that_is_absent(self):
        assert locate_aspects(["The screen is dim."], ["keyboard"]) == []

    def test_needs_no_lexicon_entry(self):
        # The storefront names aspects the vocabulary may not rank highly.
        found = locate_aspects(["The noise cancelling is superb."], ["noise cancelling"])
        assert [c.aspect for c in found] == ["noise cancelling"]

    def test_matches_an_inflected_mention(self):
        found = locate_aspects(["The batteries died."], ["battery"])
        assert found[0].total == 1
        # Grouped under the caller's label, but found as the sentence spells it.
        assert found[0].aspect == "battery"
        assert found[0].occurrences[0].surface == "batteries"

    def test_skips_an_aspect_with_no_words(self):
        assert locate_aspects(["The screen is dim."], ["...", ""]) == []

    def test_orders_by_how_much_each_is_discussed(self):
        found = locate_aspects(
            ["screen and screen", "keyboard"], ["keyboard", "screen"]
        )
        assert [c.aspect for c in found] == ["screen", "keyboard"]


class TestLexiconLoading:
    def test_an_absent_lexicon_is_empty_not_fatal(self, tmp_path):
        load_lexicon.cache_clear()
        try:
            assert load_lexicon(str(tmp_path / "nope.json")) == ()
        finally:
            load_lexicon.cache_clear()

    def test_the_real_export_is_present_and_loads(self):
        # Guards the committed data file: it ships in the repo, unlike the
        # corpus it was built from, so a missing one is a packaging bug.
        load_lexicon.cache_clear()
        try:
            entries = load_lexicon()
            assert len(entries) > 1000
            assert any(entry.term == "battery life" for entry in entries)
            # Longest-first, which is what makes overlap resolution work.
            lengths = [len(entry.stems) for entry in entries]
            assert lengths == sorted(lengths, reverse=True)
        finally:
            load_lexicon.cache_clear()


class TestTheGateCanNeverRefuseWhatWasExtracted:
    """Every extracted aspect is a verbatim span of its own sentence.

    This is the property that lets the dashboard predict in bulk without
    handling 422s: a pair exists only because the aspect was found in that
    sentence, so ``aspect_in_text`` -- the same rule, the same stemmer -- must
    agree.
    """

    REVIEWS = [
        "Battery life is amazing but the screen is dim.",
        "The batteries died and the hard drive's noise is awful.",
        "Great food, terrible service.",
        "The OS is fine. The fan noise is awful.",
    ]

    def test_extracted_surfaces_are_literal_slices(self, lexicon):
        for candidate in extract(self.REVIEWS, lexicon, limit=24):
            for occurrence in candidate.occurrences:
                assert occurrence.surface in occurrence.sentence

    def test_the_presence_gate_accepts_every_extracted_pair(self, lexicon):
        checked = 0
        for candidate in extract(self.REVIEWS, lexicon, limit=24):
            for occurrence in candidate.occurrences:
                assert aspect_in_text(occurrence.sentence, occurrence.surface)
                checked += 1
        assert checked > 0, "fixture produced no pairs to check"

    def test_the_gate_accepts_located_pairs_too(self):
        aspects = ["screen", "battery", "food", "service", "hard drive"]
        checked = 0
        for candidate in locate_aspects(self.REVIEWS, aspects):
            for occurrence in candidate.occurrences:
                assert aspect_in_text(occurrence.sentence, occurrence.surface)
                checked += 1
        assert checked > 0
