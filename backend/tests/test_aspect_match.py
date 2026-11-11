"""Unit tests for the aspect-presence gate.

These run on strings alone -- no model, no encoder, no fixtures -- because the
gate deliberately decides before the artifact is touched.
"""

from __future__ import annotations

import pytest

from app.services.aspect_match import aspect_in_text, stem, stems


# ---- stemming --------------------------------------------------------- #


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("battery", "battery"),
        ("batteries", "battery"),  # -ies -> -y, the case this exists for
        ("screen", "screen"),
        ("screens", "screen"),
        ("keys", "key"),
        ("glasses", "glass"),  # -sses keeps the double s
        ("boxes", "box"),
        ("watches", "watch"),
        ("drive's", "drive"),  # possessive, straight quote
        ("drive’s", "drive"),  # and the typographic one
        ("SCREEN", "screen"),  # casefolded
    ],
)
def test_stem_normalises_plurals_and_possessives(word, expected):
    assert stem(word) == expected


@pytest.mark.parametrize("word", ["os", "bus", "this", "ios", "ss"])
def test_stem_leaves_short_words_ending_in_s_alone(word):
    """The length guards exist so "OS" is not stemmed to "O".

    A two-letter aspect term is real -- "OS" is a SemEval laptop aspect -- and
    stripping its "s" would make it match nothing.
    """
    assert stem(word) == word


def test_stem_does_not_touch_verb_endings():
    """No -ing/-ed: over-stemming fails in the direction that lets an aspect
    match a word that is not it, which is what the gate exists to stop."""
    assert stem("running") == "running"
    assert stem("priced") == "priced"


def test_stems_drops_punctuation_and_keeps_order():
    assert stems("The battery, really, is great!") == [
        "the",
        "battery",
        "really",
        "is",
        "great",
    ]


# ---- the gate --------------------------------------------------------- #


def test_accepts_an_aspect_that_is_a_span_of_the_review():
    assert aspect_in_text("Battery life is amazing but the screen is dim.", "battery life")


def test_rejects_an_aspect_the_review_never_mentions():
    """The whole point: the model would answer "positive" at 0.98 here."""
    assert not aspect_in_text("Battery life is amazing but the screen is dim.", "keyboard")


@pytest.mark.parametrize(
    ("text", "aspect"),
    [
        ("The battery is great.", "batteries"),
        ("The batteries are great.", "battery"),
        ("Great screen.", "screens"),
        ("The hard drive's speed is fine.", "hard drive"),
        ("Nice SCREEN.", "screen"),
    ],
)
def test_accepts_inflected_and_recased_forms(text, aspect):
    """A plural or a possessive is not an absent aspect.

    Rejecting a real aspect over an "-s" would be a worse failure than the one
    being fixed, which is why this matches stems rather than raw substrings.
    """
    assert aspect_in_text(text, aspect)


def test_requires_the_words_to_be_contiguous():
    """An aspect term is a phrase, not a bag of words."""
    assert not aspect_in_text("The hard case and the test drive.", "hard drive")


def test_requires_whole_words():
    """A substring test would find "OS" inside "most"."""
    assert not aspect_in_text("I like most of it.", "OS")
    assert aspect_in_text("The OS is fine.", "OS")


def test_a_synonym_is_still_absent():
    """The gate checks presence, not meaning -- and says so.

    "display" for a review that says "screen" is rejected. Accepting it would
    need a model of synonymy, which is the job the classifier cannot do either.
    """
    assert not aspect_in_text("Nice display.", "screen")


@pytest.mark.parametrize("aspect", ["​", "...", "   "])
def test_an_aspect_with_no_words_is_left_to_the_other_checks(aspect):
    """True, not False: the schema or the encoder rejects these with a message
    that says something specific, which beats this gate guessing."""
    assert aspect_in_text("Fine laptop", aspect)
