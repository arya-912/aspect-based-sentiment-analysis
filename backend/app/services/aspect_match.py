"""Deciding whether an aspect term actually occurs in a review.

The classifier has three outputs and a softmax over them, so it cannot
abstain: an aspect that appears nowhere in the review still gets a confident
polarity, because some label has to win. Nothing in training taught it
otherwise -- every SemEval aspect term is a literal span of its sentence -- so
aspect attention falls back onto whatever carries the sentence's overall
sentiment, and the answer is sentence-level sentiment wearing an aspect's name.

Teaching the model to abstain means a fourth class, a negative-sampling set
built by hand, and a label space that no longer compares with SemEval-2014
Task 4 or with this project's own no-attention ablation. Refusing the question
is cheaper and more honest than dressing up the answer, so that is what this
module decides; the route turns a False from :func:`aspect_in_text` into a 422.

Matching is on word stems rather than raw substrings, because rejecting a real
aspect over an "-s" ("batteries" against a review that says "battery") would
be a worse failure than the one being fixed.
"""

from __future__ import annotations

import re

#: One word: letters or digits, optionally joined by an apostrophe or a hyphen
#: ("wi-fi", "don't"). `[^\W_]` is `\w` without the underscore, and `re` reads
#: it as Unicode, so an accented aspect term survives. The frontend's
#: `WORD_PATTERN` in `lib/words.ts` is deliberately the same idea of a word:
#: its pre-submit hint has to agree with this verdict or the UI contradicts the
#: server.
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")

#: Possessive endings stripped before any plural rule, straight quote and
#: typographic apostrophe alike -- a review pasted from a word processor
#: carries the curly one.
POSSESSIVES = ("'s", "’s")


def stem(word: str) -> str:
    """Reduce one word to a crude English stem.

    Only possessives and plurals, deliberately: no ``-ing``/``-ed``, and no
    Porter-style vowel measures. This is a gate on whether a noun phrase is
    *present*, and over-stemming fails in the dangerous direction -- it lets an
    aspect match a word that is not it, which is the mistake the gate exists to
    stop. Under-stemming only costs a rejection the caller can see and fix.

    The rules are ordered longest-suffix-first and each is length-guarded, so
    short aspect terms that merely end in the suffix survive: "OS" keeps its
    "s", "bus" and "this" are not mistaken for plurals.

    Both the review and the aspect go through this same function, so a rule
    that is wrong in the same way on both sides still matches; what matters is
    that it is wrong *consistently*.
    """
    stemmed = word.casefold()
    for possessive in POSSESSIVES:
        if stemmed.endswith(possessive):
            stemmed = stemmed[: -len(possessive)]
            break

    if len(stemmed) > 4 and stemmed.endswith("ies"):
        return f"{stemmed[:-3]}y"  # batteries -> battery
    if len(stemmed) > 4 and stemmed.endswith(("sses", "shes", "ches", "xes", "zes")):
        return stemmed[:-2]  # glasses -> glass, boxes -> box, watches -> watch
    if len(stemmed) > 3 and stemmed.endswith("s") and not stemmed.endswith(("ss", "us", "is")):
        return stemmed[:-1]  # screens -> screen, keys -> key
    return stemmed


def stems(text: str) -> list[str]:
    """Every word of ``text``, in order, stemmed.

    Punctuation and spacing fall away with the words kept in order, which is
    what makes the comparison in :func:`aspect_in_text` a whole-word one.
    """
    return [stem(match.group()) for match in WORD_PATTERN.finditer(text)]


def aspect_in_text(text: str, aspect: str) -> bool:
    """Whether ``aspect`` occurs in ``text`` as a contiguous run of whole words.

    Contiguous, because an aspect term is a *phrase*: "hard drive" is not
    present in a review that mentions a hard case and a test drive, and
    matching its words one at a time would report that it is. Whole words,
    because a substring test finds "OS" inside "most".

    Returns True for an aspect holding no words at all (punctuation, a
    zero-width space). That input is already rejected -- by the schema if it is
    blank, by the encoder if it tokenises to nothing -- and those errors say
    something specific, which is better than this gate shadowing them with a
    verdict it has no evidence for.
    """
    needle = stems(aspect)
    if not needle:
        return True

    haystack = stems(text)
    span = len(needle)
    return any(
        haystack[start : start + span] == needle
        for start in range(len(haystack) - span + 1)
    )
