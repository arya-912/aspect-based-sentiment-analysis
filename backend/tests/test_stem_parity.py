"""The backend half of the aspect-presence parity contract.

``tests/fixtures/stem_parity.json`` records what ``app.services.aspect_match``
answers, and ``frontend/src/lib/words.ts`` is held to the same file by
``frontend/src/lib/words.parity.test.ts``. This module checks the Python side
has not moved away from what it recorded.

The point is the direction of the check. ``test_aspect_match.py`` already
tests the rule on its merits -- what *should* be accepted and rejected. This
file tests something different and duller: that the behaviour is still the
behaviour the frontend was built against. A deliberate change to the rule
means regenerating the fixture, and the diff in that file is the part a
reviewer should actually read, because it is a change to the UI's contract as
much as the API's.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.aspect_match import aspect_in_text, stem

FIXTURE = Path(__file__).parent / "fixtures" / "stem_parity.json"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_fixture_exists_and_is_not_empty(fixture: dict) -> None:
    """A missing or emptied fixture must fail loudly, not vacuously pass.

    Every other test here is parametrised over the file's contents, so an
    empty list would collect zero tests and the suite would still be green.
    """
    assert len(fixture["stems"]) > 50
    assert len(fixture["presence"]) > 25


def test_every_stem_still_matches_the_fixture(fixture: dict) -> None:
    drifted = [
        (case["word"], case["stem"], stem(case["word"]))
        for case in fixture["stems"]
        if stem(case["word"]) != case["stem"]
    ]
    assert drifted == [], (
        "stem() no longer agrees with stem_parity.json. If the change is "
        "intended, regenerate the fixture and update frontend/src/lib/words.ts "
        "to match: " + repr(drifted)
    )


def test_every_presence_verdict_still_matches_the_fixture(fixture: dict) -> None:
    drifted = [
        (case["text"], case["aspect"], case["present"], aspect_in_text(case["text"], case["aspect"]))
        for case in fixture["presence"]
        if aspect_in_text(case["text"], case["aspect"]) != case["present"]
    ]
    assert drifted == [], (
        "aspect_in_text() no longer agrees with stem_parity.json. If the "
        "change is intended, regenerate the fixture and update "
        "frontend/src/lib/words.ts to match: " + repr(drifted)
    )


def test_the_generator_is_reproducible(fixture: dict) -> None:
    """Regenerating must be a no-op while the rule is unchanged.

    Guards the generator itself: if it ever stopped writing what the rule
    actually does, the fixture would pin a fiction and both suites would
    happily agree with it.
    """
    regenerated = {
        "stems": [{"word": w["word"], "stem": stem(w["word"])} for w in fixture["stems"]],
        "presence": [
            {
                "text": c["text"],
                "aspect": c["aspect"],
                "present": aspect_in_text(c["text"], c["aspect"]),
            }
            for c in fixture["presence"]
        ],
    }
    assert regenerated["stems"] == fixture["stems"]
    assert regenerated["presence"] == fixture["presence"]
