"""Request bodies."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """One (sentence, aspect) pair to classify.

    The aspect must be a phrase from ``text``. The model will answer whether
    or not it is -- it has no way to abstain -- so the route checks presence
    itself and refuses with a 422, rather than returning a confident polarity
    about something the review never mentions. The check lives in
    ``services/aspect_match.py``; it matches whole words on stems, so a plural
    is not a rejection.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The review sentence.",
        examples=["Battery life is amazing but the screen is dim."],
    )
    aspect: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="The aspect term to judge, normally a span of `text`.",
        examples=["screen"],
    )

    @field_validator("text", "aspect")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Treat whitespace-only input as missing.

        ``min_length`` alone accepts `" "`, which then reaches the tokenizer
        and fails there as a 400. Rejecting it here keeps it a 422 alongside
        every other malformed field.
        """
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class InsightsRequest(BaseModel):
    """A set of reviews to break down by aspect.

    ``aspects`` is optional. Left out, the aspects are extracted from the
    corpora's gold vocabulary (see ``services/aspects.py``) -- the model does
    not find aspects, so something has to, and a client that wants control
    names them itself.
    """

    reviews: list[str] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Raw review texts; each may run to several sentences.",
        examples=[["Battery life is amazing but the screen is dim."]],
    )
    aspects: list[str] | None = Field(
        None,
        max_length=24,
        description=(
            "Judge these aspects instead of extracting them. An aspect is "
            "only reported for the sentences that actually mention it."
        ),
    )
    domain: str | None = Field(
        None,
        description=(
            "Restrict the extraction vocabulary: 'laptop' or 'restaurant'. "
            "Ignored when `aspects` is given."
        ),
    )
    max_aspects: int = Field(
        12, ge=1, le=24, description="How many aspects to report, most-discussed first."
    )

    @field_validator("reviews")
    @classmethod
    def _reject_all_blank(cls, value: list[str]) -> list[str]:
        """At least one review has to carry text.

        A list of empty strings passes ``min_length`` on the list itself and
        then produces an empty summary, which looks like a model failure
        rather than an empty request.
        """
        if not any(item and item.strip() for item in value):
            raise ValueError("at least one review must contain text")
        for item in value:
            if len(item) > 5000:
                raise ValueError("each review must be at most 5000 characters")
        return value

    @field_validator("aspects")
    @classmethod
    def _clean_aspects(cls, value: list[str] | None) -> list[str] | None:
        """Drop blanks and duplicates, keeping the caller's order."""
        if value is None:
            return None
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in value:
            trimmed = " ".join(item.split())
            if not trimmed or len(trimmed) > 200:
                continue
            key = trimmed.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(trimmed)
        return cleaned or None
