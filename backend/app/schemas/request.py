"""Request bodies."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """One (sentence, aspect) pair to classify.

    The aspect is expected to be a phrase from ``text``; the model will still
    answer if it is not, but the prediction is only meaningful when the aspect
    actually appears in the sentence.
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
