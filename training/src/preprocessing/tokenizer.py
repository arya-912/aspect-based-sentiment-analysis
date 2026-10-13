"""Sentence-pair encoding of (sentence, aspect) examples for BERT-style models."""

from __future__ import annotations

import logging

import torch
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)


class ABSAEncoder:
    """Encode ``[CLS] text [SEP] aspect [SEP]`` with per-segment token masks.

    Beyond the usual ``input_ids``/``attention_mask``/``token_type_ids`` this
    emits two boolean masks the model pools over:

    ``context_mask``
        True on the review's own tokens only -- no ``[CLS]``/``[SEP]``, no
        padding, and none of the aspect segment.
    ``aspect_mask``
        True on the aspect segment's tokens only, same exclusions.

    Long reviews are truncated with ``truncation="only_first"`` so the aspect
    phrase always survives intact; the aspect is what the prediction is about.
    """

    def __init__(self, model_name: str, max_length: int = 128) -> None:
        if max_length < 4:
            raise ValueError(
                f"max_length must leave room for [CLS]/[SEP] markers, got {max_length}"
            )
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def encode(
        self, text: str, aspect: str, label: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Encode one (sentence, aspect) pair into fixed-length tensors.

        Args:
            text: the full review sentence.
            aspect: the aspect term, a literal span of ``text``.
            label: optional class id; when given a ``labels`` tensor is added.

        Returns:
            Dict of tensors, each of length ``max_length`` (``labels`` scalar).

        Raises:
            ValueError: if either side tokenises to nothing, or if truncation
                left no context tokens to pool over.
        """
        text = "" if text is None else str(text)
        aspect = "" if aspect is None else str(aspect)

        if not self.tokenizer.tokenize(text):
            raise ValueError(f"text produced zero tokens: {text!r}")
        if not self.tokenizer.tokenize(aspect):
            raise ValueError(f"aspect produced zero tokens: {aspect!r}")

        encoded = self.tokenizer(
            text,
            aspect,
            add_special_tokens=True,
            truncation="only_first",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        token_type_ids = encoded["token_type_ids"].squeeze(0)
        special = encoded["special_tokens_mask"].squeeze(0).bool()

        # Real, non-special tokens; segment 0 is the review, segment 1 the aspect.
        real = attention_mask.bool() & ~special
        context_mask = real & (token_type_ids == 0)
        aspect_mask = real & (token_type_ids == 1)

        if not bool(context_mask.any()):
            raise ValueError(
                f"no context tokens survived truncation at max_length="
                f"{self.max_length} for text={text!r}"
            )
        if not bool(aspect_mask.any()):
            raise ValueError(
                f"no aspect tokens survived encoding for aspect={aspect!r}"
            )

        example: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "context_mask": context_mask,
            "aspect_mask": aspect_mask,
        }
        if label is not None:
            example["labels"] = torch.tensor(int(label), dtype=torch.long)
        return example

    def decode(self, input_ids: torch.Tensor) -> str:
        """Human-readable round-trip of an encoded example, for debugging."""
        return self.tokenizer.decode(input_ids, skip_special_tokens=False)

    def save(self, directory) -> None:
        """Persist the tokenizer next to a trained checkpoint."""
        self.tokenizer.save_pretrained(str(directory))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ABSAEncoder(model_name={self.model_name!r}, max_length={self.max_length})"
