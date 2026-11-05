"""Turning a (text, aspect) pair into a prediction."""

from __future__ import annotations

import logging
from typing import Any

import torch
from fastapi import HTTPException, status

from ..schemas.response import PredictResponse, TokenAttention
from .model_registry import ModelRegistry, registry as default_registry

LOGGER = logging.getLogger(__name__)

#: How many attended tokens to return. The model weights every review token,
#: but a 128-slot distribution is not useful to a client and the interesting
#: mass sits in the first few; the rest is padding-adjacent noise.
MAX_ATTENTION_TOKENS = 8

#: Tensors the model's forward signature accepts, in order.
MODEL_INPUTS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "aspect_mask",
    "context_mask",
)


class PredictorService:
    """Stateless wrapper around the loaded model.

    Takes the registry as a constructor argument so tests can inject a fake
    one without patching module globals.
    """

    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self.registry = model_registry or default_registry

    # ---- single prediction -------------------------------------------- #

    def predict(self, text: str, aspect: str) -> PredictResponse:
        """Classify the sentiment expressed about ``aspect`` in ``text``.

        Args:
            text: the review sentence.
            aspect: the aspect term, normally a span of ``text``.

        Returns:
            A :class:`PredictResponse`, carrying the full label distribution
            and the most-attended review tokens alongside the chosen label.

        Raises:
            HTTPException: 400 when the encoder rejects the input (for example
                an aspect that tokenises to nothing).
            ModelNotReadyError: 503, when the model has not loaded.
        """
        model = self.registry.get_model()
        encoder = self.registry.get_encoder()
        label_map = self.registry.get_label_map()

        # 1. encode -- a tokenizer rejection is the caller's fault, not ours.
        try:
            encoded = encoder.encode(text, aspect)
        except ValueError as exc:
            LOGGER.info("rejected input (aspect=%r): %s", aspect, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

        # 2. add the batch dimension and move to the model's device.
        device = self._model_device(model)
        batch = {
            name: encoded[name].unsqueeze(0).to(device)
            for name in MODEL_INPUTS
            if name in encoded
        }

        # 3-5. forward, softmax, argmax.
        with torch.no_grad():
            output = model(**batch)
        probabilities = torch.softmax(output["logits"].float(), dim=-1)
        predicted_id = int(probabilities.argmax(dim=-1).item())

        return PredictResponse(
            aspect=aspect,
            sentiment=self._decode(label_map, predicted_id),
            confidence=round(float(probabilities[0, predicted_id].item()), 4),
            scores=self._scores(label_map, probabilities[0]),
            attention=self._attention(encoder, output, batch),
            model_version=self.registry.model_version,
        )

    # ---- helpers ------------------------------------------------------ #

    @staticmethod
    def _model_device(model: Any) -> torch.device:
        """Where the model's parameters live.

        A test double has no parameters, so fall back to CPU rather than
        raising on ``next()`` of an empty iterator.
        """
        try:
            return next(model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    @staticmethod
    def _scores(label_map: Any, row: torch.Tensor) -> dict[str, float] | None:
        """The whole softmax distribution, keyed by label name.

        Returns ``None`` rather than guessing when the label map's names do
        not line up with the logits, since a mislabelled distribution is
        worse than an absent one.
        """
        labels = getattr(label_map, "labels", None)
        if not labels or len(labels) != row.numel():
            return None
        return {str(name): round(float(row[i].item()), 4) for i, name in enumerate(labels)}

    @classmethod
    def _attention(
        cls, encoder: Any, output: Any, batch: dict[str, torch.Tensor]
    ) -> list[TokenAttention] | None:
        """The most-attended review tokens, highest weight first.

        The model computes this on every forward pass; without it the one
        feature that distinguishes this architecture is invisible to a client.

        Returns ``None`` when the pieces needed are missing -- a model that
        reports no attention, or an encoder without a tokenizer to turn ids
        back into words -- so a leaner stand-in still serves predictions.
        """
        weights = output.get("attention") if isinstance(output, dict) else None
        mask = batch.get("context_mask")
        convert = getattr(getattr(encoder, "tokenizer", None), "convert_ids_to_tokens", None)
        if weights is None or mask is None or "input_ids" not in batch or not callable(convert):
            return None

        tokens = convert(batch["input_ids"][0].cpu())
        kept = [
            (str(token), float(weight))
            for token, weight, keep in zip(
                tokens, weights[0].cpu().tolist(), mask[0].cpu().tolist()
            )
            if keep
        ]
        if not kept:
            return None

        merged = cls._merge_wordpieces(kept)
        merged.sort(key=lambda pair: -pair[1])
        return [
            # Clamped because the schema bounds the weight, and a model that
            # returned something unnormalised should not turn a good
            # prediction into a response-validation 500.
            TokenAttention(token=token, weight=round(min(max(weight, 0.0), 1.0), 4))
            for token, weight in merged[:MAX_ATTENTION_TOKENS]
        ]

    @staticmethod
    def _merge_wordpieces(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Fold ``##`` continuation pieces into the word they continue.

        Attention is computed per WordPiece, so a rare word arrives split --
        ``un`` + ``##responsive``. Rendering those as separate bars reads as
        noise. The weights are summed because they are shares of one
        distribution, so a word's share is the sum of its pieces'.
        """
        merged: list[tuple[str, float]] = []
        for token, weight in pairs:
            if token.startswith("##") and merged:
                previous_token, previous_weight = merged[-1]
                merged[-1] = (previous_token + token[2:], previous_weight + weight)
            else:
                merged.append((token, weight))
        return merged

    @staticmethod
    def _decode(label_map: Any, label_id: int) -> str:
        """Class id -> label name, tolerating either accessor.

        ``LabelMap`` exposes both ``id2label`` and ``decode()``; a stub may
        provide only one.
        """
        id2label = getattr(label_map, "id2label", None)
        if isinstance(id2label, dict):
            if label_id in id2label:
                return str(id2label[label_id])
            if str(label_id) in id2label:  # a JSON round-trip stringifies keys
                return str(id2label[str(label_id)])
        decode = getattr(label_map, "decode", None)
        if callable(decode):
            return str(decode(label_id))
        raise RuntimeError(f"cannot decode label id {label_id} from {label_map!r}")


def get_predictor() -> PredictorService:
    """FastAPI dependency returning a predictor over the shared registry."""
    return PredictorService()
