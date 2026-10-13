"""Turning a (text, aspect) pair into a prediction."""

from __future__ import annotations

import logging
from typing import Any

import torch
from fastapi import HTTPException, status

from ..schemas.response import PredictResponse
from .model_registry import ModelRegistry, registry as default_registry

LOGGER = logging.getLogger(__name__)

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
            A :class:`PredictResponse`.

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
