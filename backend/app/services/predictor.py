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
            A :class:`PredictResponse`, carrying the full label distribution,
            the most-attended review tokens, and whether the aspect was
            actually present, alongside the chosen label.

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
            aspect_in_text=self._aspect_in_text(batch),
            model_version=self.registry.model_version,
        )

    # ---- batched prediction ------------------------------------------- #

    def predict_many(
        self,
        pairs: list[tuple[str, str]],
        *,
        include_attention: bool = False,
        batch_size: int = 32,
    ) -> list[PredictResponse]:
        """Classify many (text, aspect) pairs in as few forward passes as possible.

        The insights dashboard turns one pasted review set into hundreds of
        pairs -- every aspect against every sentence that mentions it -- and
        one forward pass each is the difference between a demo that answers
        and a demo that times out. The encoder pads to ``max_length``, so the
        examples stack without further work.

        Attention is off by default: it is the most interesting field in a
        single prediction and pure weight in an aggregate, where nothing
        renders per-pair token weights.

        A pair the encoder rejects is skipped rather than failing the batch,
        because one malformed sentence in a pasted file should not lose the
        other 200. Skipped pairs are logged and simply absent from the
        result, so callers must match on ``aspect`` rather than by index.

        Args:
            pairs: (sentence, aspect) pairs. The aspect should be a span of
                its sentence; nothing here re-checks that.
            include_attention: return per-pair attention as well.
            batch_size: pairs per forward pass.

        Returns:
            One :class:`PredictResponse` per *encodable* pair, in input order.

        Raises:
            ModelNotReadyError: 503, when the model has not loaded.
        """
        model = self.registry.get_model()
        encoder = self.registry.get_encoder()
        label_map = self.registry.get_label_map()
        device = self._model_device(model)

        encoded: list[tuple[str, str, dict[str, torch.Tensor]]] = []
        for text, aspect in pairs:
            try:
                encoded.append((text, aspect, encoder.encode(text, aspect)))
            except ValueError as exc:
                LOGGER.info("skipping unencodable pair (aspect=%r): %s", aspect, exc)

        results: list[PredictResponse] = []
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start : start + batch_size]
            names = [name for name in MODEL_INPUTS if name in chunk[0][2]]
            batch = {
                name: torch.stack([item[2][name] for item in chunk]).to(device)
                for name in names
            }

            with torch.no_grad():
                output = model(**batch)
            probabilities = torch.softmax(output["logits"].float(), dim=-1)

            for row, (_, aspect, _) in enumerate(chunk):
                predicted_id = int(probabilities[row].argmax(dim=-1).item())
                # Sliced back to one example so the single-prediction helpers
                # apply unchanged; they are all written against a batch of 1.
                single = {name: tensor[row : row + 1] for name, tensor in batch.items()}
                attention = None
                if include_attention:
                    one = {
                        "logits": output["logits"][row : row + 1],
                        "attention": (
                            output["attention"][row : row + 1]
                            if isinstance(output, dict) and output.get("attention") is not None
                            else None
                        ),
                    }
                    attention = self._attention(encoder, one, single)

                results.append(
                    PredictResponse(
                        aspect=aspect,
                        sentiment=self._decode(label_map, predicted_id),
                        confidence=round(float(probabilities[row, predicted_id].item()), 4),
                        scores=self._scores(label_map, probabilities[row]),
                        attention=attention,
                        aspect_in_text=self._aspect_in_text(single),
                        model_version=self.registry.model_version,
                    )
                )
        return results

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
    def _aspect_in_text(batch: dict[str, torch.Tensor]) -> bool | None:
        """Whether the aspect occurs verbatim in the review the model saw.

        Every training example took its aspect from a literal span of its
        sentence, so an aspect that is absent puts the model outside the regime
        it learned: aspect attention still produces a distribution, but it is
        aligning the review against a phrase that is not there. A client that
        shows the prediction without saying so is overclaiming.

        Compared in token-id space rather than with a substring test on the raw
        strings, because the ids are exactly what the model consumed. That
        makes the answer agree with the model's view for free -- casing,
        punctuation and possessives are already normalised by the tokenizer
        (``OS`` matches ``os``, ``hard drive`` matches ``hard drive's``) -- and
        it correctly reports False when truncation dropped the only occurrence,
        since the aspect is then genuinely not in the context the model pooled
        over.

        Returns ``None`` when a mask is missing, rather than guessing.
        """
        ids = batch.get("input_ids")
        context_mask = batch.get("context_mask")
        aspect_mask = batch.get("aspect_mask")
        if ids is None or context_mask is None or aspect_mask is None:
            return None

        context = ids[0][context_mask[0].bool()].tolist()
        aspect = ids[0][aspect_mask[0].bool()].tolist()
        if not aspect:
            # The encoder rejects an aspect that tokenises to nothing, so this
            # is unreachable with the real one; say "unknown" over "absent".
            return None
        span = len(aspect)
        return any(
            context[start : start + span] == aspect
            for start in range(len(context) - span + 1)
        )

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
