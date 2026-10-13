"""Aspect-aware sentiment classifier: BERT + additive attention over context.

The sentence-pair encoding alone gives the model the aspect, but nothing tells
it *which* context words matter for that aspect. This architecture makes that
explicit: the aspect representation queries the review's tokens through an
additive (Bahdanau) attention, so "Battery life is amazing but the screen is
dim" can attend to *amazing* for `battery life` and *dim* for `screen` while
the underlying token states stay shared.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import AutoModel

LOGGER = logging.getLogger(__name__)


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``hidden`` over the True positions of ``mask``.

    Args:
        hidden: ``(B, T, d)`` token states.
        mask: ``(B, T)`` boolean selector.

    Returns:
        ``(B, d)``. A row with an all-False mask pools to zeros rather than
        NaN, so one degenerate example cannot poison the batch gradient.
    """
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class AspectAttention(nn.Module):
    """Additive attention scoring context tokens against the aspect vector.

    Implements ``e_i = v^T tanh(W_h h_i + W_a a)``, normalised over the
    context positions only::

        alpha = softmax(e  masked to context_mask)
        c     = sum_i alpha_i h_i

    ``W_a a`` is independent of ``i``, so it is computed once per example and
    broadcast across the sequence.
    """

    def __init__(self, hidden_size: int, attention_size: int | None = None) -> None:
        super().__init__()
        attention_size = attention_size or hidden_size
        self.W_h = nn.Linear(hidden_size, attention_size, bias=True)
        self.W_a = nn.Linear(hidden_size, attention_size, bias=False)
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        aspect: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool ``hidden`` into an aspect-conditioned context vector.

        Args:
            hidden: ``(B, T, d)`` token states.
            aspect: ``(B, d)`` aspect representation.
            context_mask: ``(B, T)`` boolean, True on review tokens only.

        Returns:
            ``(context_vector (B, d), attention_weights (B, T))``. Weights are
            exactly zero outside the mask.
        """
        # (B, T, A) + (B, 1, A) -> additive scores, broadcast over the sequence.
        scores = self.v(
            torch.tanh(self.W_h(hidden) + self.W_a(aspect).unsqueeze(1))
        ).squeeze(-1)

        # finfo.min (not -inf) keeps softmax finite: if a row had no context
        # tokens at all, every logit is equal and the row pools uniformly
        # instead of producing NaN.
        mask = context_mask.bool()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)

        # Re-zero the masked positions: a no-context row would otherwise
        # spread uniform weight over padding.
        weights = weights * mask.to(weights.dtype)

        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return context, weights


class BertAspectAttention(nn.Module):
    """Transformer encoder + aspect attention + MLP classification head.

    The head sees ``[c ; a ; h_CLS]``: the attended context, the aspect itself
    and the pair-level ``[CLS]`` summary. Keeping all three means the model can
    fall back on the sentence-pair signal when attention is uninformative.

    Pass ``encoder=`` to supply an already-built backbone -- used to reload an
    artifact offline, and to run tests against a tiny randomly-initialised
    encoder without touching the network.
    """

    def __init__(
        self,
        model_name: str | None = None,
        num_labels: int = 3,
        dropout: float = 0.1,
        encoder: nn.Module | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if encoder is None and not model_name:
            raise ValueError("pass either model_name or a prebuilt encoder")
        if num_labels < 2:
            raise ValueError(f"num_labels must be >= 2, got {num_labels}")

        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(model_name)
        self.model_name = model_name or getattr(
            self.encoder.config, "name_or_path", "unknown"
        )
        self.num_labels = num_labels

        hidden_size = self.encoder.config.hidden_size
        self.attention = AspectAttention(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        # A non-persistent buffer: the weights follow .to(device) but stay out
        # of the state dict, since they describe the training split's class
        # balance rather than the model itself.
        self.register_buffer("class_weights", None, persistent=False)
        self.set_class_weights(class_weights)

    # ---- loss --------------------------------------------------------- #

    @property
    def loss_fn(self) -> nn.CrossEntropyLoss:
        """The weighted cross-entropy this model trains against.

        A property rather than an assigned attribute on purpose. Assigning
        ``nn.CrossEntropyLoss(weight=w)`` to ``self`` would register it as a
        *submodule*, putting ``loss_fn.weight`` into the ``state_dict`` -- so a
        model trained with class weights could not be reloaded strictly into
        one built without them. Building it on access also means the weight is
        always read from the live buffer, and so always on the right device.
        """
        return nn.CrossEntropyLoss(weight=self.class_weights)

    # ---- forward ------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        aspect_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Classify a batch of (sentence, aspect) pairs.

        Returns:
            ``{"logits": (B, num_labels), "attention": (B, T)}``, plus
            ``"loss"`` when ``labels`` is given.
        """
        encoder_kwargs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # RoBERTa-style backbones take no segment ids.
        if token_type_ids is not None and self.encoder.config.model_type in (
            "bert",
            "electra",
            "albert",
            "megatron-bert",
        ):
            encoder_kwargs["token_type_ids"] = token_type_ids

        hidden = self.encoder(**encoder_kwargs).last_hidden_state  # (B, T, d)

        # Fall back to the attention mask so the model still runs if a caller
        # omits the segment masks.
        if aspect_mask is None:
            aspect_mask = attention_mask
        if context_mask is None:
            context_mask = attention_mask

        aspect = masked_mean(hidden, aspect_mask.bool())            # (B, d)
        context, weights = self.attention(hidden, aspect, context_mask.bool())
        cls_state = hidden[:, 0]                                    # (B, d)

        logits = self.classifier(torch.cat([context, aspect, cls_state], dim=-1))

        output = {"logits": logits, "attention": weights}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels.view(-1))
        return output

    # ---- convenience -------------------------------------------------- #

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predicted class ids for one collated batch."""
        self.eval()
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        return self.forward(**inputs)["logits"].argmax(dim=-1)

    def set_class_weights(self, class_weights: torch.Tensor | None) -> None:
        """Swap the loss weighting after construction, once the training
        split's class balance is known."""
        if class_weights is None:
            self.class_weights = None
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float)
            if weights.numel() != self.num_labels:
                raise ValueError(
                    f"class_weights has {weights.numel()} entries "
                    f"for {self.num_labels} labels"
                )
            device = next(self.parameters(), torch.empty(0)).device
            self.class_weights = weights.to(device, dtype=torch.float)

    def parameter_groups(
        self, lr: float, weight_decay: float, head_lr: float | None = None
    ) -> list[dict]:
        """AdamW groups: biases/LayerNorm exempt from decay, head on its own LR.

        The head is randomly initialised while the encoder is pretrained, so a
        single learning rate suits neither.

        The backbone's pooler is skipped: this model pools ``last_hidden_state``
        itself, so no gradient ever reaches those weights. They stay in the
        state dict (keeping a strict reload symmetric) but giving them to AdamW
        would only allocate optimizer moments that never move.
        """
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        head = nn.ModuleList([self.attention, self.classifier])
        groups: list[dict] = []
        for prefix, module, module_lr in (
            ("encoder", self.encoder, lr),
            ("head", head, head_lr if head_lr is not None else lr),
        ):
            decay, plain = [], []
            for name, param in module.named_parameters():
                if param.requires_grad and not name.startswith("pooler."):
                    (plain if any(n in name for n in no_decay) else decay).append(param)
            if decay:
                groups.append({"params": decay, "lr": module_lr,
                               "weight_decay": weight_decay, "name": f"{prefix}_decay"})
            if plain:
                groups.append({"params": plain, "lr": module_lr,
                               "weight_decay": 0.0, "name": f"{prefix}_no_decay"})
        return groups

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"BertAspectAttention(model={self.model_name!r}, "
            f"hidden={self.encoder.config.hidden_size}, labels={self.num_labels})"
        )
