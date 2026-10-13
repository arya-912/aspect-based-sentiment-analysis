"""Aspect-aware sentiment classifier on top of a pretrained transformer."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

LOGGER = logging.getLogger(__name__)

POOLINGS = ("cls", "aspect", "aspect_context")


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``hidden`` over the True positions of ``mask``.

    Args:
        hidden: ``(batch, seq, dim)`` token states.
        mask: ``(batch, seq)`` boolean selector.

    Returns:
        ``(batch, dim)``. Rows with an all-False mask pool to zeros rather
        than NaN, so a degenerate example cannot poison the batch gradient.
    """
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class ABSAModel(nn.Module):
    """Transformer encoder + pooled aspect/context representation + classifier.

    ``pooling`` selects what the head sees:

    ``cls``
        the ``[CLS]`` state alone, the plain sentence-pair baseline.
    ``aspect``
        mean over the aspect segment's tokens.
    ``aspect_context``
        ``[CLS] ‖ mean(aspect) ‖ mean(context)`` concatenated, which keeps the
        pair summary while giving the head direct access to both segments.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        pooling: str = "aspect_context",
        hidden_dim: int | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        if num_labels < 2:
            raise ValueError(f"num_labels must be >= 2, got {num_labels}")

        self.model_name = model_name
        self.num_labels = num_labels
        self.pooling = pooling

        self.encoder_config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.encoder_config)
        hidden_size = self.encoder_config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            LOGGER.info("encoder frozen; training the head only")

        pooled_dim = hidden_size * (3 if pooling == "aspect_context" else 1)
        self.dropout = nn.Dropout(dropout)
        if hidden_dim:
            self.classifier = nn.Sequential(
                nn.Linear(pooled_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            self.classifier = nn.Linear(pooled_dim, num_labels)

    # ---- forward ------------------------------------------------------ #

    def pool(
        self,
        hidden: torch.Tensor,
        context_mask: torch.Tensor,
        aspect_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce token states to one vector per example."""
        cls_state = hidden[:, 0]
        if self.pooling == "cls":
            return cls_state
        if self.pooling == "aspect":
            return masked_mean(hidden, aspect_mask)
        return torch.cat(
            [
                cls_state,
                masked_mean(hidden, aspect_mask),
                masked_mean(hidden, context_mask),
            ],
            dim=-1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        aspect_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_fn: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the encoder and head.

        Returns a dict with ``logits`` and, when ``labels`` is given, ``loss``.
        """
        kwargs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.encoder_config.model_type in (
            "bert",
            "electra",
            "albert",
        ):
            kwargs["token_type_ids"] = token_type_ids

        hidden = self.encoder(**kwargs).last_hidden_state

        if context_mask is None:
            context_mask = attention_mask.bool()
        if aspect_mask is None:
            aspect_mask = attention_mask.bool()

        pooled = self.pool(hidden, context_mask.bool(), aspect_mask.bool())
        logits = self.classifier(self.dropout(pooled))

        output = {"logits": logits}
        if labels is not None:
            criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
            output["loss"] = criterion(logits, labels)
        return output

    # ---- convenience -------------------------------------------------- #

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predicted class ids for one already-collated batch."""
        self.eval()
        logits = self.forward(**{k: v for k, v in batch.items() if k != "labels"})[
            "logits"
        ]
        return logits.argmax(dim=-1)

    def parameter_groups(
        self,
        learning_rate: float,
        head_learning_rate: float,
        weight_decay: float,
    ) -> list[dict]:
        """Split params into encoder/head x decay/no-decay for AdamW.

        Biases and LayerNorm weights are exempt from weight decay, and the
        randomly-initialised head gets the larger learning rate.
        """
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups: list[dict] = []
        for prefix, module, lr in (
            ("encoder", self.encoder, learning_rate),
            ("head", self.classifier, head_learning_rate),
        ):
            decay_params, plain_params = [], []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                (plain_params if any(n in name for n in no_decay) else decay_params
                 ).append(param)
            if decay_params:
                groups.append(
                    {
                        "params": decay_params,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "name": f"{prefix}_decay",
                    }
                )
            if plain_params:
                groups.append(
                    {
                        "params": plain_params,
                        "lr": lr,
                        "weight_decay": 0.0,
                        "name": f"{prefix}_no_decay",
                    }
                )
        return groups

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
