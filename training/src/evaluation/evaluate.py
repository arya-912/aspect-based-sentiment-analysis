"""Metric computation for aspect-based sentiment predictions."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LOGGER = logging.getLogger(__name__)


def _label_names(label_map, num_classes: int) -> list[str]:
    """Label names from a LabelMap, falling back to stringified class ids."""
    if label_map is not None:
        return list(label_map.labels)
    return [str(i) for i in range(num_classes)]


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
    loss: float | None = None,
    split: str = "unknown",
) -> dict[str, Any]:
    """Score predictions against gold labels.

    Args:
        y_true: gold class ids.
        y_pred: predicted class ids.
        labels: label names, indexed by class id.
        loss: mean loss to include in the report, when known.
        split: name recorded as ``evaluation_split``.

    Returns:
        ``accuracy``, ``macro_f1``, ``weighted_f1``, ``per_class``,
        ``confusion_matrix``, ``n``, ``evaluation_split`` and, when given,
        ``loss``. Every class appears in ``per_class`` even with zero support,
        so the shape of the report does not depend on what the model predicted.
    """
    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true/y_pred length mismatch: {len(y_true)} vs {len(y_pred)}"
        )
    if not y_true:
        raise ValueError("cannot compute metrics on zero examples")

    class_ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=class_ids, average="weighted", zero_division=0)
        ),
        "per_class": {
            labels[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in class_ids
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
        "n": len(y_true),
        "evaluation_split": split,
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run inference over a dataloader.

    Returns ``{"probabilities", "y_pred", "y_true", "loss", "attention"}``.
    ``y_true`` and ``loss`` are ``None`` for unlabelled batches.
    """
    model.eval()
    model.to(device)

    probs_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []
    total_loss, n_batches = 0.0, 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels", None)
        output = model(**batch, labels=labels)

        probs_chunks.append(torch.softmax(output["logits"].float(), dim=-1).cpu().numpy())
        if labels is not None:
            true_chunks.append(labels.cpu().numpy())
            total_loss += float(output["loss"])
            n_batches += 1

    if not probs_chunks:
        raise ValueError("dataloader yielded no batches")

    probabilities = np.concatenate(probs_chunks, axis=0)
    return {
        "probabilities": probabilities,
        "y_pred": probabilities.argmax(axis=1),
        "y_true": np.concatenate(true_chunks, axis=0) if true_chunks else None,
        "loss": (total_loss / n_batches) if n_batches else None,
    }


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | str = "cpu",
    label_map=None,
    split: str = "test",
) -> dict[str, Any]:
    """Score a labelled dataloader end to end.

    Args:
        model: a :class:`BertAspectAttention` (or anything with the same
            forward contract).
        dataloader: yields collated batches including ``labels``.
        device: where to run.
        label_map: optional :class:`LabelMap`, used to name the classes.
        split: recorded as ``evaluation_split``.

    Returns:
        The dict described by :func:`compute_metrics`.
    """
    outputs = predict(model, dataloader, device)
    if outputs["y_true"] is None:
        raise ValueError(
            "evaluate() needs labelled batches; use predict() for unlabelled data"
        )

    labels = _label_names(label_map, outputs["probabilities"].shape[1])
    metrics = compute_metrics(
        outputs["y_true"],
        outputs["y_pred"],
        labels,
        loss=outputs["loss"],
        split=split,
    )
    LOGGER.info(
        "%s: n=%d acc=%.4f macro_f1=%.4f",
        split,
        metrics["n"],
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    return metrics


def per_class_f1(metrics: dict[str, Any]) -> dict[str, float]:
    """Just the F1 per label, as the trainer records each epoch."""
    return {name: scores["f1"] for name, scores in metrics["per_class"].items()}


def sklearn_text_report(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]
) -> str:
    """scikit-learn's own classification report, for the console."""
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=list(labels),
        zero_division=0,
        digits=4,
    )
