"""Metric computation and inference loops."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Metrics plus the raw predictions they were computed from."""

    loss: float | None
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    labels: tuple[str, ...]
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)
    probabilities: list[list[float]] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """The scalars worth logging each epoch."""
        out = {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
        }
        if self.loss is not None:
            out["loss"] = self.loss
        return out

    def to_dict(self, include_predictions: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "labels": list(self.labels),
        }
        if include_predictions:
            payload["y_true"] = self.y_true
            payload["y_pred"] = self.y_pred
            payload["probabilities"] = self.probabilities
        return payload


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
    loss: float | None = None,
    probabilities: Sequence[Sequence[float]] | None = None,
) -> EvalResult:
    """Accuracy, macro/weighted F1, per-class P/R/F1 and a confusion matrix."""
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
    per_class = {
        labels[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in class_ids
    }

    return EvalResult(
        loss=None if loss is None else float(loss),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=class_ids, average="weighted", zero_division=0)
        ),
        per_class=per_class,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
        labels=tuple(labels),
        y_true=y_true,
        y_pred=y_pred,
        probabilities=[[float(p) for p in row] for row in (probabilities or [])],
    )


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


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
    loss_fn: torch.nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """Run inference over a loader.

    Returns ``(probabilities, y_true, mean_loss)``. The last two are ``None``
    when the loader yields unlabelled batches.
    """
    model.eval()
    model.to(device)

    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    total_loss, n_batches = 0.0, 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels", None)
        output = model(**batch, labels=labels, loss_fn=loss_fn)

        all_probs.append(torch.softmax(output["logits"].float(), dim=-1).cpu().numpy())
        if labels is not None:
            all_true.append(labels.cpu().numpy())
            total_loss += float(output["loss"])
            n_batches += 1

    if not all_probs:
        raise ValueError("dataloader yielded no batches")

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0) if all_true else None
    mean_loss = (total_loss / n_batches) if n_batches else None
    return probs, y_true, mean_loss


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    labels: Sequence[str],
    device: torch.device | str = "cpu",
    loss_fn: torch.nn.Module | None = None,
) -> EvalResult:
    """Score a labelled dataloader end to end."""
    probs, y_true, mean_loss = predict(model, dataloader, device, loss_fn)
    if y_true is None:
        raise ValueError("evaluate() needs labelled batches; use predict() instead")
    y_pred = probs.argmax(axis=1)
    return compute_metrics(y_true, y_pred, labels, mean_loss, probs.tolist())


def error_analysis(result: EvalResult, texts: Sequence[str], aspects: Sequence[str], limit: int = 25) -> list[dict[str, Any]]:
    """The most confident mistakes, highest-confidence first."""
    rows: list[dict[str, Any]] = []
    for idx, (gold, pred) in enumerate(zip(result.y_true, result.y_pred)):
        if gold == pred:
            continue
        confidence = (
            max(result.probabilities[idx]) if idx < len(result.probabilities) else float("nan")
        )
        rows.append(
            {
                "index": idx,
                "text": texts[idx] if idx < len(texts) else None,
                "aspect": aspects[idx] if idx < len(aspects) else None,
                "gold": result.labels[gold],
                "predicted": result.labels[pred],
                "confidence": confidence,
            }
        )
    rows.sort(key=lambda r: (-r["confidence"] if r["confidence"] == r["confidence"] else 0))
    return rows[:limit]
