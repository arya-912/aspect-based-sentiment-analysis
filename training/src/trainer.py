"""Training loop: per-epoch fit/eval, gradient clipping, best-checkpoint tracking."""

from __future__ import annotations

import copy
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .evaluation.evaluate import compute_metrics, per_class_f1

LOGGER = logging.getLogger(__name__)

CHECKPOINT_FILE = "best.pt"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch (including CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """``cuda`` when available, else ``cpu``; an explicit request wins."""
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Fine-tunes a :class:`BertAspectAttention` and keeps the best weights.

    "Best" is the highest **validation macro F1**, not the last epoch and not
    accuracy: `neutral` is the minority class here, so accuracy can climb while
    the model quietly gives up on it.

    The optimizer and scheduler are supplied by the caller, which keeps the
    schedule (warmup length, parameter groups) visible at the call site instead
    of buried in here.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str | torch.device = "cpu",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = resolve_device(device)
        self.config = config or {}

        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.label_map = self.config.get("label_map")
        self.checkpoint_dir = (
            Path(self.config["checkpoint_dir"]) if self.config.get("checkpoint_dir") else None
        )

        self.history: list[dict[str, Any]] = []
        self.best_macro_f1 = -float("inf")
        self.best_epoch = -1
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_val_metrics: dict[str, Any] | None = None

        self.model.to(self.device)
        LOGGER.info(
            "device=%s  trainable params=%s  max_grad_norm=%s",
            self.device,
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
            self.max_grad_norm,
        )

    # ---- one epoch ---------------------------------------------------- #

    def train_epoch(self) -> dict[str, Any]:
        """One pass over the training data.

        Returns:
            ``{"loss", "accuracy", "macro_f1"}`` over the epoch's own
            predictions, which are collected as training proceeds -- so they
            reflect a model that was still changing, and will read lower than a
            clean pass over the same data.
        """
        self.model.train()
        total_loss, n_batches = 0.0, 0
        y_true: list[int] = []
        y_pred: list[int] = []

        progress = tqdm(self.train_loader, desc="train", leave=False)
        for batch in progress:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("labels")

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(**batch, labels=labels)
            loss = output["loss"]
            loss.backward()

            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += float(loss.detach())
            n_batches += 1
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(output["logits"].detach().argmax(dim=-1).cpu().tolist())
            progress.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        labels_names = self._label_names()
        metrics = compute_metrics(
            y_true, y_pred, labels_names, loss=total_loss / max(1, n_batches), split="train"
        )
        return {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }

    @torch.no_grad()
    def eval_epoch(self, loader=None) -> dict[str, Any]:
        """Evaluate without updating weights.

        Returns:
            ``{"loss", "accuracy", "macro_f1", "per_class_f1"}``.
        """
        loader = loader if loader is not None else self.val_loader
        self.model.eval()

        total_loss, n_batches = 0.0, 0
        y_true: list[int] = []
        y_pred: list[int] = []

        for batch in tqdm(loader, desc="eval", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("labels", None)
            output = self.model(**batch, labels=labels)
            y_pred.extend(output["logits"].argmax(dim=-1).cpu().tolist())
            if labels is not None:
                y_true.extend(labels.cpu().tolist())
                total_loss += float(output["loss"])
                n_batches += 1

        if not y_true:
            raise ValueError("eval_epoch() needs labelled batches")

        metrics = compute_metrics(
            y_true,
            y_pred,
            self._label_names(),
            loss=total_loss / max(1, n_batches),
            split="val",
        )
        return {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_f1": per_class_f1(metrics),
            "metrics": metrics,
        }

    # ---- the loop ----------------------------------------------------- #

    def fit(self, num_epochs: int) -> list[dict[str, Any]]:
        """Train for ``num_epochs``, keeping the best-val-macro-F1 weights.

        Returns:
            One dict per epoch. The model is left holding the best weights,
            not the last epoch's.
        """
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

        started = time.time()
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            # Captured before the epoch's steps: a completed linear schedule
            # ends at exactly 0, which says nothing about how the epoch ran.
            epoch_lr = self.optimizer.param_groups[0]["lr"]
            train_metrics = self.train_epoch()
            val_metrics = self.eval_epoch()

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_per_class_f1": val_metrics["per_class_f1"],
                "lr": epoch_lr,
                "seconds": round(time.time() - epoch_start, 1),
            }
            self.history.append(record)

            LOGGER.info(
                "epoch %d/%d  train_loss=%.4f acc=%.4f f1=%.4f | "
                "val_loss=%.4f acc=%.4f f1=%.4f",
                epoch, num_epochs,
                record["train_loss"], record["train_accuracy"], record["train_macro_f1"],
                record["val_loss"], record["val_accuracy"], record["val_macro_f1"],
            )

            if record["val_macro_f1"] > self.best_macro_f1:
                self.best_macro_f1 = record["val_macro_f1"]
                self.best_epoch = epoch
                self.best_val_metrics = val_metrics["metrics"]
                self.best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                )
                self._save_checkpoint(epoch)
                LOGGER.info(
                    "new best val macro_f1=%.4f (epoch %d)", self.best_macro_f1, epoch
                )

        if self.best_state is None:
            raise RuntimeError("training finished without a best checkpoint")

        self.model.load_state_dict(self.best_state)
        LOGGER.info(
            "restored best checkpoint: epoch %d, val macro_f1=%.4f (%.1fs total)",
            self.best_epoch, self.best_macro_f1, time.time() - started,
        )
        return self.history

    # ---- helpers ------------------------------------------------------ #

    def _label_names(self) -> list[str]:
        if self.label_map is not None:
            return list(self.label_map.labels)
        return [str(i) for i in range(getattr(self.model, "num_labels", 3))]

    def _save_checkpoint(self, epoch: int) -> Path | None:
        """Write the current best weights, if a checkpoint dir was configured."""
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / CHECKPOINT_FILE
        torch.save(
            {
                "state_dict": self.best_state,
                "epoch": epoch,
                "val_macro_f1": self.best_macro_f1,
            },
            path,
        )
        return path
