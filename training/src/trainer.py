"""Training loop with warmup, gradient clipping and best-checkpoint tracking."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import TrainConfig
from .data.dataset import collate_batch
from .evaluation.evaluate import EvalResult, evaluate

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch (including CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick a device, honouring an explicit request when it is available."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainingOutcome:
    """What a finished run leaves behind."""

    best_epoch: int
    best_metric: float
    monitor: str
    history: list[dict[str, Any]] = field(default_factory=list)
    best_result: EvalResult | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    seconds: float = 0.0


class ABSATrainer:
    """Fine-tunes an :class:`ABSAModel` and keeps the best-scoring weights.

    The best checkpoint is chosen by ``config.monitor`` on the validation
    split, so the returned weights are not simply the last epoch's.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        labels: tuple[str, ...],
        class_weights: torch.Tensor | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.labels = labels
        self.device = resolve_device(config.device)
        self.output_dir = Path(output_dir) if output_dir else None

        weights = None
        if config.class_weighting and class_weights is not None:
            weights = class_weights.to(self.device)
            LOGGER.info(
                "class weights: %s",
                {l: round(float(w), 3) for l, w in zip(labels, weights)},
            )
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weights, label_smoothing=config.label_smoothing
        )

        self.model.to(self.device)
        LOGGER.info(
            "device=%s, trainable params=%s",
            self.device,
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        )

    # ---- loaders ------------------------------------------------------ #

    def make_loader(self, dataset, shuffle: bool, batch_size: int | None = None) -> DataLoader:
        """Build a DataLoader with the project's collate function."""
        return DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_batch,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    # ---- optimisation ------------------------------------------------- #

    def build_optimizer(self) -> torch.optim.Optimizer:
        groups = (
            self.model.parameter_groups(
                self.config.learning_rate,
                self.config.head_learning_rate,
                self.config.weight_decay,
            )
            if hasattr(self.model, "parameter_groups")
            else [{"params": list(self.model.parameters())}]
        )
        return torch.optim.AdamW(groups, lr=self.config.learning_rate)

    def _is_better(self, candidate: float, incumbent: float) -> bool:
        """Lower is better for losses, higher for every other metric."""
        if "loss" in self.config.monitor:
            return candidate < incumbent
        return candidate > incumbent

    # ---- the loop ----------------------------------------------------- #

    def train(self, train_dataset, val_dataset) -> TrainingOutcome:
        """Fine-tune for ``config.epochs``, evaluating after each one."""
        set_seed(self.config.seed)
        train_loader = self.make_loader(train_dataset, shuffle=True)
        val_loader = self.make_loader(
            val_dataset, shuffle=False, batch_size=self.config.eval_batch_size
        )

        total_steps = max(1, len(train_loader) * self.config.epochs)
        optimizer = self.build_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps,
        )

        worst = float("inf") if "loss" in self.config.monitor else -float("inf")
        outcome = TrainingOutcome(
            best_epoch=-1, best_metric=worst, monitor=self.config.monitor
        )
        epochs_without_gain = 0
        started = time.time()

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch(
                train_loader, optimizer, scheduler, epoch
            )
            result = evaluate(
                self.model, val_loader, self.labels, self.device, self.loss_fn
            )

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": result.loss,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "weighted_f1": result.weighted_f1,
                "lr": optimizer.param_groups[0]["lr"],
            }
            outcome.history.append(record)
            LOGGER.info(
                "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  acc=%.4f  macro_f1=%.4f",
                epoch,
                self.config.epochs,
                train_loss,
                result.loss or float("nan"),
                result.accuracy,
                result.macro_f1,
            )

            metric = record.get(self.config.monitor)
            if metric is None:
                raise ValueError(
                    f"monitor={self.config.monitor!r} is not a recorded metric; "
                    f"available: {sorted(k for k in record if k != 'epoch')}"
                )

            if self._is_better(metric, outcome.best_metric):
                outcome.best_metric = float(metric)
                outcome.best_epoch = epoch
                outcome.best_result = result
                outcome.best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                epochs_without_gain = 0
                LOGGER.info(
                    "new best %s=%.4f at epoch %d", self.config.monitor, metric, epoch
                )
            else:
                epochs_without_gain += 1
                if (
                    self.config.early_stopping_patience
                    and epochs_without_gain >= self.config.early_stopping_patience
                ):
                    LOGGER.info(
                        "early stopping: no %s gain for %d epoch(s)",
                        self.config.monitor,
                        epochs_without_gain,
                    )
                    break

        outcome.seconds = time.time() - started
        if outcome.best_state_dict is None:
            raise RuntimeError("training finished without a best checkpoint")

        self.model.load_state_dict(outcome.best_state_dict)
        LOGGER.info(
            "restored best checkpoint (epoch %d, %s=%.4f) after %.1fs",
            outcome.best_epoch,
            outcome.monitor,
            outcome.best_metric,
            outcome.seconds,
        )
        return outcome

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
    ) -> float:
        """One pass over the training data; returns the mean batch loss."""
        self.model.train()
        running, n_batches = 0.0, 0
        progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)

        for batch in progress:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad(set_to_none=True)
            loss = self.model(**batch, labels=labels, loss_fn=self.loss_fn)["loss"]
            loss.backward()
            if self.config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            optimizer.step()
            scheduler.step()

            running += float(loss)
            n_batches += 1
            progress.set_postfix(loss=f"{running / n_batches:.4f}")

        return running / max(1, n_batches)

    def evaluate(self, dataset) -> EvalResult:
        """Score any dataset with the trainer's device and loss function."""
        loader = self.make_loader(
            dataset, shuffle=False, batch_size=self.config.eval_batch_size
        )
        return evaluate(self.model, loader, self.labels, self.device, self.loss_fn)
