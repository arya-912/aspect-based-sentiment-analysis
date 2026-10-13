"""End-to-end orchestration: load data, fine-tune, evaluate, save artifact."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .artifact import ArtifactMetadata, save_artifact
from .config import ExperimentConfig
from .data.dataset import (
    ABSADataset,
    LabelMap,
    label_distribution,
    load_official_test,
    load_raw_csv,
    split_train_val,
)
from .evaluation import reporting
from .evaluation.evaluate import sklearn_text_report
from .models.absa_model import ABSAModel
from .preprocessing.tokenizer import ABSAEncoder
from .trainer import ABSATrainer, set_seed

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Single-line log format, safe to call more than once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("transformers").setLevel(logging.ERROR)


class Experiment:
    """One reproducible ABSA run, driven entirely by an ``ExperimentConfig``."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.label_map = LabelMap(labels=tuple(config.data.labels))
        self.run_dir = config.run_dir
        self.encoder: ABSAEncoder | None = None
        self.model: ABSAModel | None = None
        self.trainer: ABSATrainer | None = None

    # ---- stages ------------------------------------------------------- #

    def prepare_data(self) -> dict[str, Any]:
        """Load, clean, split and optionally persist the splits."""
        data_cfg = self.config.data
        train_path = self.config.resolve(data_cfg.train_csv)

        full = load_raw_csv(train_path, self.label_map.labels)
        train_df, val_df = split_train_val(
            full, val_fraction=data_cfg.val_fraction, seed=data_cfg.seed
        )

        test_df = None
        test_path = self.config.resolve(data_cfg.test_csv)
        if test_path.is_file():
            test_df = load_official_test(test_path, self.label_map.labels)
        else:
            LOGGER.warning("no gold test file at %s; skipping test evaluation", test_path)

        if data_cfg.save_splits:
            splits_dir = self.config.resolve(data_cfg.splits_dir)
            splits_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(splits_dir / "train.csv", index=False)
            val_df.to_csv(splits_dir / "val.csv", index=False)
            if test_df is not None:
                test_df.to_csv(splits_dir / "test.csv", index=False)
            LOGGER.info("wrote splits to %s", splits_dir)

        for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            if frame is not None:
                LOGGER.info("%-5s %4d rows  %s", name, len(frame), label_distribution(frame))

        self.frames = {"train": train_df, "val": val_df, "test": test_df}
        return self.frames

    def build(self) -> ABSAModel:
        """Instantiate the encoder and the model from config."""
        model_cfg = self.config.model
        self.encoder = ABSAEncoder(model_cfg.model_name, max_length=model_cfg.max_length)
        self.model = ABSAModel(
            model_name=model_cfg.model_name,
            num_labels=self.label_map.num_labels,
            dropout=model_cfg.dropout,
            pooling=model_cfg.pooling,
            hidden_dim=model_cfg.hidden_dim,
            freeze_encoder=model_cfg.freeze_encoder,
        )
        return self.model

    def run(self) -> dict[str, Any]:
        """Execute the whole pipeline and return a summary dict."""
        configure_logging()
        set_seed(self.config.train.seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("run '%s' -> %s", self.config.name, self.run_dir)

        frames = self.prepare_data()
        self.build()
        assert self.encoder is not None and self.model is not None

        datasets = {
            name: ABSADataset(frame, self.encoder, self.label_map)
            for name, frame in frames.items()
            if frame is not None
        }

        self.trainer = ABSATrainer(
            model=self.model,
            config=self.config.train,
            labels=self.label_map.labels,
            class_weights=datasets["train"].class_weights(),
            output_dir=self.run_dir,
        )
        outcome = self.trainer.train(datasets["train"], datasets["val"])

        print()
        print(reporting.format_history(outcome.history))
        print()

        reports: dict[str, Any] = {}
        if outcome.best_result is not None:
            reporting.save_report(
                outcome.best_result,
                self.run_dir,
                stem="validation",
                title=f"Validation (best epoch {outcome.best_epoch})",
            )
            reporting.save_predictions_csv(
                outcome.best_result, frames["val"], self.run_dir / "validation_predictions.csv"
            )
            reports["validation"] = outcome.best_result.summary()
            print(reporting.format_result(
                outcome.best_result, f"Validation (best epoch {outcome.best_epoch})"
            ))
            print()

        if "test" in datasets:
            test_result = self.trainer.evaluate(datasets["test"])
            reporting.save_report(
                test_result, self.run_dir, stem="test", title="Official gold test"
            )
            reporting.save_predictions_csv(
                test_result, frames["test"], self.run_dir / "test_predictions.csv"
            )
            reports["test"] = test_result.summary()
            print(reporting.format_result(test_result, "Official gold test"))
            print()
            print(sklearn_text_report(
                test_result.y_true, test_result.y_pred, self.label_map.labels
            ))
            print()

        metadata = ArtifactMetadata(
            name=self.config.name,
            model_name=self.config.model.model_name,
            labels=self.label_map.labels,
            max_length=self.config.model.max_length,
            pooling=self.config.model.pooling,
            best_epoch=outcome.best_epoch,
            monitor=outcome.monitor,
            best_metric=outcome.best_metric,
            metrics=reports,
            history=outcome.history,
            dataset={
                name: {"rows": len(frame), "labels": label_distribution(frame)}
                for name, frame in frames.items()
                if frame is not None
            },
        )
        save_artifact(
            self.run_dir, self.model, self.encoder, self.config, metadata,
            state_dict=outcome.best_state_dict,
        )

        summary = {
            "run_dir": str(self.run_dir),
            "best_epoch": outcome.best_epoch,
            "monitor": outcome.monitor,
            "best_metric": outcome.best_metric,
            "seconds": round(outcome.seconds, 1),
            "metrics": reports,
        }
        reporting.save_json(summary, self.run_dir / "summary.json")
        return summary


def run_experiment(config_path: str | Path | None = None, **overrides: Any) -> dict[str, Any]:
    """Convenience entry point for notebooks and scripts."""
    config = (
        ExperimentConfig.load(config_path) if config_path else ExperimentConfig()
    )
    if overrides:
        config = config.with_overrides(overrides)
    return Experiment(config).run()
