#!/usr/bin/env python3
"""CLI entry point for an ABSA training run.

Examples::

    python train.py                                   # defaults
    python train.py --config configs/bert_base.yaml
    python train.py --epochs 2 --batch-size 16 --name quick-test
    python train.py --model-name roberta-base --pooling cls
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ExperimentConfig  # noqa: E402
from src.experiment import Experiment, configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune an aspect-based sentiment classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="YAML config to load as the base")
    parser.add_argument("--name", help="run name; also the artifact sub-directory")

    data = parser.add_argument_group("data")
    data.add_argument("--train-csv", help="training CSV, relative to training/")
    data.add_argument("--test-csv", help="gold test CSV, relative to training/")
    data.add_argument("--val-fraction", type=float, help="validation row fraction")
    data.add_argument("--seed", type=int, help="seed for the split and training")

    model = parser.add_argument_group("model")
    model.add_argument("--model-name", help="HuggingFace backbone id")
    model.add_argument("--max-length", type=int, help="max tokens per example")
    model.add_argument("--dropout", type=float)
    model.add_argument("--pooling", choices=("cls", "aspect", "aspect_context"))
    model.add_argument("--hidden-dim", type=int, help="MLP head size; omit for linear")
    model.add_argument("--freeze-encoder", action="store_true", help="train head only")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--eval-batch-size", type=int)
    train.add_argument("--learning-rate", type=float, help="encoder LR")
    train.add_argument("--head-learning-rate", type=float)
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--warmup-ratio", type=float)
    train.add_argument("--label-smoothing", type=float)
    train.add_argument("--no-class-weighting", action="store_true")
    train.add_argument("--monitor", choices=("macro_f1", "accuracy", "weighted_f1", "val_loss"))
    train.add_argument("--early-stopping-patience", type=int)
    train.add_argument("--device", help="auto | cpu | cuda | mps")
    train.add_argument("--num-workers", type=int)

    parser.add_argument(
        "--dump-config",
        type=Path,
        help="write the resolved config to this path and exit without training",
    )
    parser.add_argument("--verbose", action="store_true", help="debug-level logging")
    return parser


#: CLI flag -> dotted config key. Flags left unset are not applied.
OVERRIDE_MAP = {
    "name": "name",
    "train_csv": "data.train_csv",
    "test_csv": "data.test_csv",
    "val_fraction": "data.val_fraction",
    "model_name": "model.model_name",
    "max_length": "model.max_length",
    "dropout": "model.dropout",
    "pooling": "model.pooling",
    "hidden_dim": "model.hidden_dim",
    "epochs": "train.epochs",
    "batch_size": "train.batch_size",
    "eval_batch_size": "train.eval_batch_size",
    "learning_rate": "train.learning_rate",
    "head_learning_rate": "train.head_learning_rate",
    "weight_decay": "train.weight_decay",
    "warmup_ratio": "train.warmup_ratio",
    "label_smoothing": "train.label_smoothing",
    "monitor": "train.monitor",
    "early_stopping_patience": "train.early_stopping_patience",
    "device": "train.device",
    "num_workers": "train.num_workers",
}


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load the base config and layer the explicitly-passed CLI flags on top."""
    config = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()

    overrides = {
        key: getattr(args, flag)
        for flag, key in OVERRIDE_MAP.items()
        if getattr(args, flag, None) is not None
    }
    if args.seed is not None:
        overrides["data.seed"] = args.seed
        overrides["train.seed"] = args.seed
    if args.freeze_encoder:
        overrides["model.freeze_encoder"] = True
    if args.no_class_weighting:
        overrides["train.class_weighting"] = False

    return config.with_overrides(overrides) if overrides else config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    config = resolve_config(args)

    if args.dump_config:
        path = config.save(args.dump_config)
        print(f"wrote {path}")
        return 0

    summary = Experiment(config).run()
    print("=== summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
