#!/usr/bin/env python3
"""ABSA training entry point.

Three subcommands, each usable on its own::

    python train.py prepare               # split the corpus -> train.csv / val.csv
    python train.py train                 # fine-tune, then write the artifact
    python train.py evaluate              # score the official gold test set

Config comes from ``training/configs/default.yaml``; ``--config`` points at
another file and the per-command flags override individual values.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.artifact import build_artifact, load_artifact, predict_pairs
from src.config import (
    artifact_dir,
    describe,
    for_artifact,
    get,
    load_config,
    resolve_path,
)
from src.data.dataset import (
    ABSADataset,
    LabelMap,
    collate_batch,
    label_distribution,
    load_official_test,
    load_raw_csv,
    split_train_val,
)
from src.evaluation import reporting
from src.evaluation.evaluate import (
    compute_metrics,
    evaluate,
    predict,
    sklearn_text_report,
)
from src.models.absa_model import BertAspectAttention
from src.preprocessing.tokenizer import ABSAEncoder
from src.trainer import Trainer, resolve_device, set_seed

LOGGER = logging.getLogger("absa.train")

TRAIN_SPLIT = "train.csv"
VAL_SPLIT = "val.csv"


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("transformers").setLevel(logging.ERROR)


def pick_device(requested: str | None = None) -> torch.device:
    """``cuda`` when available, else ``cpu``."""
    return resolve_device(requested)


def build_label_map(cfg: dict[str, Any]) -> LabelMap:
    labels = get(cfg, "data.labels") or ("negative", "neutral", "positive")
    return LabelMap(labels=tuple(labels))


def make_loader(
    dataset: ABSADataset, batch_size: int, shuffle: bool, cfg: dict[str, Any]
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(get(cfg, "optim.num_workers", 0) or 0),
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )


# --------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------- #


def cmd_prepare(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Clean the raw corpus and write the train/val split."""
    label_map = build_label_map(cfg)
    raw_path = resolve_path(cfg, "paths.raw_train")
    out_dir = resolve_path(cfg, "paths.processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    full = load_raw_csv(raw_path, label_map.labels)
    train_df, val_df = split_train_val(
        full,
        val_fraction=float(get(cfg, "training.val_fraction")),
        seed=int(get(cfg, "seed")),
    )

    train_df.to_csv(out_dir / TRAIN_SPLIT, index=False)
    val_df.to_csv(out_dir / VAL_SPLIT, index=False)

    print(f"raw        : {raw_path}")
    print(f"usable     : {len(full)} rows, {full['text'].nunique()} sentences")
    for name, frame in (("train", train_df), ("val", val_df)):
        print(f"{name:<10} : {len(frame):>5} rows  {label_distribution(frame)}")
    print(f"wrote      : {out_dir / TRAIN_SPLIT}")
    print(f"             {out_dir / VAL_SPLIT}")

    overlap = set(train_df["text"]) & set(val_df["text"])
    assert not overlap, f"{len(overlap)} sentence(s) leaked across the split"
    print("leakage    : none (split is grouped by sentence)")
    return 0


def load_splits(cfg: dict[str, Any], label_map: LabelMap, force_prepare: bool = False):
    """Read the prepared splits, creating them if they are absent."""
    out_dir = resolve_path(cfg, "paths.processed_dir")
    train_path, val_path = out_dir / TRAIN_SPLIT, out_dir / VAL_SPLIT

    if force_prepare or not (train_path.is_file() and val_path.is_file()):
        LOGGER.info("splits missing or stale; preparing them now")
        cmd_prepare(cfg, argparse.Namespace())

    import pandas as pd

    return (
        pd.read_csv(train_path, dtype={"review_id": str}),
        pd.read_csv(val_path, dtype={"review_id": str}),
    )


# --------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------- #


def cmd_train(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Fine-tune the model and write the artifact."""
    set_seed(int(get(cfg, "seed")))
    device = pick_device(args.device)
    label_map = build_label_map(cfg)

    LOGGER.info("%s  device=%s", describe(cfg), device)

    train_df, val_df = load_splits(cfg, label_map, force_prepare=args.prepare)
    LOGGER.info("train %d rows %s", len(train_df), label_distribution(train_df))
    LOGGER.info("val   %d rows %s", len(val_df), label_distribution(val_df))

    encoder = ABSAEncoder(
        get(cfg, "model.name"), max_length=int(get(cfg, "model.max_length"))
    )
    train_dataset = ABSADataset(train_df, encoder, label_map)
    val_dataset = ABSADataset(val_df, encoder, label_map)

    class_weights = (
        train_dataset.class_weights() if get(cfg, "data.class_weighting", True) else None
    )
    model = BertAspectAttention(
        model_name=get(cfg, "model.name"),
        num_labels=int(get(cfg, "model.num_labels")),
        dropout=float(get(cfg, "model.dropout", 0.1)),
        class_weights=class_weights,
    )
    LOGGER.info("model: %r (%s params)", model, f"{model.num_trainable_parameters():,}")

    batch_size = int(get(cfg, "training.batch_size"))
    eval_batch_size = int(get(cfg, "optim.eval_batch_size", batch_size) or batch_size)
    train_loader = make_loader(train_dataset, batch_size, True, cfg)
    val_loader = make_loader(val_dataset, eval_batch_size, False, cfg)

    epochs = int(args.epochs or get(cfg, "training.epochs"))
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            lr=float(get(cfg, "training.lr")),
            weight_decay=float(get(cfg, "training.weight_decay", 0.01)),
            head_lr=(
                float(get(cfg, "optim.head_lr")) if get(cfg, "optim.head_lr") else None
            ),
        ),
        lr=float(get(cfg, "training.lr")),
    )
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(get(cfg, "training.warmup_ratio", 0.1))),
        num_training_steps=total_steps,
    )

    run_dir = artifact_dir(cfg)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config={
            "max_grad_norm": float(get(cfg, "training.max_grad_norm", 1.0)),
            "label_map": label_map,
            "checkpoint_dir": str(run_dir),
        },
    )
    history = trainer.fit(epochs)

    print()
    print(reporting.format_history(history))
    print()
    if trainer.best_val_metrics:
        print(reporting.format_metrics(
            trainer.best_val_metrics, f"Validation (best epoch {trainer.best_epoch})"
        ))
        print()

    # Score the official gold test set with the best weights, so the artifact
    # ships the number that actually matters.
    test_metrics: dict[str, Any] = {}
    test_path = resolve_path(cfg, "paths.official_test")
    if test_path.is_file():
        test_df = load_official_test(test_path, label_map.labels)
        test_loader = make_loader(
            ABSADataset(test_df, encoder, label_map), eval_batch_size, False, cfg
        )
        test_metrics = evaluate(model, test_loader, device, label_map, split="official_test")
        print(reporting.format_metrics(test_metrics, "Official gold test"))
        print()
    else:
        LOGGER.warning("no gold test file at %s; artifact metrics will be validation only", test_path)
        test_metrics = trainer.best_val_metrics or {}

    directory = build_artifact(
        cfg=for_artifact(cfg),
        run_id=args.run_id or str(get(cfg, "artifact.version")),
        model=model,
        encoder=encoder,
        label_map=label_map,
        test_metrics=test_metrics,
        best_epoch=trainer.best_epoch,
        state_dict=trainer.best_state,
        history=history,
    )
    if trainer.best_val_metrics:
        reporting.save_metrics_report(
            trainer.best_val_metrics, directory, stem="validation_metrics",
            title=f"Validation (best epoch {trainer.best_epoch})",
        )

    print(f"artifact   : {directory}")
    print(f"best epoch : {trainer.best_epoch} (val macro_f1={trainer.best_macro_f1:.4f})")
    return 0


# --------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------- #


def cmd_evaluate(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Score a saved artifact on the official gold test set."""
    directory = Path(args.artifact) if args.artifact else artifact_dir(cfg)
    device = pick_device(args.device)

    model, encoder, label_map, art_config = load_artifact(directory, device=device)
    LOGGER.info("loaded %s (best epoch %s)", directory, art_config.get("best_epoch"))

    test_path = Path(args.test_csv) if args.test_csv else resolve_path(cfg, "paths.official_test")
    test_df = load_official_test(test_path, label_map.labels)
    dataset = ABSADataset(test_df, encoder, label_map)
    loader = make_loader(
        dataset, int(get(cfg, "optim.eval_batch_size", 64) or 64), False, cfg
    )

    # One inference pass, then score it -- the reports below all read from
    # these same predictions.
    outputs = predict(model, loader, device)
    if outputs["y_true"] is None:
        raise ValueError(f"{test_path} has no labels to score against")
    metrics = compute_metrics(
        outputs["y_true"], outputs["y_pred"], label_map.labels,
        loss=outputs["loss"], split=test_path.name,
    )
    print(reporting.format_metrics(metrics, f"Official gold test — {directory.name}"))
    print()
    print(sklearn_text_report(outputs["y_true"], outputs["y_pred"], label_map.labels))

    out_dir = Path(args.out_dir) if args.out_dir else directory
    paths = reporting.save_metrics_report(
        metrics, out_dir, stem="evaluation", title="Official gold test"
    )
    reporting.save_predictions_csv(
        test_df, outputs["y_true"], outputs["y_pred"], label_map.labels,
        outputs["probabilities"], out_dir / "evaluation_predictions.csv",
    )
    print()
    for kind, path in paths.items():
        print(f"{kind:<5}: {path}")

    if args.show_attention:
        print("\n=== attention over context (first few examples) ===")
        pairs = list(zip(test_df["text"].head(3), test_df["aspect"].head(3)))
        for result in predict_pairs(model, encoder, label_map, pairs):
            top = sorted(result["attention"], key=lambda t: -t["weight"])[:6]
            tokens = "  ".join(f"{t['token']}({t['weight']:.2f})" for t in top)
            print(f"\naspect={result['aspect']!r} -> {result['label']} "
                  f"({result['confidence']:.3f})\n  {tokens}")
    return 0


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py", description="Aspect-based sentiment analysis pipeline."
    )
    parser.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--verbose", action="store_true", help="debug-level logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="split the corpus into train/val")
    prepare.add_argument("--val-fraction", type=float, dest="val_fraction")
    prepare.add_argument("--seed", type=int)
    prepare.set_defaults(func=cmd_prepare)

    train = subparsers.add_parser("train", help="fine-tune and write an artifact")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int, dest="batch_size")
    train.add_argument("--lr", type=float)
    train.add_argument("--max-length", type=int, dest="max_length")
    train.add_argument("--model-name", dest="model_name", help="HuggingFace backbone id")
    train.add_argument("--version", help="artifact version directory name")
    train.add_argument("--run-id", dest="run_id", help="run id recorded in config.json")
    train.add_argument("--seed", type=int)
    train.add_argument("--device", help="cuda | cpu (default: cuda when available)")
    train.add_argument("--prepare", action="store_true", help="re-split before training")
    train.set_defaults(func=cmd_train)

    evaluate_cmd = subparsers.add_parser("evaluate", help="score a saved artifact")
    evaluate_cmd.add_argument("--artifact", help="artifact directory (default: from config)")
    evaluate_cmd.add_argument("--test-csv", dest="test_csv", help="CSV to score")
    evaluate_cmd.add_argument("--out-dir", dest="out_dir", help="where to write reports")
    evaluate_cmd.add_argument("--device", help="cuda | cpu")
    evaluate_cmd.add_argument("--version", help="artifact version directory name")
    evaluate_cmd.add_argument(
        "--show-attention", action="store_true",
        help="print the most-attended context tokens for a few examples",
    )
    evaluate_cmd.set_defaults(func=cmd_evaluate)

    return parser


#: CLI flag -> dotted config key.
OVERRIDES = {
    "seed": "seed",
    "model_name": "model.name",
    "max_length": "model.max_length",
    "epochs": "training.epochs",
    "batch_size": "training.batch_size",
    "lr": "training.lr",
    "val_fraction": "training.val_fraction",
    "version": "artifact.version",
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    overrides = {
        key: getattr(args, flag)
        for flag, key in OVERRIDES.items()
        if getattr(args, flag, None) is not None
    }
    cfg = load_config(args.config, overrides)
    if overrides:
        LOGGER.info("config overrides: %s", overrides)

    try:
        return args.func(cfg, args)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
