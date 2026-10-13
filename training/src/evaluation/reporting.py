"""Rendering of metric dicts for the console and for saved reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)


def format_confusion_matrix(
    matrix: Sequence[Sequence[int]], labels: Sequence[str]
) -> str:
    """Render a confusion matrix as an aligned text table (rows = gold)."""
    width = max([len(str(label)) for label in labels] + [5])
    cell = max(width, max((len(str(v)) for row in matrix for v in row), default=1))
    lines = [
        f"{'':{width}}   pred ->",
        " " * (width + 3) + "  ".join(f"{label:>{cell}}" for label in labels),
    ]
    for label, row in zip(labels, matrix):
        lines.append(f"{label:>{width}} | " + "  ".join(f"{int(v):>{cell}}" for v in row))
    return "\n".join(lines)


def format_per_class(per_class: Mapping[str, Mapping[str, float]]) -> str:
    """Render per-class precision/recall/F1/support as an aligned table."""
    name_width = max([len(name) for name in per_class] + [5])
    lines = [
        f"{'label':<{name_width}}  {'prec':>7}  {'recall':>7}  {'f1':>7}  {'n':>6}",
        "-" * (name_width + 34),
    ]
    for name, scores in per_class.items():
        lines.append(
            f"{name:<{name_width}}  {scores['precision']:>7.4f}  "
            f"{scores['recall']:>7.4f}  {scores['f1']:>7.4f}  "
            f"{int(scores['support']):>6}"
        )
    return "\n".join(lines)


def format_metrics(metrics: Mapping[str, Any], title: str | None = None) -> str:
    """Full text report for one metric dict."""
    title = title or f"Evaluation ({metrics.get('evaluation_split', 'unknown')})"
    loss = metrics.get("loss")
    lines = [
        f"=== {title} ===",
        f"examples    : {metrics.get('n', '?')}",
        f"loss        : {'n/a' if loss is None else f'{loss:.4f}'}",
        f"accuracy    : {metrics['accuracy']:.4f}",
        f"macro F1    : {metrics['macro_f1']:.4f}",
    ]
    if "weighted_f1" in metrics:
        lines.append(f"weighted F1 : {metrics['weighted_f1']:.4f}")
    if metrics.get("per_class"):
        lines += ["", format_per_class(metrics["per_class"])]
    if metrics.get("confusion_matrix"):
        lines += [
            "",
            format_confusion_matrix(
                metrics["confusion_matrix"], list(metrics["per_class"])
            ),
        ]
    return "\n".join(lines)


def format_markdown(metrics: Mapping[str, Any], title: str | None = None) -> str:
    """The same content as Markdown, for a README or notebook."""
    title = title or f"Evaluation ({metrics.get('evaluation_split', 'unknown')})"
    loss = metrics.get("loss")
    lines = [
        f"## {title}",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| examples | {metrics.get('n', '?')} |",
        f"| loss | {'n/a' if loss is None else f'{loss:.4f}'} |",
        f"| accuracy | {metrics['accuracy']:.4f} |",
        f"| macro F1 | {metrics['macro_f1']:.4f} |",
    ]
    if "weighted_f1" in metrics:
        lines.append(f"| weighted F1 | {metrics['weighted_f1']:.4f} |")
    if metrics.get("per_class"):
        lines += [
            "",
            "| label | precision | recall | f1 | support |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, scores in metrics["per_class"].items():
            lines.append(
                f"| {name} | {scores['precision']:.4f} | {scores['recall']:.4f} "
                f"| {scores['f1']:.4f} | {int(scores['support'])} |"
            )
    if metrics.get("confusion_matrix"):
        lines += [
            "",
            "### Confusion matrix (rows = gold)",
            "",
            "```",
            format_confusion_matrix(metrics["confusion_matrix"], list(metrics["per_class"])),
            "```",
        ]
    return "\n".join(lines)


def format_history(history: Sequence[Mapping[str, Any]]) -> str:
    """Render the per-epoch training history as a table."""
    if not history:
        return "(no epochs recorded)"
    preferred = [
        "epoch", "train_loss", "train_accuracy", "train_macro_f1",
        "val_loss", "val_accuracy", "val_macro_f1", "lr", "seconds",
    ]
    columns = [c for c in preferred if any(c in row for row in history)]
    lines = ["  ".join(f"{c:>14}" for c in columns), "-" * (16 * len(columns))]
    for row in history:
        cells = []
        for col in columns:
            value = row.get(col)
            if value is None:
                cells.append(f"{'-':>14}")
            elif col == "epoch":
                cells.append(f"{int(value):>14}")
            elif col == "lr":
                cells.append(f"{float(value):>14.2e}")
            else:
                cells.append(f"{float(value):>14.4f}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def save_json(payload: Any, path: str | Path) -> Path:
    """Write JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    LOGGER.info("wrote %s", path)
    return path


def save_metrics_report(
    metrics: Mapping[str, Any],
    directory: str | Path,
    stem: str = "metrics",
    title: str | None = None,
) -> dict[str, Path]:
    """Persist a metric dict as ``<stem>.json``, ``.txt`` and ``.md``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"json": save_json(dict(metrics), directory / f"{stem}.json")}
    for key, renderer in (("txt", format_metrics), ("md", format_markdown)):
        path = directory / f"{stem}.{key}"
        path.write_text(renderer(metrics, title) + "\n", encoding="utf-8")
        paths[key] = path
    return paths


def save_predictions_csv(frame, y_true, y_pred, labels, probabilities, path) -> Path:
    """Write a per-example predictions CSV beside the source columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.reset_index(drop=True).copy()
    out["gold"] = [labels[int(i)] for i in y_true]
    out["predicted"] = [labels[int(i)] for i in y_pred]
    out["correct"] = out["gold"] == out["predicted"]
    if probabilities is not None:
        for idx, name in enumerate(labels):
            out[f"p_{name}"] = [float(row[idx]) for row in probabilities]
    out.to_csv(path, index=False)
    LOGGER.info("wrote %s", path)
    return path
