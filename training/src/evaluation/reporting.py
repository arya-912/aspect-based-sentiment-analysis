"""Human-readable and machine-readable renderings of evaluation results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluate import EvalResult

LOGGER = logging.getLogger(__name__)


def format_confusion_matrix(
    matrix: Sequence[Sequence[int]], labels: Sequence[str]
) -> str:
    """Render a confusion matrix as an aligned text table (rows = gold)."""
    width = max([len(str(label)) for label in labels] + [5])
    cell = max(width, max((len(str(v)) for row in matrix for v in row), default=1))

    header = " " * (width + 3) + "  ".join(f"{label:>{cell}}" for label in labels)
    lines = [f"{'':{width}}   pred ->", header]
    for label, row in zip(labels, matrix):
        lines.append(
            f"{label:>{width}} | " + "  ".join(f"{int(v):>{cell}}" for v in row)
        )
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


def format_result(result: EvalResult, title: str = "Evaluation") -> str:
    """Full text report: headline scalars, per-class table, confusion matrix."""
    loss = "n/a" if result.loss is None else f"{result.loss:.4f}"
    return "\n".join(
        [
            f"=== {title} ===",
            f"examples    : {len(result.y_true)}",
            f"loss        : {loss}",
            f"accuracy    : {result.accuracy:.4f}",
            f"macro F1    : {result.macro_f1:.4f}",
            f"weighted F1 : {result.weighted_f1:.4f}",
            "",
            format_per_class(result.per_class),
            "",
            format_confusion_matrix(result.confusion_matrix, result.labels),
        ]
    )


def format_markdown(result: EvalResult, title: str = "Evaluation") -> str:
    """Same content as Markdown, for the README or a notebook."""
    loss = "n/a" if result.loss is None else f"{result.loss:.4f}"
    lines = [
        f"## {title}",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| examples | {len(result.y_true)} |",
        f"| loss | {loss} |",
        f"| accuracy | {result.accuracy:.4f} |",
        f"| macro F1 | {result.macro_f1:.4f} |",
        f"| weighted F1 | {result.weighted_f1:.4f} |",
        "",
        "| label | precision | recall | f1 | support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, scores in result.per_class.items():
        lines.append(
            f"| {name} | {scores['precision']:.4f} | {scores['recall']:.4f} "
            f"| {scores['f1']:.4f} | {int(scores['support'])} |"
        )
    lines += ["", "### Confusion matrix (rows = gold)", "", "```", 
              format_confusion_matrix(result.confusion_matrix, result.labels), "```"]
    return "\n".join(lines)


def format_history(history: Sequence[Mapping[str, Any]]) -> str:
    """Render the per-epoch training history as a table."""
    if not history:
        return "(no epochs recorded)"
    columns = ["epoch", "train_loss", "val_loss", "accuracy", "macro_f1", "lr"]
    present = [c for c in columns if any(c in row for row in history)]
    lines = [
        "  ".join(f"{c:>11}" for c in present),
        "-" * (13 * len(present)),
    ]
    for row in history:
        cells = []
        for col in present:
            value = row.get(col)
            if value is None:
                cells.append(f"{'-':>11}")
            elif col == "epoch":
                cells.append(f"{int(value):>11}")
            elif col == "lr":
                cells.append(f"{float(value):>11.2e}")
            else:
                cells.append(f"{float(value):>11.4f}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def save_json(payload: Any, path: str | Path) -> Path:
    """Write a JSON report, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=str)
    LOGGER.info("wrote %s", path)
    return path


def save_report(
    result: EvalResult,
    directory: str | Path,
    stem: str = "evaluation",
    title: str = "Evaluation",
    include_predictions: bool = True,
) -> dict[str, Path]:
    """Persist a result as ``<stem>.json``, ``<stem>.txt`` and ``<stem>.md``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = save_json(result.to_dict(include_predictions), directory / f"{stem}.json")
    txt_path = directory / f"{stem}.txt"
    txt_path.write_text(format_result(result, title) + "\n", encoding="utf-8")
    md_path = directory / f"{stem}.md"
    md_path.write_text(format_markdown(result, title) + "\n", encoding="utf-8")
    return {"json": json_path, "txt": txt_path, "md": md_path}


def save_predictions_csv(
    result: EvalResult,
    frame,
    path: str | Path,
) -> Path:
    """Write a per-example predictions CSV alongside the source columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.reset_index(drop=True).copy()
    out["gold"] = [result.labels[i] for i in result.y_true]
    out["predicted"] = [result.labels[i] for i in result.y_pred]
    out["correct"] = out["gold"] == out["predicted"]
    if result.probabilities:
        for idx, name in enumerate(result.labels):
            out[f"p_{name}"] = [row[idx] for row in result.probabilities]
    out.to_csv(path, index=False)
    LOGGER.info("wrote %s", path)
    return path
