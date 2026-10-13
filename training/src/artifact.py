"""Package a trained run as a self-contained, offline-loadable artifact.

Layout of ``training/artifacts/<version>/``::

    model.pt         state dict of the best checkpoint (weights only)
    encoder/         the base transformer's *config* -- enough to rebuild the
                     architecture without touching the network
    tokenizer/       saved tokenizer files
    config.json      training hyperparameters + artifact version
    label_map.json   label2id / id2label
    metrics.json     accuracy, macro_f1, per_class, confusion_matrix

``encoder/`` deliberately holds the config and not a second copy of the
weights: every weight the model needs is already in ``model.pt``, so
reconstruction is ``AutoModel.from_config`` followed by a state-dict load.
That makes :func:`load_artifact` work with no network and no duplicated
~440MB of parameters.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel

from .config import get as cfg_get
from .config import resolve_path
from .data.dataset import LabelMap
from .models.absa_model import BertAspectAttention
from .preprocessing.tokenizer import ABSAEncoder

LOGGER = logging.getLogger(__name__)

MODEL_FILE = "model.pt"
ENCODER_DIR = "encoder"
TOKENIZER_DIR = "tokenizer"
CONFIG_FILE = "config.json"
LABEL_MAP_FILE = "label_map.json"
METRICS_FILE = "metrics.json"

#: Everything :func:`load_artifact` needs to rebuild a run.
REQUIRED_ENTRIES = (MODEL_FILE, ENCODER_DIR, CONFIG_FILE, LABEL_MAP_FILE)


def _library_versions() -> dict[str, str]:
    """Record what produced a checkpoint, for debugging a stale artifact."""
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover - transformers is a hard dependency
        pass
    return versions


def _write_json(payload: Any, path: Path) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def build_artifact(
    cfg: dict[str, Any],
    run_id: str,
    model: BertAspectAttention,
    encoder: ABSAEncoder,
    label_map: LabelMap,
    test_metrics: dict[str, Any],
    best_epoch: int,
    state_dict: dict[str, torch.Tensor] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a complete artifact bundle.

    Args:
        cfg: the resolved training config; stored verbatim in ``config.json``.
        run_id: identifies this run inside ``config.json``.
        model: the trained model (best weights already loaded).
        encoder: the :class:`ABSAEncoder`, whose tokenizer is saved.
        label_map: the label vocabulary.
        test_metrics: the metric dict to store in ``metrics.json``.
        best_epoch: which epoch the saved weights came from.
        state_dict: weights to save; defaults to the model's current ones.
        history: optional per-epoch training history.

    Returns:
        The artifact directory.
    """
    version = str(cfg_get(cfg, "artifact.version") or run_id or "artifact")
    # Resolved through config.resolve_path, so a repo-relative artifacts_dir
    # lands in the same place whether the command ran from the repo root or
    # from training/ -- and in the same place as the trainer's checkpoint.
    artifacts_root = (
        resolve_path(cfg, "paths.artifacts_dir")
        if cfg_get(cfg, "paths.artifacts_dir")
        else Path("training/artifacts")
    )
    directory = artifacts_root / version
    directory.mkdir(parents=True, exist_ok=True)

    # 1. weights only -- the architecture is rebuilt from encoder/ + config.
    weights = state_dict if state_dict is not None else model.state_dict()
    torch.save({k: v.detach().cpu() for k, v in weights.items()}, directory / MODEL_FILE)

    # 2. the base transformer's config, so reconstruction needs no network.
    encoder_dir = directory / ENCODER_DIR
    encoder_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(encoder_dir)

    # 3. the tokenizer.
    encoder.save(directory / TOKENIZER_DIR)

    # 4. hyperparameters and provenance.
    _write_json(
        {
            **cfg,
            "run_id": run_id,
            "version": version,
            "best_epoch": best_epoch,
            "num_labels": label_map.num_labels,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "library_versions": _library_versions(),
            **({"history": history} if history else {}),
        },
        directory / CONFIG_FILE,
    )

    # 5. the label vocabulary, so inference never has to guess the order.
    _write_json(
        {
            "labels": list(label_map.labels),
            "label2id": label_map.label2id,
            "id2label": {str(k): v for k, v in label_map.id2label.items()},
        },
        directory / LABEL_MAP_FILE,
    )

    # 6. how well it did.
    _write_json(dict(test_metrics), directory / METRICS_FILE)

    LOGGER.info("built artifact %s (best epoch %s)", directory, best_epoch)
    return directory


def load_artifact(
    path: str | Path, device: str | torch.device | None = None
) -> tuple[BertAspectAttention, ABSAEncoder, LabelMap, dict[str, Any]]:
    """Reload a bundle written by :func:`build_artifact`, offline.

    Args:
        path: the artifact directory.
        device: where to place the model; defaults to cuda when available.

    Returns:
        ``(model, encoder, label_map, config_dict)``. The model is in
        ``eval()`` mode.

    Raises:
        FileNotFoundError: if the directory or any required entry is missing.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"no such artifact directory: {directory}")

    missing = [name for name in REQUIRED_ENTRIES if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"incomplete artifact {directory}: missing {', '.join(sorted(missing))}"
        )

    with (directory / CONFIG_FILE).open(encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)
    with (directory / LABEL_MAP_FILE).open(encoding="utf-8") as handle:
        label_payload = json.load(handle)

    labels = label_payload.get("labels")
    if not labels:
        # Fall back to label2id ordering if "labels" is absent.
        label2id = label_payload.get("label2id") or {}
        labels = [name for name, _ in sorted(label2id.items(), key=lambda kv: kv[1])]
    label_map = LabelMap(labels=tuple(labels))

    model_cfg = config.get("model") or {}
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Rebuild the backbone from the saved config -- no download, no second
    # copy of the pretrained weights.
    encoder_config = AutoConfig.from_pretrained(directory / ENCODER_DIR)
    backbone = AutoModel.from_config(encoder_config)

    model = BertAspectAttention(
        model_name=model_cfg.get("name") or getattr(encoder_config, "name_or_path", None),
        num_labels=int(config.get("num_labels") or model_cfg.get("num_labels") or label_map.num_labels),
        dropout=float(model_cfg.get("dropout", 0.1)),
        encoder=backbone,
    )

    state = torch.load(directory / MODEL_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    tokenizer_dir = directory / TOKENIZER_DIR
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"incomplete artifact {directory}: missing {TOKENIZER_DIR}")
    encoder = ABSAEncoder(
        model_name=str(tokenizer_dir),
        max_length=int(model_cfg.get("max_length", 128)),
    )

    LOGGER.info(
        "loaded artifact %s on %s (labels=%s)", directory.name, device, label_map.labels
    )
    return model, encoder, label_map, config


def read_metrics(path: str | Path) -> dict[str, Any]:
    """Read an artifact's ``metrics.json``."""
    metrics_path = Path(path) / METRICS_FILE
    if not metrics_path.is_file():
        raise FileNotFoundError(f"no metrics at {metrics_path}")
    with metrics_path.open(encoding="utf-8") as handle:
        return json.load(handle)


@torch.no_grad()
def predict_pairs(
    model: BertAspectAttention,
    encoder: ABSAEncoder,
    label_map: LabelMap,
    pairs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Classify ``(sentence, aspect)`` pairs with a loaded artifact.

    Also returns each example's attention over its context tokens, which is
    the main reason this architecture is worth its extra parameters.
    """
    if not pairs:
        return []
    device = next(model.parameters()).device
    encoded = [encoder.encode(text, aspect) for text, aspect in pairs]
    batch = {
        key: torch.stack([ex[key] for ex in encoded]).to(device) for key in encoded[0]
    }

    model.eval()
    output = model(**batch)
    probs = torch.softmax(output["logits"].float(), dim=-1).cpu()
    attention = output["attention"].cpu()

    results = []
    for i, ((text, aspect), row) in enumerate(zip(pairs, probs)):
        label_id = int(row.argmax())
        tokens = encoder.tokenizer.convert_ids_to_tokens(batch["input_ids"][i].cpu())
        context = batch["context_mask"][i].cpu()
        results.append(
            {
                "text": text,
                "aspect": aspect,
                "label": label_map.decode(label_id),
                "confidence": float(row[label_id]),
                "scores": {
                    name: float(row[j]) for j, name in enumerate(label_map.labels)
                },
                "attention": [
                    {"token": tok, "weight": float(w)}
                    for tok, w, keep in zip(tokens, attention[i].tolist(), context.tolist())
                    if keep
                ],
            }
        )
    return results
