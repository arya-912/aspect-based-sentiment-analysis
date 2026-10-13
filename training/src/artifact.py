"""Save and reload a trained ABSA model as a self-contained artifact bundle.

Layout of an artifact directory::

    <run>/
      config.yaml          the full ExperimentConfig used for the run
      model.pt             state dict of the best checkpoint
      metadata.json        labels, metrics, versions, provenance
      tokenizer/           saved AutoTokenizer files

The bundle carries everything inference needs, so the backend can load a run
without importing the training configuration by hand.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .data.dataset import LabelMap
from .models.absa_model import ABSAModel
from .preprocessing.tokenizer import ABSAEncoder

LOGGER = logging.getLogger(__name__)

CONFIG_FILE = "config.yaml"
WEIGHTS_FILE = "model.pt"
METADATA_FILE = "metadata.json"
TOKENIZER_DIR = "tokenizer"


def _versions() -> dict[str, str]:
    """Record the library versions a checkpoint was produced with."""
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover - transformers is a hard dep in practice
        pass
    return versions


@dataclass
class ArtifactMetadata:
    """Everything about a run that is not weights or config."""

    name: str
    model_name: str
    labels: tuple[str, ...]
    max_length: int
    pooling: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    best_epoch: int | None = None
    monitor: str | None = None
    best_metric: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    dataset: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=_versions)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["labels"] = list(self.labels)
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ArtifactMetadata":
        raw = dict(raw)
        raw["labels"] = tuple(raw.get("labels", ()))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class LoadedArtifact:
    """A rehydrated run: model, encoder, label map and metadata."""

    model: ABSAModel
    encoder: ABSAEncoder
    label_map: LabelMap
    config: ExperimentConfig
    metadata: ArtifactMetadata
    directory: Path

    def predict(self, text: str, aspect: str) -> dict[str, Any]:
        """Classify one (sentence, aspect) pair."""
        return self.predict_batch([(text, aspect)])[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """Classify many pairs in a single forward pass."""
        if not pairs:
            return []
        device = next(self.model.parameters()).device
        encoded = [self.encoder.encode(text, aspect) for text, aspect in pairs]
        batch = {
            key: torch.stack([ex[key] for ex in encoded]).to(device)
            for key in encoded[0]
        }

        self.model.eval()
        with torch.no_grad():
            logits = self.model(**batch)["logits"]
            probs = torch.softmax(logits.float(), dim=-1).cpu()

        results = []
        for (text, aspect), row in zip(pairs, probs):
            label_id = int(row.argmax())
            results.append(
                {
                    "text": text,
                    "aspect": aspect,
                    "label": self.label_map.decode(label_id),
                    "confidence": float(row[label_id]),
                    "scores": {
                        name: float(row[i])
                        for i, name in enumerate(self.label_map.labels)
                    },
                }
            )
        return results


def save_artifact(
    directory: str | Path,
    model: ABSAModel,
    encoder: ABSAEncoder,
    config: ExperimentConfig,
    metadata: ArtifactMetadata,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Write a complete artifact bundle and return its directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    weights = state_dict if state_dict is not None else model.state_dict()
    torch.save(
        {k: v.detach().cpu() for k, v in weights.items()}, directory / WEIGHTS_FILE
    )
    config.save(directory / CONFIG_FILE)
    encoder.save(directory / TOKENIZER_DIR)

    with (directory / METADATA_FILE).open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, indent=2, default=str)

    LOGGER.info("saved artifact to %s", directory)
    return directory


def load_artifact(
    directory: str | Path, device: str | torch.device = "cpu"
) -> LoadedArtifact:
    """Reload a bundle written by :func:`save_artifact`.

    The tokenizer is read from the bundle when present so inference does not
    depend on re-downloading the hub model.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"no such artifact directory: {directory}")
    for required in (WEIGHTS_FILE, CONFIG_FILE, METADATA_FILE):
        if not (directory / required).is_file():
            raise FileNotFoundError(f"{directory} is missing {required}")

    config = ExperimentConfig.load(directory / CONFIG_FILE)
    with (directory / METADATA_FILE).open(encoding="utf-8") as handle:
        metadata = ArtifactMetadata.from_dict(json.load(handle))

    label_map = LabelMap(labels=tuple(metadata.labels or config.data.labels))

    tokenizer_dir = directory / TOKENIZER_DIR
    encoder = ABSAEncoder(
        model_name=str(tokenizer_dir) if tokenizer_dir.is_dir() else metadata.model_name,
        max_length=metadata.max_length or config.model.max_length,
    )

    model = ABSAModel(
        model_name=metadata.model_name or config.model.model_name,
        num_labels=label_map.num_labels,
        dropout=config.model.dropout,
        pooling=metadata.pooling or config.model.pooling,
        hidden_dim=config.model.hidden_dim,
    )
    state = torch.load(directory / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        LOGGER.warning(
            "state dict mismatch: %d missing, %d unexpected key(s)",
            len(missing),
            len(unexpected),
        )
    model.to(device).eval()

    LOGGER.info("loaded artifact %s (labels=%s)", directory.name, label_map.labels)
    return LoadedArtifact(
        model=model,
        encoder=encoder,
        label_map=label_map,
        config=config,
        metadata=metadata,
        directory=directory,
    )
