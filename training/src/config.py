"""Typed, YAML-backed configuration for an ABSA training run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "training"


@dataclass
class DataConfig:
    """Where the corpus lives and how it is split."""

    train_csv: str = "data/raw/Laptop_Train_v2.csv"
    test_csv: str = "data/raw/Laptops_Test_Gold.csv"
    splits_dir: str = "data/processed/splits"
    labels: tuple[str, ...] = ("negative", "neutral", "positive")
    val_fraction: float = 0.1
    seed: int = 42
    save_splits: bool = True


@dataclass
class ModelConfig:
    """Backbone and classification head."""

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    dropout: float = 0.1
    pooling: str = "aspect_context"  # cls | aspect | aspect_context
    freeze_encoder: bool = False
    hidden_dim: int | None = None  # None -> straight linear head


@dataclass
class TrainConfig:
    """Optimisation schedule."""

    epochs: int = 4
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    class_weighting: bool = True
    num_workers: int = 0
    device: str = "auto"  # auto | cpu | cuda | mps
    seed: int = 42
    early_stopping_patience: int = 0  # 0 disables
    monitor: str = "macro_f1"  # metric used to pick the best checkpoint


@dataclass
class ExperimentConfig:
    """Everything one run needs."""

    name: str = "absa-laptops"
    artifacts_dir: str = "artifacts"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- paths -------------------------------------------------------- #

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a config path against ``training/`` unless already absolute."""
        path = Path(relative)
        return path if path.is_absolute() else (TRAINING_ROOT / path)

    @property
    def run_dir(self) -> Path:
        return self.resolve(self.artifacts_dir) / self.name

    # ---- (de)serialisation -------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ExperimentConfig":
        return _build(cls, raw or {})

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no such config file: {path}")
        with path.open(encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        return path

    def with_overrides(self, overrides: dict[str, Any]) -> "ExperimentConfig":
        """Return a copy updated from dotted keys, e.g. ``train.epochs=2``."""
        merged = self.to_dict()
        for dotted, value in overrides.items():
            cursor: Any = merged
            *parents, leaf = dotted.split(".")
            for part in parents:
                if part not in cursor:
                    raise KeyError(f"unknown config section {part!r} in {dotted!r}")
                cursor = cursor[part]
            if leaf not in cursor:
                raise KeyError(f"unknown config key {dotted!r}")
            cursor[leaf] = value
        return ExperimentConfig.from_dict(merged)


def _build(target: type, raw: Any) -> Any:
    """Recursively instantiate nested dataclasses from plain dicts."""
    if not is_dataclass(target):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"expected a mapping for {target.__name__}, got {type(raw)}")

    known = {f.name for f in fields(target)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise KeyError(f"unknown key(s) for {target.__name__}: {unknown}")

    # Resolved, because `from __future__ import annotations` stringifies types.
    hints = get_type_hints(target)

    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in raw:
            continue
        value = raw[name]
        hint = hints.get(name)
        if is_dataclass(hint):
            kwargs[name] = _build(hint, value)
        elif name == "labels" and isinstance(value, (list, tuple)):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return target(**kwargs)
