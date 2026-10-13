"""Loading and validating the YAML training config.

The config is a plain nested dict rather than a dataclass tree: it is written
verbatim into every artifact's ``config.json``, so keeping it as data means an
artifact records exactly what produced it, with no schema to keep in sync.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

#: Repository root: this file is <root>/training/src/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "default.yaml"

#: Dotted keys every config must define.
REQUIRED_KEYS = (
    "seed",
    "model.name",
    "model.max_length",
    "model.num_labels",
    "training.epochs",
    "training.batch_size",
    "training.lr",
    "training.val_fraction",
    "paths.raw_train",
    "paths.official_test",
    "paths.processed_dir",
    "paths.artifacts_dir",
    "artifact.version",
)


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read a nested value by dotted key, e.g. ``get(cfg, "model.name")``."""
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def set_in(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """Write a nested value by dotted key, creating sections as needed."""
    cursor = cfg
    *parents, leaf = dotted.split(".")
    for part in parents:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise TypeError(f"{dotted}: '{part}' is not a section")
    cursor[leaf] = value


#: Dotted keys coerced to a number after loading, with their target type.
#: PyYAML only parses a float when it sees both a dot and a signed exponent,
#: so the perfectly reasonable ``lr: 3e-5`` loads as the *string* "3e-5" and
#: would otherwise reach AdamW untouched and fail far from the cause.
NUMERIC_KEYS: dict[str, type] = {
    "seed": int,
    "model.max_length": int,
    "model.num_labels": int,
    "model.dropout": float,
    "training.epochs": int,
    "training.batch_size": int,
    "training.lr": float,
    "training.weight_decay": float,
    "training.warmup_ratio": float,
    "training.max_grad_norm": float,
    "training.val_fraction": float,
    "optim.head_lr": float,
    "optim.eval_batch_size": int,
    "optim.num_workers": int,
}


def coerce_numeric(cfg: dict[str, Any]) -> None:
    """Coerce the keys in :data:`NUMERIC_KEYS` in place."""
    for dotted, caster in NUMERIC_KEYS.items():
        value = get(cfg, dotted, None)
        if value is None or isinstance(value, bool):
            continue
        try:
            set_in(cfg, dotted, caster(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"config key {dotted!r} must be a {caster.__name__}, got {value!r}"
            ) from exc


def validate_config(cfg: dict[str, Any]) -> None:
    """Fail loudly on a config missing keys the pipeline depends on."""
    missing = [key for key in REQUIRED_KEYS if get(cfg, key, None) is None]
    if missing:
        raise KeyError(f"config is missing required key(s): {missing}")

    if not 0.0 < float(get(cfg, "training.val_fraction")) < 1.0:
        raise ValueError(
            f"training.val_fraction must be in (0, 1), got "
            f"{get(cfg, 'training.val_fraction')}"
        )
    if int(get(cfg, "training.epochs")) < 1:
        raise ValueError(f"training.epochs must be >= 1, got {get(cfg, 'training.epochs')}")
    if int(get(cfg, "model.num_labels")) < 2:
        raise ValueError(
            f"model.num_labels must be >= 2, got {get(cfg, 'model.num_labels')}"
        )
    labels = get(cfg, "data.labels")
    if labels is not None and len(labels) != int(get(cfg, "model.num_labels")):
        raise ValueError(
            f"model.num_labels ({get(cfg, 'model.num_labels')}) does not match "
            f"data.labels ({labels})"
        )


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load the YAML config, apply dotted overrides and validate it.

    Args:
        path: YAML file; defaults to ``training/configs/default.yaml``.
        overrides: dotted key -> value, applied on top of the file.

    Returns:
        The resolved config dict.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"no such config file: {path}")

    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"{path}: expected a mapping at the top level")

    cfg["config_path"] = str(path)
    for dotted, value in (overrides or {}).items():
        if value is not None:
            set_in(cfg, dotted, value)

    coerce_numeric(cfg)
    validate_config(cfg)
    return cfg


def resolve_path(cfg: dict[str, Any], dotted: str) -> Path:
    """Resolve a configured path against the repository root.

    Config paths are written repo-relative (``training/data/raw/...``) so the
    same file works whether a command is run from the repo root or from
    ``training/``.
    """
    raw = get(cfg, dotted)
    if raw is None:
        raise KeyError(f"config has no path at {dotted!r}")
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def artifact_dir(cfg: dict[str, Any]) -> Path:
    """Where this run's artifact belongs: ``<artifacts_dir>/<version>``."""
    return resolve_path(cfg, "paths.artifacts_dir") / str(get(cfg, "artifact.version"))


def describe(cfg: dict[str, Any]) -> str:
    """One-line summary for the log at the start of a run."""
    return (
        f"model={get(cfg, 'model.name')} "
        f"max_len={get(cfg, 'model.max_length')} "
        f"epochs={get(cfg, 'training.epochs')} "
        f"bs={get(cfg, 'training.batch_size')} "
        f"lr={get(cfg, 'training.lr')} "
        f"seed={get(cfg, 'seed')}"
    )


def for_artifact(cfg: dict[str, Any]) -> dict[str, Any]:
    """A copy with repo-relative paths kept intact, for ``config.json``."""
    return copy.deepcopy(cfg)
