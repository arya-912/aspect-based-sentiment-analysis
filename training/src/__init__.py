"""Training package for aspect-based sentiment analysis on the SemEval laptop corpus."""

from .artifact import build_artifact, load_artifact, predict_pairs, read_metrics
from .config import artifact_dir, get, load_config, resolve_path, validate_config
from .data import ABSADataset, LabelMap, load_official_test, load_raw_csv, split_train_val
from .evaluation import compute_metrics, evaluate, per_class_f1
from .models import AspectAttention, BertAspectAttention
from .preprocessing import ABSAEncoder
from .trainer import Trainer, resolve_device, set_seed

__version__ = "0.2.0"

__all__ = [
    "ABSADataset",
    "ABSAEncoder",
    "AspectAttention",
    "BertAspectAttention",
    "LabelMap",
    "Trainer",
    "__version__",
    "artifact_dir",
    "build_artifact",
    "compute_metrics",
    "evaluate",
    "get",
    "load_artifact",
    "load_config",
    "load_official_test",
    "load_raw_csv",
    "per_class_f1",
    "predict_pairs",
    "read_metrics",
    "resolve_device",
    "resolve_path",
    "set_seed",
    "split_train_val",
    "validate_config",
]
