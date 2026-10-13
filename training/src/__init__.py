"""Training package for aspect-based sentiment analysis on the SemEval laptop corpus."""

from .artifact import (
    ArtifactMetadata,
    LoadedArtifact,
    load_artifact,
    save_artifact,
)
from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from .data import ABSADataset, LabelMap, load_official_test, load_raw_csv, split_train_val
from .experiment import Experiment, configure_logging, run_experiment
from .models import ABSAModel
from .preprocessing import ABSAEncoder
from .trainer import ABSATrainer, TrainingOutcome, resolve_device, set_seed

__version__ = "0.1.0"

__all__ = [
    "ABSADataset",
    "ABSAEncoder",
    "ABSAModel",
    "ABSATrainer",
    "ArtifactMetadata",
    "DataConfig",
    "Experiment",
    "ExperimentConfig",
    "LabelMap",
    "LoadedArtifact",
    "ModelConfig",
    "TrainConfig",
    "TrainingOutcome",
    "__version__",
    "configure_logging",
    "load_artifact",
    "load_official_test",
    "load_raw_csv",
    "resolve_device",
    "run_experiment",
    "save_artifact",
    "set_seed",
    "split_train_val",
]
