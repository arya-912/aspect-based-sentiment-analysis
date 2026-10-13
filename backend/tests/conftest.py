"""Fixtures providing a fully mocked model, so the suite needs no artifact.

The real artifact is hundreds of megabytes and may not exist yet. These
doubles honour the same contracts the service depends on -- the encoder's
tensor dict, the model's ``{"logits": ...}`` output and the label map's
``id2label`` -- so the routes, schemas and error handling are exercised for
real while inference is deterministic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.model_registry import ModelRegistry  # noqa: E402

MAX_LENGTH = 128
LABELS = ("negative", "neutral", "positive")

#: Deterministic logits: positive (index 2) always wins.
FAKE_LOGITS = torch.tensor([[0.1, 0.2, 0.7]])


class FakeModel:
    """Stands in for ``BertAspectAttention``.

    Returns fixed logits so the predicted label and confidence are
    predictable, and echoes the key the real model emits alongside them.
    """

    def __init__(self, logits: torch.Tensor | None = None) -> None:
        self.logits = logits if logits is not None else FAKE_LOGITS
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        return {"logits": self.logits, "attention": torch.zeros(1, MAX_LENGTH)}

    def parameters(self):
        """The real model has parameters; this one has none on purpose, which
        exercises the predictor's CPU fallback."""
        return iter(())

    def eval(self) -> "FakeModel":
        return self


class FakeEncoder:
    """Stands in for ``ABSAEncoder``.

    Produces correctly shaped zero tensors for any input, and raises
    ``ValueError`` for blank input exactly as the real encoder does -- that is
    the path the predictor turns into a 400.
    """

    def __init__(self, max_length: int = MAX_LENGTH) -> None:
        self.max_length = max_length
        self.calls: list[tuple[str, str]] = []

    def encode(
        self, text: str, aspect: str, label: int | None = None
    ) -> dict[str, torch.Tensor]:
        if not str(text).strip():
            raise ValueError(f"text produced zero tokens: {text!r}")
        if not str(aspect).strip():
            raise ValueError(f"aspect produced zero tokens: {aspect!r}")
        self.calls.append((text, aspect))

        example = {
            "input_ids": torch.zeros(self.max_length, dtype=torch.long),
            "attention_mask": torch.ones(self.max_length, dtype=torch.long),
            "token_type_ids": torch.zeros(self.max_length, dtype=torch.long),
            "context_mask": torch.zeros(self.max_length, dtype=torch.bool),
            "aspect_mask": torch.zeros(self.max_length, dtype=torch.bool),
        }
        example["context_mask"][1:5] = True
        example["aspect_mask"][6:8] = True
        if label is not None:
            example["labels"] = torch.tensor(int(label), dtype=torch.long)
        return example


class FakeLabelMap:
    """Stands in for ``LabelMap``."""

    def __init__(self, labels: tuple[str, ...] = LABELS) -> None:
        self.labels = labels

    @property
    def label2id(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.labels))

    def decode(self, label_id: int) -> str:
        return self.labels[int(label_id)]


FAKE_METRICS: dict[str, Any] = {
    "accuracy": 0.7821,
    "macro_f1": 0.7447,
    "weighted_f1": 0.7821,
    "per_class": {
        "negative": {"precision": 0.64, "recall": 0.86, "f1": 0.73, "support": 128},
        "neutral": {"precision": 0.67, "recall": 0.57, "f1": 0.62, "support": 169},
        "positive": {"precision": 0.91, "recall": 0.86, "f1": 0.88, "support": 341},
    },
    "confusion_matrix": [[110, 13, 5], [47, 97, 25], [15, 34, 292]],
    "n": 638,
    "evaluation_split": "official_test",
}

FAKE_CONFIG: dict[str, Any] = {
    "version": "absa-v1-test",
    "run_id": "absa-v1-test",
    "best_epoch": 2,
    "model": {"name": "bert-base-uncased", "max_length": MAX_LENGTH, "num_labels": 3},
    "training": {"epochs": 4, "batch_size": 16, "lr": 3e-5},
    "data": {"labels": list(LABELS)},
}


@pytest.fixture
def fake_artifact_dir(tmp_path: Path) -> Path:
    """A directory holding just the JSON files ``/model`` reads."""
    directory = tmp_path / "absa-v1-test"
    directory.mkdir()
    (directory / "metrics.json").write_text(json.dumps(FAKE_METRICS), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps(FAKE_CONFIG), encoding="utf-8")
    (directory / "label_map.json").write_text(
        json.dumps({"labels": list(LABELS)}), encoding="utf-8"
    )
    return directory


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def mocked_registry(fake_model, fake_encoder, fake_artifact_dir, monkeypatch):
    """The shared registry, populated with doubles instead of a real artifact.

    Patches the singleton's private state directly and restores it afterwards,
    so no test leaks a loaded (or unloaded) model into the next one.
    """
    registry = ModelRegistry()
    saved = registry.__dict__.copy()

    registry._model = fake_model
    registry._encoder = fake_encoder
    registry._label_map = FakeLabelMap()
    registry._config = dict(FAKE_CONFIG)
    registry._artifact_dir = fake_artifact_dir
    registry._device = "cpu"
    registry._load_error = None

    # Startup must not try to load the real artifact over the doubles.
    monkeypatch.setattr(registry, "load", lambda *args, **kwargs: None)

    yield registry

    registry.__dict__.clear()
    registry.__dict__.update(saved)


@pytest.fixture
def unloaded_registry(monkeypatch):
    """The registry with nothing loaded, for the 503 paths."""
    registry = ModelRegistry()
    saved = registry.__dict__.copy()

    registry.reset()
    monkeypatch.setattr(registry, "load", lambda *args, **kwargs: None)

    yield registry

    registry.__dict__.clear()
    registry.__dict__.update(saved)


@pytest.fixture
def settings() -> Settings:
    return Settings(ARTIFACT_DIR="training/artifacts/absa-v1", DEVICE="cpu")


@pytest.fixture
def client(mocked_registry, settings) -> TestClient:
    """A TestClient over the app, with the mocked registry in place.

    Entering the context manager runs the lifespan, whose ``load`` is patched
    out by ``mocked_registry``.
    """
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def raising_client(mocked_registry, settings) -> TestClient:
    """A client that returns the 500 response rather than re-raising.

    ``TestClient`` re-raises unhandled server exceptions by default, which is
    useful for debugging but hides the response the ``Exception`` handler
    actually produces in production. This fixture asserts on that response.
    """
    with TestClient(create_app(settings), raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def degraded_client(unloaded_registry, settings) -> TestClient:
    """A TestClient over an app whose model never loaded."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client
