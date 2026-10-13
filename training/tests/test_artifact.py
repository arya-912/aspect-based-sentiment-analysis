"""Tests for artifact packaging, offline reload, and the config loader."""

from __future__ import annotations

import json

import pytest
import torch

from src.artifact import (
    CONFIG_FILE,
    ENCODER_DIR,
    LABEL_MAP_FILE,
    METRICS_FILE,
    MODEL_FILE,
    TOKENIZER_DIR,
    build_artifact,
    load_artifact,
    predict_pairs,
    read_metrics,
)
from src.config import coerce_numeric, get, load_config, set_in, validate_config
from src.data.dataset import LabelMap
from src.evaluation.evaluate import compute_metrics
from src.models.absa_model import BertAspectAttention

LABELS = ("negative", "neutral", "positive")


@pytest.fixture
def artifact_cfg(tmp_path):
    """A config whose artifacts_dir points into tmp_path."""
    return {
        "seed": 42,
        "model": {
            "name": "tiny-bert-test",
            "max_length": 24,
            "dropout": 0.1,
            "num_labels": 3,
        },
        "training": {"epochs": 1, "batch_size": 4, "lr": 3e-5, "val_fraction": 0.1},
        "paths": {
            "raw_train": "training/data/raw/Laptop_Train_v2.csv",
            "official_test": "training/data/raw/Laptops_Test_Gold.csv",
            "processed_dir": "training/data/processed/splits",
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
        "artifact": {"version": "absa-test"},
        "data": {"labels": list(LABELS)},
    }


@pytest.fixture
def metrics():
    return compute_metrics([0, 1, 2, 2], [0, 1, 2, 1], LABELS, loss=0.42, split="official_test")


@pytest.fixture
def built(artifact_cfg, tiny_model, tiny_encoder, metrics):
    """A written artifact, plus the pieces that went into it."""
    directory = build_artifact(
        cfg=artifact_cfg,
        run_id="run-123",
        model=tiny_model,
        encoder=tiny_encoder,
        label_map=LabelMap(LABELS),
        test_metrics=metrics,
        best_epoch=2,
        history=[{"epoch": 1, "val_macro_f1": 0.5}, {"epoch": 2, "val_macro_f1": 0.7}],
    )
    return directory, tiny_model, tiny_encoder


# ---- build_artifact --------------------------------------------------- #


def test_build_artifact_writes_every_required_entry(built):
    directory, _, _ = built
    for name in (MODEL_FILE, CONFIG_FILE, LABEL_MAP_FILE, METRICS_FILE):
        assert (directory / name).is_file(), name
    for name in (ENCODER_DIR, TOKENIZER_DIR):
        assert (directory / name).is_dir(), name


def test_build_artifact_uses_the_configured_version(built):
    directory, _, _ = built
    assert directory.name == "absa-test"


def test_encoder_dir_holds_the_backbone_config_not_weights(built):
    """Weights live once, in model.pt; encoder/ only needs to rebuild the shape."""
    directory, _, _ = built
    assert (directory / ENCODER_DIR / "config.json").is_file()
    assert not list((directory / ENCODER_DIR).glob("*.bin"))
    assert not list((directory / ENCODER_DIR).glob("*.safetensors"))


def test_model_file_holds_a_plain_state_dict(built):
    directory, model, _ = built
    state = torch.load(directory / MODEL_FILE, map_location="cpu", weights_only=True)
    assert set(state) == set(model.state_dict())
    assert all(isinstance(v, torch.Tensor) for v in state.values())


def test_config_json_records_hyperparams_and_provenance(built):
    directory, _, _ = built
    config = json.loads((directory / CONFIG_FILE).read_text())
    assert config["model"]["name"] == "tiny-bert-test"
    assert config["training"]["epochs"] == 1
    assert config["run_id"] == "run-123"
    assert config["version"] == "absa-test"
    assert config["best_epoch"] == 2
    assert "created_at" in config
    assert "torch" in config["library_versions"]
    assert len(config["history"]) == 2


def test_label_map_json_round_trips_both_directions(built):
    directory, _, _ = built
    payload = json.loads((directory / LABEL_MAP_FILE).read_text())
    assert payload["labels"] == list(LABELS)
    assert payload["label2id"] == {"negative": 0, "neutral": 1, "positive": 2}
    assert payload["id2label"] == {"0": "negative", "1": "neutral", "2": "positive"}


def test_metrics_json_holds_the_evaluation(built, metrics):
    directory, _, _ = built
    stored = json.loads((directory / METRICS_FILE).read_text())
    assert stored["accuracy"] == metrics["accuracy"]
    assert stored["macro_f1"] == metrics["macro_f1"]
    assert set(stored["per_class"]) == set(LABELS)
    assert stored["confusion_matrix"] == metrics["confusion_matrix"]


def test_read_metrics_reads_it_back(built, metrics):
    directory, _, _ = built
    assert read_metrics(directory)["macro_f1"] == metrics["macro_f1"]


def test_build_artifact_saves_an_explicit_state_dict(artifact_cfg, tiny_model, tiny_encoder, metrics):
    """The trainer hands over the *best* weights, not the model's current ones."""
    best = {k: torch.zeros_like(v) for k, v in tiny_model.state_dict().items()}
    directory = build_artifact(
        cfg=artifact_cfg, run_id="r", model=tiny_model, encoder=tiny_encoder,
        label_map=LabelMap(LABELS), test_metrics=metrics, best_epoch=1, state_dict=best,
    )
    saved = torch.load(directory / MODEL_FILE, map_location="cpu", weights_only=True)
    assert all(float(v.abs().sum()) == 0.0 for v in saved.values())


# ---- load_artifact ---------------------------------------------------- #


def test_weighted_model_round_trips(artifact_cfg, tiny_backbone, tiny_encoder, metrics):
    """End-to-end regression: a model trained *with* class weights must reload.

    This is the path real training takes -- the unweighted fixture above would
    not have caught `loss_fn.weight` leaking into the state dict.
    """
    torch.manual_seed(0)
    model = BertAspectAttention(
        model_name="tiny-bert-test",
        num_labels=3,
        encoder=tiny_backbone(),
        class_weights=torch.tensor([1.1, 1.3, 0.6]),
    )
    directory = build_artifact(
        cfg=artifact_cfg, run_id="weighted", model=model, encoder=tiny_encoder,
        label_map=LabelMap(LABELS), test_metrics=metrics, best_epoch=1,
    )
    reloaded, _, _, _ = load_artifact(directory, device="cpu")
    for key, value in model.state_dict().items():
        assert torch.allclose(reloaded.state_dict()[key], value), key


def test_load_artifact_round_trips(built):
    directory, original, _ = built
    model, encoder, label_map, config = load_artifact(directory, device="cpu")

    assert isinstance(model, BertAspectAttention)
    assert label_map.labels == LABELS
    assert config["run_id"] == "run-123"
    assert encoder.max_length == 24

    for key, value in original.state_dict().items():
        assert torch.allclose(model.state_dict()[key], value), key


def test_load_artifact_returns_a_model_in_eval_mode(built):
    directory, _, _ = built
    model, _, _, _ = load_artifact(directory, device="cpu")
    assert not model.training


def test_loaded_model_reproduces_the_original_logits(built, random_batch):
    """The reload must be exact, not merely the right shape."""
    directory, original, _ = built
    loaded, _, _, _ = load_artifact(directory, device="cpu")

    batch = random_batch(with_labels=False, seq_len=16)
    original.eval()
    with torch.no_grad():
        assert torch.allclose(original(**batch)["logits"], loaded(**batch)["logits"], atol=1e-6)


def test_load_artifact_needs_no_network(built, monkeypatch):
    """Offline reload: refuse any outbound HTTP during load_artifact."""
    directory, _, _ = built
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    import socket

    def no_network(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("load_artifact attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    model, encoder, label_map, _ = load_artifact(directory, device="cpu")
    assert model.num_labels == 3
    assert label_map.labels == LABELS


def test_loaded_encoder_still_encodes(built):
    directory, _, _ = built
    _, encoder, _, _ = load_artifact(directory, device="cpu")
    example = encoder.encode("the screen is bright", "screen")
    assert example["input_ids"].shape == (24,)
    assert bool(example["aspect_mask"].any())
    assert bool(example["context_mask"].any())


def test_predict_pairs_returns_labels_scores_and_attention(built):
    directory, _, _ = built
    model, encoder, label_map, _ = load_artifact(directory, device="cpu")

    results = predict_pairs(
        model, encoder, label_map,
        [("the screen is bright", "screen"), ("the keyboard is nice", "keyboard")],
    )
    assert len(results) == 2
    for result in results:
        assert result["label"] in LABELS
        assert 0.0 <= result["confidence"] <= 1.0
        assert set(result["scores"]) == set(LABELS)
        assert pytest.approx(sum(result["scores"].values()), abs=1e-5) == 1.0
        assert result["attention"], "attention should cover the context tokens"
        assert all(0.0 <= t["weight"] <= 1.0 for t in result["attention"])


def test_predict_pairs_on_no_input():
    assert predict_pairs(None, None, None, []) == []


# ---- incomplete artifacts --------------------------------------------- #


def test_load_artifact_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such artifact directory"):
        load_artifact(tmp_path / "nope")


def test_load_artifact_rejects_an_empty_directory(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError, match="incomplete artifact"):
        load_artifact(tmp_path / "run")


@pytest.mark.parametrize("entry", [MODEL_FILE, CONFIG_FILE, LABEL_MAP_FILE, ENCODER_DIR, TOKENIZER_DIR])
def test_load_artifact_rejects_each_missing_entry(built, entry):
    """Every required piece must be reported, not discovered as a later crash."""
    import shutil

    directory, _, _ = built
    target = directory / entry
    shutil.rmtree(target) if target.is_dir() else target.unlink()

    with pytest.raises(FileNotFoundError, match="incomplete artifact"):
        load_artifact(directory)


# ---- config loader ---------------------------------------------------- #


def test_default_config_loads_and_validates():
    cfg = load_config()
    assert get(cfg, "model.name") == "bert-base-uncased"
    assert get(cfg, "model.num_labels") == 3
    assert get(cfg, "artifact.version") == "absa-v1"
    assert get(cfg, "data.labels") == ["negative", "neutral", "positive"]


def test_default_config_coerces_scientific_notation():
    """`lr: 3e-5` is a YAML *string*; it must reach the optimizer as a float."""
    cfg = load_config()
    assert isinstance(get(cfg, "training.lr"), float)
    assert get(cfg, "training.lr") == pytest.approx(3e-5)


def test_coerce_numeric_rejects_nonsense():
    cfg = {"training": {"lr": "fast"}}
    with pytest.raises(ValueError, match="training.lr"):
        coerce_numeric(cfg)


def test_config_overrides_apply_by_dotted_key():
    cfg = load_config(overrides={"training.epochs": 1, "model.name": "distilbert-base-uncased"})
    assert get(cfg, "training.epochs") == 1
    assert get(cfg, "model.name") == "distilbert-base-uncased"


def test_get_returns_the_default_for_an_absent_key():
    assert get({"a": {"b": 1}}, "a.b") == 1
    assert get({"a": {"b": 1}}, "a.zz", "fallback") == "fallback"
    assert get({"a": {"b": 1}}, "x.y.z") is None


def test_set_in_creates_missing_sections():
    cfg: dict = {}
    set_in(cfg, "optim.head_lr", 1e-4)
    assert cfg == {"optim": {"head_lr": 1e-4}}


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_validate_config_reports_missing_keys():
    with pytest.raises(KeyError, match="missing required key"):
        validate_config({"seed": 1})


def test_validate_config_rejects_an_out_of_range_val_fraction():
    cfg = load_config()
    cfg["training"]["val_fraction"] = 1.5
    with pytest.raises(ValueError, match="val_fraction"):
        validate_config(cfg)


def test_validate_config_rejects_label_count_mismatch():
    cfg = load_config()
    cfg["data"]["labels"] = ["negative", "positive"]
    with pytest.raises(ValueError, match="num_labels"):
        validate_config(cfg)
