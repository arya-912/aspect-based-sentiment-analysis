"""Config, model, metric and artifact tests -- no pretrained download needed."""

from __future__ import annotations

import pytest
import torch

from src.config import ExperimentConfig
from src.data.dataset import LabelMap
from src.evaluation.evaluate import compute_metrics
from src.evaluation.reporting import format_markdown, format_result
from src.models.absa_model import masked_mean

LABELS = LabelMap().labels


# ---- config ----------------------------------------------------------- #


def test_default_config_has_three_labels():
    config = ExperimentConfig()
    assert config.data.labels == LABELS
    assert config.model.pooling == "aspect_context"


def test_config_round_trips_through_yaml(tmp_path):
    config = ExperimentConfig()
    reloaded = ExperimentConfig.load(config.save(tmp_path / "c.yaml"))
    assert reloaded.to_dict() == config.to_dict()


def test_config_nested_sections_are_dataclasses(tmp_path):
    path = ExperimentConfig().save(tmp_path / "c.yaml")
    reloaded = ExperimentConfig.load(path)
    assert isinstance(reloaded.train.epochs, int)
    assert reloaded.model.model_name == "bert-base-uncased"


def test_config_overrides_apply_by_dotted_key():
    config = ExperimentConfig().with_overrides(
        {"train.epochs": 1, "model.pooling": "cls", "name": "x"}
    )
    assert (config.train.epochs, config.model.pooling, config.name) == (1, "cls", "x")


def test_config_rejects_unknown_override():
    with pytest.raises(KeyError):
        ExperimentConfig().with_overrides({"train.nope": 1})


def test_config_rejects_unknown_yaml_key(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(KeyError, match="bogus"):
        ExperimentConfig.load(path)


def test_run_dir_sits_under_artifacts():
    config = ExperimentConfig(name="demo")
    assert config.run_dir.parts[-2:] == ("artifacts", "demo")


# ---- masked_mean ------------------------------------------------------ #


def test_masked_mean_averages_only_selected_positions():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.allclose(masked_mean(hidden, mask), torch.tensor([[2.0, 2.0]]))


def test_masked_mean_returns_zeros_for_an_empty_mask():
    hidden = torch.randn(2, 4, 3)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    pooled = masked_mean(hidden, mask)
    assert torch.allclose(pooled, torch.zeros_like(pooled))
    assert not torch.isnan(pooled).any()


# ---- metrics ---------------------------------------------------------- #


def test_perfect_predictions_score_one():
    result = compute_metrics([0, 1, 2], [0, 1, 2], LABELS)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0


def test_metrics_report_every_class_even_when_absent():
    result = compute_metrics([0, 0], [0, 0], LABELS)
    assert set(result.per_class) == set(LABELS)
    assert result.per_class["positive"]["support"] == 0


def test_confusion_matrix_is_square_and_totals_the_examples():
    result = compute_metrics([0, 1, 2, 0], [0, 2, 2, 1], LABELS)
    matrix = result.confusion_matrix
    assert len(matrix) == len(matrix[0]) == len(LABELS)
    assert sum(sum(row) for row in matrix) == 4


def test_metrics_reject_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        compute_metrics([0, 1], [0], LABELS)


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError, match="zero examples"):
        compute_metrics([], [], LABELS)


def test_summary_includes_loss_only_when_given():
    assert "loss" not in compute_metrics([0], [0], LABELS).summary()
    assert "loss" in compute_metrics([0], [0], LABELS, loss=0.5).summary()


# ---- reporting -------------------------------------------------------- #


def test_text_report_mentions_every_label():
    report = format_result(compute_metrics([0, 1, 2], [0, 1, 1], LABELS))
    for label in LABELS:
        assert label in report
    assert "macro F1" in report


def test_markdown_report_is_a_table():
    markdown = format_markdown(compute_metrics([0, 1, 2], [0, 1, 1], LABELS))
    assert markdown.startswith("## ")
    assert "| metric | value |" in markdown
    assert "Confusion matrix" in markdown


# ---- artifact metadata ------------------------------------------------ #


def test_artifact_metadata_round_trips():
    from src.artifact import ArtifactMetadata

    meta = ArtifactMetadata(
        name="run",
        model_name="bert-base-uncased",
        labels=LABELS,
        max_length=128,
        pooling="aspect_context",
        best_epoch=2,
        best_metric=0.71,
    )
    reloaded = ArtifactMetadata.from_dict(meta.to_dict())
    assert reloaded.labels == LABELS
    assert reloaded.best_epoch == 2
    assert reloaded.versions["torch"] == torch.__version__


def test_load_artifact_rejects_a_missing_directory(tmp_path):
    from src.artifact import load_artifact

    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "nope")


def test_load_artifact_rejects_an_incomplete_bundle(tmp_path):
    from src.artifact import load_artifact

    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        load_artifact(tmp_path / "run")
