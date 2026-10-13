"""Tests for BertAspectAttention and AspectAttention, all CPU, all offline."""

from __future__ import annotations

import pytest
import torch

from src.evaluation.evaluate import compute_metrics, evaluate, per_class_f1
from src.models.absa_model import AspectAttention, BertAspectAttention, masked_mean

LABELS = ("negative", "neutral", "positive")


# ---- masked_mean ------------------------------------------------------ #


def test_masked_mean_averages_only_selected_positions():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.allclose(masked_mean(hidden, mask), torch.tensor([[2.0, 2.0]]))


def test_masked_mean_returns_zeros_for_an_empty_mask():
    pooled = masked_mean(torch.randn(2, 4, 3), torch.zeros(2, 4, dtype=torch.bool))
    assert torch.allclose(pooled, torch.zeros_like(pooled))
    assert not torch.isnan(pooled).any()


# ---- AspectAttention -------------------------------------------------- #


def test_attention_weights_sum_to_one_over_the_context():
    attention = AspectAttention(hidden_size=8)
    hidden, aspect = torch.randn(3, 10, 8), torch.randn(3, 8)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[:, 1:6] = True

    context, weights = attention(hidden, aspect, mask)
    assert context.shape == (3, 8)
    assert weights.shape == (3, 10)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_attention_is_exactly_zero_outside_the_context_mask():
    attention = AspectAttention(hidden_size=8)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 2:5] = True
    _, weights = attention(torch.randn(2, 10, 8), torch.randn(2, 8), mask)
    assert float(weights[~mask].detach().abs().max()) == 0.0


def test_attention_survives_a_row_with_no_context_tokens():
    """A degenerate row must not produce NaN and poison the batch."""
    attention = AspectAttention(hidden_size=8)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[1, 1:4] = True  # row 0 has no context at all

    context, weights = attention(torch.randn(2, 10, 8), torch.randn(2, 8), mask)
    assert not torch.isnan(context).any()
    assert not torch.isnan(weights).any()
    assert float(weights[0].detach().sum()) == 0.0
    assert torch.allclose(weights[1].sum(), torch.tensor(1.0), atol=1e-5)


def test_attention_responds_to_the_aspect_vector():
    """Different aspects must produce different attention over the same text."""
    torch.manual_seed(0)
    attention = AspectAttention(hidden_size=8)
    hidden = torch.randn(1, 10, 8)
    mask = torch.ones(1, 10, dtype=torch.bool)

    _, w1 = attention(hidden, torch.randn(1, 8), mask)
    _, w2 = attention(hidden, torch.randn(1, 8) * 5, mask)
    assert not torch.allclose(w1, w2, atol=1e-4)


# ---- construction ----------------------------------------------------- #


def test_model_requires_a_name_or_an_encoder():
    with pytest.raises(ValueError, match="model_name or a prebuilt encoder"):
        BertAspectAttention()


def test_model_rejects_too_few_labels(tiny_backbone):
    with pytest.raises(ValueError, match="num_labels"):
        BertAspectAttention(encoder=tiny_backbone(), num_labels=1)


def test_model_rejects_mismatched_class_weights(tiny_backbone):
    with pytest.raises(ValueError, match="class_weights"):
        BertAspectAttention(
            encoder=tiny_backbone(), num_labels=3, class_weights=torch.ones(5)
        )


def test_classifier_consumes_three_concatenated_vectors(tiny_model):
    """The head sees [c ; a ; h_CLS], so its input is 3 x hidden_size."""
    hidden_size = tiny_model.encoder.config.hidden_size
    first_linear = next(m for m in tiny_model.classifier if isinstance(m, torch.nn.Linear))
    assert first_linear.in_features == 3 * hidden_size
    assert first_linear.out_features == hidden_size

    last_linear = [m for m in tiny_model.classifier if isinstance(m, torch.nn.Linear)][-1]
    assert last_linear.out_features == 3


def test_classifier_has_gelu_and_two_dropouts(tiny_model):
    kinds = [type(m).__name__ for m in tiny_model.classifier]
    assert kinds == ["Dropout", "Linear", "GELU", "Dropout", "Linear"]


# ---- forward ---------------------------------------------------------- #


def test_forward_returns_logits_and_attention(tiny_model, random_batch):
    batch = random_batch(with_labels=False)
    output = tiny_model(**batch)
    assert set(output) == {"logits", "attention"}
    assert output["logits"].shape == (4, 3)
    assert output["attention"].shape == (4, 16)


def test_forward_adds_loss_when_labels_are_passed(tiny_model, random_batch):
    output = tiny_model(**random_batch())
    assert set(output) == {"logits", "attention", "loss"}
    assert output["loss"].ndim == 0
    assert float(output["loss"].detach()) > 0


def test_forward_attention_is_masked_to_the_context(tiny_model, random_batch):
    batch = random_batch()
    weights = tiny_model(**batch)["attention"]
    assert float(weights[~batch["context_mask"]].detach().abs().max()) == 0.0
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_backward_pass_reaches_attention_and_head(tiny_model, random_batch):
    tiny_model(**random_batch())["loss"].backward()
    for name in ("attention.W_h.weight", "attention.W_a.weight", "attention.v.weight"):
        param = dict(tiny_model.named_parameters())[name]
        assert param.grad is not None, name
        assert float(param.grad.abs().sum()) > 0, name


def test_backward_pass_reaches_the_encoder(tiny_model, random_batch):
    tiny_model(**random_batch())["loss"].backward()
    embedding = tiny_model.encoder.embeddings.word_embeddings.weight
    assert embedding.grad is not None
    assert float(embedding.grad.abs().sum()) > 0


def test_forward_works_without_segment_masks(tiny_model, random_batch):
    """Masks are optional: the model falls back to the attention mask."""
    batch = random_batch(with_labels=False)
    del batch["aspect_mask"], batch["context_mask"]
    assert tiny_model(**batch)["logits"].shape == (4, 3)


def test_forward_is_deterministic_in_eval_mode(tiny_model, random_batch):
    batch = random_batch(with_labels=False)
    tiny_model.eval()
    with torch.no_grad():
        assert torch.allclose(tiny_model(**batch)["logits"], tiny_model(**batch)["logits"])


def test_variable_batch_and_sequence_shapes(tiny_model, random_batch):
    for batch_size, seq_len in ((1, 8), (3, 24), (8, 12)):
        output = tiny_model(**random_batch(batch_size, seq_len, with_labels=False))
        assert output["logits"].shape == (batch_size, 3)
        assert output["attention"].shape == (batch_size, seq_len)


def test_class_weights_change_the_loss(tiny_backbone, random_batch):
    batch = random_batch()
    torch.manual_seed(0)
    plain = BertAspectAttention(encoder=tiny_backbone(), num_labels=3)
    torch.manual_seed(0)
    weighted = BertAspectAttention(
        encoder=tiny_backbone(), num_labels=3, class_weights=torch.tensor([5.0, 0.2, 1.0])
    )
    plain.eval(), weighted.eval()
    with torch.no_grad():
        assert not torch.allclose(plain(**batch)["loss"], weighted(**batch)["loss"])


def test_class_weights_stay_out_of_the_state_dict(tiny_backbone):
    """They describe the data split, not the model, so a reload must not need them."""
    model = BertAspectAttention(
        encoder=tiny_backbone(), num_labels=3, class_weights=torch.tensor([1.0, 2.0, 0.5])
    )
    assert not any("class_weights" in key for key in model.state_dict())


def test_no_loss_state_leaks_into_the_state_dict(tiny_backbone):
    """Regression: an assigned nn.CrossEntropyLoss would register as a
    submodule and put `loss_fn.weight` in the state dict, so a weighted model
    could not be reloaded strictly into an unweighted one."""
    weighted = BertAspectAttention(
        encoder=tiny_backbone(), num_labels=3, class_weights=torch.tensor([1.0, 2.0, 0.5])
    )
    assert not any(key.startswith("loss_fn") for key in weighted.state_dict())

    plain = BertAspectAttention(encoder=tiny_backbone(), num_labels=3)
    assert set(weighted.state_dict()) == set(plain.state_dict())
    # The strict load an artifact reload performs.
    plain.load_state_dict(weighted.state_dict(), strict=True)


def test_loss_fn_tracks_the_current_class_weights(tiny_model):
    """Built on access, so it can never hold a stale tensor or stale device."""
    assert tiny_model.loss_fn.weight is None
    tiny_model.set_class_weights(torch.tensor([1.0, 2.0, 0.5]))
    assert torch.allclose(tiny_model.loss_fn.weight, torch.tensor([1.0, 2.0, 0.5]))


def test_set_class_weights_rebuilds_the_loss(tiny_model):
    assert tiny_model.loss_fn.weight is None
    tiny_model.set_class_weights(torch.tensor([1.0, 2.0, 0.5]))
    assert tiny_model.loss_fn.weight is not None
    tiny_model.set_class_weights(None)
    assert tiny_model.loss_fn.weight is None


def test_predict_returns_class_ids(tiny_model, random_batch):
    predictions = tiny_model.predict(random_batch())
    assert predictions.shape == (4,)
    assert set(predictions.tolist()) <= {0, 1, 2}


# ---- optimizer groups ------------------------------------------------- #


def test_parameter_groups_split_decay_and_learning_rates(tiny_model):
    groups = tiny_model.parameter_groups(lr=2e-5, weight_decay=0.01, head_lr=1e-4)
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {
        "encoder_decay", "encoder_no_decay", "head_decay", "head_no_decay"
    }
    assert by_name["encoder_decay"]["weight_decay"] == 0.01
    assert by_name["encoder_no_decay"]["weight_decay"] == 0.0
    assert by_name["encoder_decay"]["lr"] == 2e-5
    assert by_name["head_decay"]["lr"] == 1e-4


def test_parameter_groups_exclude_the_unused_pooler(tiny_model):
    """The model pools last_hidden_state itself, so no grad reaches the pooler."""
    pooler = {id(p) for n, p in tiny_model.encoder.named_parameters() if n.startswith("pooler.")}
    assert pooler, "the tiny BERT should have a pooler to exclude"
    grouped = {id(p) for g in tiny_model.parameter_groups(2e-5, 0.01) for p in g["params"]}
    assert not (pooler & grouped)


def test_parameter_groups_cover_every_other_trainable_parameter(tiny_model):
    grouped = {id(p) for g in tiny_model.parameter_groups(2e-5, 0.01) for p in g["params"]}
    expected = {
        id(p)
        for name, p in tiny_model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.pooler.")
    }
    assert grouped == expected


# ---- evaluate over a real DataLoader ---------------------------------- #


def test_evaluate_scores_a_dataloader(tiny_model, random_batch):
    from src.data.dataset import LabelMap

    batches = [random_batch(batch_size=4, seq_len=16) for _ in range(3)]
    metrics = evaluate(tiny_model, batches, "cpu", LabelMap(), split="unit")

    assert metrics["n"] == 12
    assert metrics["evaluation_split"] == "unit"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert set(metrics["per_class"]) == set(LABELS)
    assert len(metrics["confusion_matrix"]) == 3
    assert metrics["loss"] > 0


def test_evaluate_rejects_unlabelled_batches(tiny_model, random_batch):
    batches = [random_batch(with_labels=False)]
    with pytest.raises(ValueError, match="needs labelled batches"):
        evaluate(tiny_model, batches, "cpu")


# ---- compute_metrics -------------------------------------------------- #


def test_perfect_predictions_score_one():
    metrics = compute_metrics([0, 1, 2], [0, 1, 2], LABELS)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["n"] == 3


def test_metrics_report_every_class_even_when_absent():
    metrics = compute_metrics([0, 0], [0, 0], LABELS)
    assert set(metrics["per_class"]) == set(LABELS)
    assert metrics["per_class"]["positive"]["support"] == 0


def test_confusion_matrix_is_square_and_totals_the_examples():
    matrix = compute_metrics([0, 1, 2, 0], [0, 2, 2, 1], LABELS)["confusion_matrix"]
    assert len(matrix) == len(matrix[0]) == 3
    assert sum(sum(row) for row in matrix) == 4


def test_metrics_include_the_required_keys():
    metrics = compute_metrics([0, 1], [0, 1], LABELS, loss=0.3, split="val")
    for key in ("accuracy", "macro_f1", "per_class", "confusion_matrix", "n", "evaluation_split"):
        assert key in metrics, key
    assert metrics["evaluation_split"] == "val"


@pytest.mark.parametrize(
    "y_true,y_pred,message",
    [([0, 1], [0], "length mismatch"), ([], [], "zero examples")],
)
def test_metrics_reject_bad_input(y_true, y_pred, message):
    with pytest.raises(ValueError, match=message):
        compute_metrics(y_true, y_pred, LABELS)


def test_per_class_f1_extracts_one_score_per_label():
    scores = per_class_f1(compute_metrics([0, 1, 2], [0, 1, 1], LABELS))
    assert set(scores) == set(LABELS)
    assert all(isinstance(v, float) for v in scores.values())
