"""Tests for ABSAEncoder and ABSADataset. Needs a local BERT tokenizer."""

from __future__ import annotations

import pytest
import torch

from src.data.dataset import ABSADataset, LabelMap, collate_batch

MODEL_NAME = "bert-base-uncased"


@pytest.fixture(scope="module")
def encoder():
    from src.preprocessing.tokenizer import ABSAEncoder

    try:
        return ABSAEncoder(MODEL_NAME, max_length=32)
    except Exception as exc:  # offline / no cached tokenizer
        pytest.skip(f"tokenizer {MODEL_NAME} unavailable: {exc}")


@pytest.fixture
def example(encoder):
    return encoder.encode("The screen is bright and crisp", "screen", label=2)


# ---- shapes and keys -------------------------------------------------- #


def test_encode_returns_expected_keys(example):
    assert set(example) == {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "context_mask",
        "aspect_mask",
        "labels",
    }


def test_encode_pads_every_tensor_to_max_length(encoder, example):
    for key in ("input_ids", "attention_mask", "token_type_ids", "context_mask", "aspect_mask"):
        assert example[key].shape == (encoder.max_length,), key


def test_masks_are_boolean(example):
    assert example["context_mask"].dtype is torch.bool
    assert example["aspect_mask"].dtype is torch.bool


def test_label_omitted_when_not_supplied(encoder):
    assert "labels" not in encoder.encode("The screen is bright", "screen")


def test_label_is_a_long_scalar(example):
    assert example["labels"].dtype is torch.long
    assert example["labels"].ndim == 0
    assert int(example["labels"]) == 2


# ---- input format ----------------------------------------------------- #


def test_input_is_cls_text_sep_aspect_sep(encoder, example):
    decoded = encoder.decode(example["input_ids"])
    head = decoded.split("[PAD]")[0].strip()
    assert head.startswith("[CLS]")
    assert head.endswith("[SEP]")
    assert head.count("[SEP]") == 2
    text_part, aspect_part = head[len("[CLS]") :].split("[SEP]")[:2]
    assert "screen is bright" in text_part
    assert aspect_part.strip() == "screen"


def test_token_type_ids_separate_the_two_segments(example):
    types = example["token_type_ids"]
    assert int(types[0]) == 0  # [CLS] belongs to the review
    assert set(types.tolist()) <= {0, 1}


# ---- mask semantics --------------------------------------------------- #


def test_masks_are_disjoint_and_exclude_specials_and_padding(encoder, example):
    context, aspect = example["context_mask"], example["aspect_mask"]
    assert not bool((context & aspect).any()), "masks must not overlap"

    # Neither mask may select [CLS], [SEP] or [PAD].
    special_ids = set(encoder.tokenizer.all_special_ids)
    selected = example["input_ids"][context | aspect].tolist()
    assert not (set(selected) & special_ids)

    padding = ~example["attention_mask"].bool()
    assert not bool((context & padding).any())
    assert not bool((aspect & padding).any())


def test_aspect_mask_covers_exactly_the_aspect_tokens(encoder):
    example = encoder.encode("The battery life is great", "battery life")
    aspect_ids = example["input_ids"][example["aspect_mask"]].tolist()
    assert encoder.tokenizer.convert_ids_to_tokens(aspect_ids) == ["battery", "life"]


def test_context_mask_covers_the_review_tokens(encoder):
    text = "The screen is bright"
    example = encoder.encode(text, "screen")
    context_ids = example["input_ids"][example["context_mask"]].tolist()
    assert encoder.tokenizer.convert_ids_to_tokens(context_ids) == [
        "the",
        "screen",
        "is",
        "bright",
    ]


def test_multiword_aspect_mask_is_longer_than_single_word(encoder):
    one = encoder.encode("The screen is nice", "screen")["aspect_mask"].sum()
    two = encoder.encode("The battery life is nice", "battery life")["aspect_mask"].sum()
    assert int(two) > int(one)


# ---- truncation and errors -------------------------------------------- #


def test_long_review_is_truncated_but_aspect_survives(encoder):
    long_text = "the laptop is good " * 200 + "screen"
    example = encoder.encode(long_text, "screen")
    assert example["input_ids"].shape == (encoder.max_length,)
    assert int(example["aspect_mask"].sum()) == 1  # "screen" kept intact
    assert bool(example["context_mask"].any())


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_aspect_raises(encoder, bad):
    with pytest.raises(ValueError, match="aspect produced zero tokens"):
        encoder.encode("The screen is bright", bad)


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_text_raises(encoder, bad):
    with pytest.raises(ValueError, match="text produced zero tokens"):
        encoder.encode(bad, "screen")


def test_tiny_max_length_is_rejected():
    from src.preprocessing.tokenizer import ABSAEncoder

    with pytest.raises(ValueError, match="max_length"):
        ABSAEncoder(MODEL_NAME, max_length=2)


# ---- ABSADataset ------------------------------------------------------ #


def test_dataset_encodes_every_row(toy_frame, encoder):
    dataset = ABSADataset(toy_frame, encoder, LabelMap())
    assert len(dataset) == len(toy_frame)


def test_dataset_item_is_a_tensor_dict(toy_frame, encoder):
    item = ABSADataset(toy_frame, encoder, LabelMap())[0]
    assert all(isinstance(v, torch.Tensor) for v in item.values())
    assert "labels" in item


def test_dataset_labels_match_the_frame(toy_frame, encoder):
    label_map = LabelMap()
    dataset = ABSADataset(toy_frame, encoder, label_map)
    assert dataset.labels == [label_map.encode(l) for l in toy_frame["label"]]


def test_dataset_without_labels(toy_frame, encoder):
    dataset = ABSADataset(toy_frame.drop(columns=["label"]), encoder, LabelMap())
    assert not dataset.has_labels
    assert "labels" not in dataset[0]
    assert dataset.labels == []


def test_dataset_class_weights_favour_rare_classes(toy_frame, encoder):
    dataset = ABSADataset(toy_frame, encoder, LabelMap())
    weights = dataset.class_weights()
    assert weights.shape == (3,)
    assert torch.all(weights > 0)
    # toy frame: 3 positive, 1 neutral, 1 negative -> positive weighted lowest
    assert float(weights[2]) < float(weights[1])


def test_collate_stacks_a_batch(toy_frame, encoder):
    dataset = ABSADataset(toy_frame, encoder, LabelMap())
    batch = collate_batch([dataset[i] for i in range(3)])
    assert batch["input_ids"].shape == (3, encoder.max_length)
    assert batch["labels"].shape == (3,)
