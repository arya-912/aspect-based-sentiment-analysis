"""Tests for loading, cleaning and splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.dataset import (
    LabelMap,
    load_official_test,
    load_raw_csv,
    split_train_val,
)


# ---- LabelMap --------------------------------------------------------- #


def test_label_map_ids_follow_declared_order():
    label_map = LabelMap()
    assert label_map.labels == ("negative", "neutral", "positive")
    assert label_map.label2id == {"negative": 0, "neutral": 1, "positive": 2}
    assert label_map.id2label == {0: "negative", 1: "neutral", 2: "positive"}
    assert label_map.num_labels == 3 == len(label_map)


def test_label_map_encode_and_decode_round_trip():
    label_map = LabelMap()
    for label in label_map.labels:
        assert label_map.decode(label_map.encode(label)) == label


def test_label_map_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown label"):
        LabelMap().encode("conflict")


def test_label_map_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate labels"):
        LabelMap(labels=("positive", "positive"))


# ---- load_raw_csv ----------------------------------------------------- #


def test_load_raw_csv_renames_columns(toy_frame: pd.DataFrame):
    assert list(toy_frame.columns) == ["review_id", "text", "aspect", "label", "from", "to"]


def test_load_raw_csv_drops_conflict_rows(toy_frame: pd.DataFrame):
    assert "conflict" not in set(toy_frame["label"])
    assert len(toy_frame) == 5  # 6 rows minus the one conflict


def test_load_raw_csv_offsets_are_integers_and_consistent(toy_frame: pd.DataFrame):
    assert toy_frame["from"].dtype.kind == "i"
    assert toy_frame["to"].dtype.kind == "i"
    for row in toy_frame.itertuples(index=False):
        start, end = int(row[4]), int(row[5])
        assert row.text[start:end] == row.aspect


def test_load_raw_csv_rejects_bad_offsets(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "id,Sentence,Aspect Term,polarity,from,to\n"
        "1,The screen is bright,screen,positive,0,6\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"text\[from:to\] != aspect"):
        load_raw_csv(path)


def test_load_raw_csv_rejects_unexpected_polarity(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "id,Sentence,Aspect Term,polarity,from,to\n"
        "1,The screen is bright,screen,amazing,4,10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected polarities"):
        load_raw_csv(path)


def test_load_raw_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_csv(tmp_path / "nope.csv")


def test_load_raw_csv_requires_offsets(tmp_path):
    """PhaseA/PhaseB carry only id+Sentence and must be rejected here."""
    path = tmp_path / "phase.csv"
    path.write_text("id,Sentence\n1,The screen is bright\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_raw_csv(path)


def test_load_raw_csv_realigns_column_shifted_rows(tmp_path):
    """Some real training rows carry a stray empty field after Sentence."""
    path = tmp_path / "shifted.csv"
    path.write_text(
        "id,Sentence,Aspect Term,polarity,from,to,\n"
        "1,The screen is bright,screen,positive,4,10,\n"
        "2,The keyboard is nice,,keyboard,positive,4,12\n",
        encoding="utf-8",
    )
    df = load_raw_csv(path)
    assert len(df) == 2
    repaired = df.loc[df["review_id"] == "2"].iloc[0]
    assert repaired["aspect"] == "keyboard"
    assert repaired["label"] == "positive"
    assert repaired["text"][int(repaired["from"]) : int(repaired["to"])] == "keyboard"


# ---- real corpus ------------------------------------------------------ #


def test_real_training_corpus_loads_clean(train_csv):
    df = load_raw_csv(train_csv)
    assert len(df) > 2000
    assert set(df["label"]) == {"negative", "neutral", "positive"}
    for row in df.itertuples(index=False):
        assert row.text[int(row[4]) : int(row[5])] == row.aspect


def test_real_gold_corpus_loads_clean(gold_csv):
    df = load_official_test(gold_csv)
    assert len(df) > 600
    assert set(df["label"]) <= {"negative", "neutral", "positive"}


# ---- split_train_val -------------------------------------------------- #


def test_split_keeps_sentences_whole(train_csv):
    df = load_raw_csv(train_csv)
    train_df, val_df = split_train_val(df, val_fraction=0.1, seed=42)

    assert len(train_df) + len(val_df) == len(df)
    assert not set(train_df["text"]) & set(val_df["text"])
    assert 0.05 < len(val_df) / len(df) < 0.16


def test_split_is_deterministic_for_a_seed(train_csv):
    df = load_raw_csv(train_csv)
    a_train, a_val = split_train_val(df, seed=7)
    b_train, b_val = split_train_val(df, seed=7)
    assert a_val["text"].tolist() == b_val["text"].tolist()
    assert a_train["text"].tolist() == b_train["text"].tolist()


def test_split_differs_across_seeds(train_csv):
    df = load_raw_csv(train_csv)
    _, val_a = split_train_val(df, seed=1)
    _, val_b = split_train_val(df, seed=2)
    assert set(val_a["text"]) != set(val_b["text"])


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_out_of_range_fractions(toy_frame, fraction):
    with pytest.raises(ValueError, match="val_fraction"):
        split_train_val(toy_frame, val_fraction=fraction)


def test_split_rejects_empty_frame(toy_frame):
    with pytest.raises(ValueError, match="empty"):
        split_train_val(toy_frame.iloc[0:0])
