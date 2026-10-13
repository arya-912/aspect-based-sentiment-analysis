"""Loading, cleaning and splitting of the SemEval laptop ABSA data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

#: Polarity that SemEval uses for aspects carrying both sentiments. Dropped.
CONFLICT_LABEL = "conflict"

#: Raw column name -> internal column name.
COLUMN_RENAMES = {
    "Sentence": "text",
    "Aspect Term": "aspect",
    "polarity": "label",
    "id": "review_id",
}

OUTPUT_COLUMNS = ["review_id", "text", "aspect", "label", "from", "to"]


@dataclass(frozen=True)
class LabelMap:
    """Ordered polarity vocabulary; the position in ``labels`` is the class id."""

    labels: tuple[str, ...] = ("negative", "neutral", "positive")

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("LabelMap needs at least one label")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"duplicate labels in {self.labels!r}")

    @property
    def label2id(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.labels)}

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def encode(self, label: str) -> int:
        """Map a polarity string to its class id."""
        try:
            return self.label2id[label]
        except KeyError:
            raise ValueError(
                f"unknown label {label!r}; expected one of {self.labels}"
            ) from None

    def decode(self, label_id: int) -> str:
        """Map a class id back to its polarity string."""
        try:
            return self.id2label[int(label_id)]
        except KeyError:
            raise ValueError(
                f"label id {label_id!r} out of range for {self.labels}"
            ) from None

    def __len__(self) -> int:
        return len(self.labels)


def _realign_shifted_rows(
    df: pd.DataFrame, known_labels: Sequence[str], path: Path
) -> pd.DataFrame:
    """Repair rows whose fields are shifted one column to the right.

    ``Laptop_Train_v2.csv`` ships a handful of rows with a stray empty field
    directly after ``Sentence``, e.g.::

        3074,The screen is bright and the keyboard is nice,,screen,positive,4,10

    Those rows parse with an empty aspect, the aspect sitting in ``polarity``
    and so on. They are detected by a ``polarity`` value that is not a known
    label *and* a blank aspect, and fixed by pulling every field after
    ``Sentence`` one position left out of the trailing extra column.
    """
    valid = set(known_labels) | {CONFLICT_LABEL}

    # The shift leaves a blank aspect and pushes "to" into a trailing extra
    # column (pandas names it "Unnamed: 6"), so both must be present. A row
    # that is merely mislabelled is left alone for the polarity check in
    # load_raw_csv to report.
    spill = df.columns[-1]
    if spill in OUTPUT_COLUMNS:
        return df

    shifted = ~df["label"].isin(valid) & df["aspect"].astype(str).str.strip().eq("")
    if not shifted.any():
        return df

    if "from" not in df.columns or "to" not in df.columns:
        raise ValueError(f"{path}: cannot realign shifted rows without offsets")

    rows = df.loc[shifted]
    realigned = pd.DataFrame(
        {
            "review_id": rows["review_id"],
            "text": rows["text"],
            "aspect": rows["label"],
            "label": rows["from"],
            "from": rows["to"],
            "to": rows[spill],
        }
    )
    still_bad = ~realigned["label"].isin(valid)
    if still_bad.any():
        bad = realigned.loc[still_bad].head(3).to_dict("records")
        raise ValueError(
            f"{path}: {int(still_bad.sum())} row(s) have an unrecognised polarity "
            f"and could not be realigned, e.g. {bad}"
        )

    LOGGER.warning(
        "%s: realigned %d column-shifted row(s)", path.name, len(realigned)
    )
    df = df.copy()
    df.loc[shifted, ["aspect", "label", "from", "to"]] = realigned[
        ["aspect", "label", "from", "to"]
    ]
    return df


def _validate_offsets(df: pd.DataFrame, path: Path) -> None:
    """Assert ``text[from:to] == aspect`` for every row."""
    sliced = pd.Series(
        [
            text[int(start) : int(end)]
            for text, start, end in zip(df["text"], df["from"], df["to"])
        ],
        index=df.index,
    )
    mismatched = df.loc[sliced != df["aspect"]]
    if not mismatched.empty:
        examples = [
            {
                "review_id": rec["review_id"],
                "aspect": rec["aspect"],
                "from": rec["from"],
                "to": rec["to"],
                "sliced": sliced[idx],
            }
            for idx, rec in mismatched.head(3).to_dict("index").items()
        ]
        raise ValueError(
            f"{path}: {len(mismatched)} row(s) where text[from:to] != aspect, "
            f"e.g. {examples}"
        )


def load_raw_csv(
    path: str | Path, known_labels: Sequence[str] = LabelMap().labels
) -> pd.DataFrame:
    """Read one SemEval laptop CSV into a clean, validated frame.

    Args:
        path: CSV with ``id, Sentence, Aspect Term, polarity, from, to``.
        known_labels: the polarities to keep. ``conflict`` rows are always
            dropped; any other unexpected polarity is an error.

    Returns:
        Frame with columns ``review_id, text, aspect, label, from, to``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such dataset file: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.rename(columns=COLUMN_RENAMES)

    missing = [c for c in ("review_id", "text", "aspect", "label") if c not in df]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}; got {list(df.columns)}"
        )
    for offset_col in ("from", "to"):
        if offset_col not in df:
            raise ValueError(
                f"{path}: missing offset column {offset_col!r}; "
                "this file carries no aspect annotations"
            )

    n_raw = len(df)
    df = _realign_shifted_rows(df, known_labels, path)

    conflicts = int((df["label"] == CONFLICT_LABEL).sum())
    df = df.loc[df["label"] != CONFLICT_LABEL]

    unexpected = sorted(set(df["label"]) - set(known_labels))
    if unexpected:
        raise ValueError(f"{path}: unexpected polarities {unexpected}")

    df = df.assign(
        from_=pd.to_numeric(df["from"], errors="raise").astype(int),
        to_=pd.to_numeric(df["to"], errors="raise").astype(int),
    ).drop(columns=["from", "to"]).rename(columns={"from_": "from", "to_": "to"})

    _validate_offsets(df, path)

    blank = df["aspect"].str.strip().eq("") | df["text"].str.strip().eq("")
    if blank.any():
        raise ValueError(f"{path}: {int(blank.sum())} row(s) with empty text/aspect")

    df = df[OUTPUT_COLUMNS].reset_index(drop=True)
    LOGGER.info(
        "%s: %d rows -> %d usable (dropped %d conflict), %d unique sentences",
        path.name,
        n_raw,
        len(df),
        conflicts,
        df["text"].nunique(),
    )
    return df


def split_train_val(
    df: pd.DataFrame, val_fraction: float = 0.1, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by sentence so no sentence's aspects straddle the two splits.

    Sentences are shuffled deterministically and assigned to validation until
    the target row fraction is reached, which keeps multi-aspect sentences
    whole and prevents the context leaking across the split.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if df.empty:
        raise ValueError("cannot split an empty frame")

    groups = df.groupby("text", sort=True).size()
    order = groups.sample(frac=1.0, random_state=seed).index.to_list()

    target = len(df) * val_fraction
    val_texts: set[str] = set()
    taken = 0
    for text in order:
        if taken >= target:
            break
        val_texts.add(text)
        taken += int(groups[text])

    in_val = df["text"].isin(val_texts)
    train_df = df.loc[~in_val].reset_index(drop=True)
    val_df = df.loc[in_val].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"val_fraction={val_fraction} produced an empty split "
            f"({len(train_df)} train / {len(val_df)} val)"
        )
    assert not set(train_df["text"]) & set(val_df["text"]), "sentence leaked across split"

    LOGGER.info(
        "split: %d train / %d val rows (%d / %d sentences)",
        len(train_df),
        len(val_df),
        train_df["text"].nunique(),
        val_df["text"].nunique(),
    )
    return train_df, val_df


def load_official_test(
    path: str | Path, known_labels: Sequence[str] = LabelMap().labels
) -> pd.DataFrame:
    """Load the held-out gold test file with the same cleaning, unsplit."""
    return load_raw_csv(path, known_labels)


def label_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Row count per polarity, handy for logs and reports."""
    return df["label"].value_counts().sort_index().to_dict()


class ABSADataset(Dataset):
    """Tensorised (sentence, aspect) pairs, encoded once up front.

    The corpus is small enough (a few thousand pairs) that eager encoding
    costs little memory and keeps the training loop free of tokenizer work.
    """

    def __init__(self, df: pd.DataFrame, encoder, label_map: LabelMap) -> None:
        self.label_map = label_map
        self.frame = df.reset_index(drop=True)
        self.has_labels = "label" in self.frame.columns

        self.examples: list[dict[str, torch.Tensor]] = []
        for row in self.frame.itertuples(index=False):
            label = label_map.encode(row.label) if self.has_labels else None
            try:
                self.examples.append(encoder.encode(row.text, row.aspect, label))
            except ValueError as exc:
                raise ValueError(
                    f"failed to encode review_id={row.review_id!r} "
                    f"aspect={row.aspect!r}: {exc}"
                ) from exc

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]

    @property
    def labels(self) -> list[int]:
        """Gold class ids in dataset order (empty when unlabelled)."""
        if not self.has_labels:
            return []
        return [int(ex["labels"]) for ex in self.examples]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights, normalised to mean 1, for imbalance."""
        if not self.has_labels:
            raise ValueError("class_weights() needs a labelled dataset")
        counts = torch.zeros(self.label_map.num_labels, dtype=torch.float)
        for label_id in self.labels:
            counts[label_id] += 1
        counts = counts.clamp(min=1.0)
        weights = counts.sum() / (self.label_map.num_labels * counts)
        return weights / weights.mean()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ABSADataset(n={len(self)}, labelled={self.has_labels}, "
            f"labels={self.label_map.labels})"
        )


def collate_batch(batch: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack equally-padded encodings into a batch."""
    batch = list(batch)
    return {key: torch.stack([ex[key] for ex in batch]) for key in batch[0]}
