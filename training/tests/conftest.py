"""Shared fixtures; puts ``training/`` on the path so ``src`` imports work."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

RAW_DIR = TRAINING_ROOT / "data" / "raw"

#: A tiny randomly-initialised BERT. Keeps every model test on CPU, offline
#: and fast -- the architecture under test is the attention and the head, not
#: the pretrained weights.
TINY_ENCODER_KWARGS = dict(
    vocab_size=256,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=37,
    max_position_embeddings=64,
)


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    return RAW_DIR


@pytest.fixture(scope="session")
def train_csv(raw_dir: Path) -> Path:
    path = raw_dir / "Laptop_Train_v2.csv"
    if not path.is_file():
        pytest.skip(f"missing corpus: {path}")
    return path


@pytest.fixture(scope="session")
def gold_csv(raw_dir: Path) -> Path:
    path = raw_dir / "Laptops_Test_Gold.csv"
    if not path.is_file():
        pytest.skip(f"missing corpus: {path}")
    return path


@pytest.fixture
def toy_csv(tmp_path: Path) -> Path:
    """A tiny CSV exercising conflict rows, multi-aspect sentences and quoting."""
    path = tmp_path / "toy.csv"
    path.write_text(
        "id,Sentence,Aspect Term,polarity,from,to\n"
        "1,The screen is bright and the keyboard is nice,screen,positive,4,10\n"
        "1,The screen is bright and the keyboard is nice,keyboard,positive,29,37\n"
        '2,"Battery life is short, but the price is fine.",Battery life,negative,0,12\n'
        '2,"Battery life is short, but the price is fine.",price,positive,31,36\n'
        "3,It came with the usual software,software,neutral,23,31\n"
        "4,The fan is loud but cooling is great,fan,conflict,4,7\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def toy_frame(toy_csv: Path) -> pd.DataFrame:
    from src.data.dataset import load_raw_csv

    return load_raw_csv(toy_csv)


@pytest.fixture(scope="session")
def tiny_backbone():
    """A fresh tiny BertModel, built from config -- no download."""
    from transformers import BertConfig, BertModel

    def factory():
        torch.manual_seed(0)
        return BertModel(BertConfig(**TINY_ENCODER_KWARGS))

    return factory


@pytest.fixture
def tiny_model(tiny_backbone):
    """A :class:`BertAspectAttention` over the tiny encoder."""
    from src.models.absa_model import BertAspectAttention

    torch.manual_seed(0)
    return BertAspectAttention(
        model_name="tiny-bert-test", num_labels=3, dropout=0.1, encoder=tiny_backbone()
    )


@pytest.fixture
def random_batch():
    """A collated batch with realistic, disjoint segment masks."""

    def factory(batch_size: int = 4, seq_len: int = 16, with_labels: bool = True):
        torch.manual_seed(1)
        aspect_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        context_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        # [CLS] context... [SEP] aspect [SEP]
        context_mask[:, 1 : seq_len - 4] = True
        aspect_mask[:, seq_len - 3 : seq_len - 1] = True
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        token_type_ids[:, seq_len - 4 :] = 1

        batch = {
            "input_ids": torch.randint(0, 256, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "token_type_ids": token_type_ids,
            "aspect_mask": aspect_mask,
            "context_mask": context_mask,
        }
        if with_labels:
            batch["labels"] = torch.randint(0, 3, (batch_size,))
        return batch

    return factory


@pytest.fixture
def tiny_encoder(tmp_path):
    """An ABSAEncoder over a minimal hand-built WordPiece tokenizer.

    Keeps the artifact tests offline: ``build_artifact`` saves whatever
    tokenizer it is given, and ``load_artifact`` must read it back.
    """
    from transformers import BertTokenizerFast

    from src.preprocessing.tokenizer import ABSAEncoder

    vocab = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "the", "screen", "is", "bright", "keyboard", "nice",
        "battery", "life", "price", "software", "fan", "loud", "and", "but",
        "short", "fine", "came", "with", "usual", "cooling", "great", "it",
    ]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    tokenizer_dir = tmp_path / "tiny-tokenizer"
    BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True).save_pretrained(
        tokenizer_dir
    )
    return ABSAEncoder(str(tokenizer_dir), max_length=24)
