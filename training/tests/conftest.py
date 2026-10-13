"""Shared fixtures; puts ``training/`` on the path so ``src`` imports work."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

RAW_DIR = TRAINING_ROOT / "data" / "raw"


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
