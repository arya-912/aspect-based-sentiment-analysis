"""Pool several raw ABSA CSVs into one corpus for a joint run.

The merge goes through ``load_raw_csv`` rather than ``pd.concat`` on the raw
files, because the raw laptop CSV cannot simply be stacked with the others:

* It ends its header with a trailing comma, so pandas reads an extra
  ``Unnamed: 6`` column, and seven of its rows are shifted one field right
  with their real ``to`` offset sitting in that column. ``load_raw_csv``
  repairs those; a naive concat would either drop the column (destroying the
  seven rows) or keep it (leaving a column the other inputs lack).
* ``_realign_shifted_rows`` detects the spill column as ``df.columns[-1]``, so
  a merged file with any extra trailing column -- a ``source`` tag, say --
  would send the realigner reading the wrong field. That is why provenance is
  carried in the id prefix instead of a new column.

So each input is loaded, cleaned and validated by the project's own code, and
the result is written back out under the raw header names. The merged file is
therefore already clean: loading it reports zero conflict rows dropped,
because they were dropped on the way in.

``review_id`` is prefixed with a tag from the source filename, since ids are
only unique within one corpus -- the laptop file numbers from 1, the
restaurant file uses SemEval's string ids. Nothing in the pipeline splits on
the id (``split_train_val`` groups by sentence text), but colliding ids in a
merged file are a trap for anything that later joins on them.

Usage::

    python merge_corpora.py data/raw/Laptop_Train_v2.csv data/raw/Restaurants_Train_v2.csv \\
        --out data/raw/Joint_Train.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.dataset import load_raw_csv  # noqa: E402

LOGGER = logging.getLogger("merge_corpora")

#: Internal name -> the raw header the loader expects to read back.
RAW_HEADER = {
    "review_id": "id",
    "text": "Sentence",
    "aspect": "Aspect Term",
    "label": "polarity",
    "from": "from",
    "to": "to",
}


def source_tag(path: Path) -> str:
    """A short, stable prefix for one input's ids, from its filename.

    ``Laptop_Train_v2.csv`` -> ``laptop``, ``Restaurants_Test_Gold.csv`` ->
    ``restaurants``.
    """
    return path.stem.split("_")[0].lower()


def merge(paths: list[Path], out_path: Path) -> pd.DataFrame:
    """Load, tag and concatenate the inputs, writing one raw-shaped CSV."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = load_raw_csv(path)
        frame = frame.assign(review_id=f"{source_tag(path)}:" + frame["review_id"].astype(str))
        LOGGER.info(
            "%-28s %5d usable rows, %5d sentences, %s",
            path.name,
            len(frame),
            frame["text"].nunique(),
            frame["label"].value_counts().to_dict(),
        )
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    out = merged.rename(columns=RAW_HEADER)[list(RAW_HEADER.values())]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    LOGGER.info(
        "wrote %s: %d rows, %d sentences, %s",
        out_path,
        len(out),
        merged["text"].nunique(),
        merged["label"].value_counts().to_dict(),
    )
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+", type=Path, help="raw CSVs to pool, in order")
    parser.add_argument("--out", type=Path, required=True, help="merged CSV to write")
    args = parser.parse_args(argv)

    for path in args.csv:
        if not path.is_file():
            parser.error(f"no such file: {path}")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    merge(args.csv, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
