"""Convert official SemEval ABSA XML into the CSV this project's loader reads.

Why this exists: the laptop data arrived as CSV, but SemEval releases (and
MAMS) ship XML. The schemas line up exactly -- ``<sentence><text>`` plus
``<aspectTerm term= polarity= from= to=>`` -- so the only thing standing
between another domain and this pipeline is a column rename.

The output is deliberately the *raw* shape `load_raw_csv` expects rather than
anything cleaner: `conflict` rows are kept so that loader goes on reporting how
many it dropped, and the sentence id is carried through untouched.

Usage::

    python convert_semeval_xml.py Restaurants_Train_v2.xml Restaurants_Test_Gold.xml
    python convert_semeval_xml.py path/to/train.xml --out data/raw/MAMS_Train.csv

Offsets are verified, not trusted: every aspect term must slice back out of its
sentence at the stated character offsets, because the encoder's `aspect_mask`
is built from that span and a silent off-by-one would train the model to attend
to the wrong words. A mismatch fails the run unless
``--allow-offset-mismatch`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

LOGGER = logging.getLogger("convert_semeval_xml")

#: The header `load_raw_csv` renames from, in its order. `polarity` keeps the
#: SemEval spelling because the loader's COLUMN_RENAMES looks for that name.
FIELDNAMES = ["id", "Sentence", "Aspect Term", "polarity", "from", "to"]


def convert(xml_path: Path, out_path: Path, allow_offset_mismatch: bool = False) -> dict[str, int]:
    """Write one XML file out as CSV. Returns a count summary."""
    root = ET.parse(xml_path).getroot()
    sentences = root.findall("sentence")

    rows: list[dict[str, object]] = []
    mismatches: list[str] = []
    without_terms = 0

    for index, sentence in enumerate(sentences):
        text = sentence.findtext("text") or ""
        terms = sentence.findall("aspectTerms/aspectTerm")
        if not terms:
            # Normal and expected: SemEval omits <aspectTerms> entirely for a
            # sentence that expresses no aspect-level opinion. Nothing to train
            # on, so it is skipped rather than written with an empty aspect.
            without_terms += 1
            continue

        # The official ids are strings ("32897564#894393#2"); the laptop CSV
        # used integers. Either rides through as review_id, so the real one is
        # kept -- it is the only way back to the source sentence.
        sentence_id = sentence.get("id") or str(index)

        for term in terms:
            start, end = int(term.get("from", -1)), int(term.get("to", -1))
            surface = term.get("term", "")
            if text[start:end] != surface:
                mismatches.append(f"{sentence_id}: {surface!r} != {text[start:end]!r}")
            rows.append(
                {
                    "id": sentence_id,
                    "Sentence": text,
                    "Aspect Term": surface,
                    "polarity": term.get("polarity", ""),
                    "from": start,
                    "to": end,
                }
            )

    if mismatches and not allow_offset_mismatch:
        for line in mismatches[:10]:
            LOGGER.error("offset mismatch -- %s", line)
        raise SystemExit(
            f"{len(mismatches)} aspect term(s) do not slice back out of their sentence; "
            f"refusing to write {out_path.name}. Pass --allow-offset-mismatch to override."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sentences": len(sentences),
        "skipped_no_terms": without_terms,
        "rows": len(rows),
        "offset_mismatches": len(mismatches),
    }
    LOGGER.info(
        "%s -> %s: %d sentences (%d with no aspect terms, skipped), %d rows",
        xml_path.name,
        out_path,
        summary["sentences"],
        summary["skipped_no_terms"],
        summary["rows"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xml", nargs="+", type=Path, help="SemEval/MAMS XML file(s)")
    parser.add_argument(
        "--out",
        type=Path,
        help="output CSV path; only valid with a single input (default: <name>.csv in --out-dir)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "raw",
        help="directory for the CSVs (default: training/data/raw)",
    )
    parser.add_argument(
        "--allow-offset-mismatch",
        action="store_true",
        help="write the CSV even if an aspect term does not match its offsets",
    )
    args = parser.parse_args(argv)

    if args.out and len(args.xml) > 1:
        parser.error("--out takes a single input file; use --out-dir for several")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for xml_path in args.xml:
        if not xml_path.is_file():
            parser.error(f"no such file: {xml_path}")
        out_path = args.out or args.out_dir / f"{xml_path.stem}.csv"
        convert(xml_path, out_path, args.allow_offset_mismatch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
