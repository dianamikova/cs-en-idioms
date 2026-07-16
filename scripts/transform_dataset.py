#!/usr/bin/env python3
"""
Transform the Czech idiom dataset (CSV export of cs_idioms_final) into the
lean, exploded JSONL format needed for Experiment 1.

One input row (one idiom, possibly several example sentences) becomes
several output rows (one per example sentence), each carrying the idiom's
id, the single sentence, and the primary figurative meaning / rendering.

Usage (run from anywhere, paths default from config.py, relative to repo root):
    python transform_dataset.py
    python transform_dataset.py path/to/idioms.csv output.jsonl
    python transform_dataset.py path/to/idioms.csv output.jsonl --exclude-ids 8 64 78 50
"""

import argparse
import csv
import json
import sys

from config import DEFAULT_INPUT_CSV, DEFAULT_EXPLODED_JSONL

DEFAULT_INPUT_PATH = DEFAULT_INPUT_CSV
DEFAULT_OUTPUT_PATH = DEFAULT_EXPLODED_JSONL

# The only columns this script reads from the source CSV, by name
REQUIRED_COLUMNS = ["id", "cs_examples", "meanings_en", "preffered_rendering_en"]


def detect_delimiter(path):
    """Sniff comma vs. semicolon from the header line (CZ-locale Excel/Sheets
    exports sometimes use semicolon); default to comma."""
    with open(path, encoding="utf-8-sig") as f:
        first_line = f.readline()
    return ";" if first_line.count(";") > first_line.count(",") else ","


def load_rows(csv_path):
    delimiter = detect_delimiter(csv_path)
    print(f"Detected column delimiter: {delimiter!r}")
    # utf-8-sig strips the BOM some spreadsheet apps prepend, so diacritics read correctly
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            print(f"ERROR: required column(s) not found in header: {missing}", file=sys.stderr)
            print(f"Columns found: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)

        # Project down to only the required columns, by name, row by row.
        rows = [{col: row.get(col, "") for col in REQUIRED_COLUMNS} for row in reader]
    return rows


def explode_examples(rows):
    """
    rows: list of dicts already projected to REQUIRED_COLUMNS (one dict per idiom).
    Yields one dict per example sentence.
    """
    for r in rows:
        cs_examples = (r.get("cs_examples") or "").strip()
        if not cs_examples:
            continue  # nothing to explode for this idiom, flagged as a warning in main()

        sentences = [s.strip() for s in cs_examples.split("|") if s.strip()]

        meanings = (r.get("meanings_en") or "").strip()
        renderings = (r.get("preffered_rendering_en") or "").strip()

        # Take the first/primary sense — these columns can hold multiple
        # pipe-separated senses; Experiment 1 uses the primary one only.
        primary_meaning = meanings.split("|")[0].strip() if meanings else ""
        primary_rendering = renderings.split("|")[0].strip() if renderings else ""

        # id may come through as a string ("3") or a formula artifact — normalise to int
        raw_id = (r.get("id") or "").strip()
        try:
            idiom_id = int(float(raw_id))
        except ValueError:
            idiom_id = raw_id  # leave as-is; will surface in the warnings below

        for s in sentences:
            yield {
                "id": idiom_id,
                "cs_sentence": s,
                "figurative_meaning": primary_meaning,
                "preferred_rendering": primary_rendering,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv", nargs="?", default=DEFAULT_INPUT_PATH,
        help=f"Path to the source .csv file (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "output_jsonl", nargs="?", default=DEFAULT_OUTPUT_PATH,
        help=f"Path to write the exploded JSONL (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--exclude-ids",
        nargs="*",
        type=int,
        default=[],
        help="Idiom ids to exclude (e.g. borderline/unverified entries)",
    )
    args = parser.parse_args()

    try:
        rows = load_rows(args.input_csv)  # already projected to REQUIRED_COLUMNS only
    except FileNotFoundError:
        print(f"ERROR: file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR reading {args.input_csv}: {e}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("ERROR: no rows read — check the file path and header row.", file=sys.stderr)
        sys.exit(1)

    if args.exclude_ids:
        before = len(rows)
        def _id_of(r):
            try:
                return int(float((r.get("id") or "").strip()))
            except ValueError:
                return None
        rows = [r for r in rows if _id_of(r) not in args.exclude_ids]
        print(f"Excluded {before - len(rows)} idiom(s) by id: {args.exclude_ids}")

    exploded = list(explode_examples(rows))

    # --- sanity warnings ---
    skipped_no_examples = [
        (r.get("id") or "?") for r in rows if not (r.get("cs_examples") or "").strip()
    ]
    empty_meaning = [e["id"] for e in exploded if not e["figurative_meaning"]]
    empty_rendering = [e["id"] for e in exploded if not e["preferred_rendering"]]

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for entry in exploded:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Read {len(rows)} idioms from {args.input_csv}")
    print(f"Wrote {len(exploded)} exploded sentence rows -> {args.output_jsonl}")

    if skipped_no_examples:
        print(f"WARNING: {len(skipped_no_examples)} idiom(s) had no cs_examples, skipped entirely: {skipped_no_examples}")
    if empty_meaning:
        print(f"WARNING: {len(set(empty_meaning))} idiom(s) produced rows with empty figurative_meaning: {sorted(set(empty_meaning), key=str)}")
    if empty_rendering:
        print(f"WARNING: {len(set(empty_rendering))} idiom(s) produced rows with empty preferred_rendering: {sorted(set(empty_rendering), key=str)}")
    if not (skipped_no_examples or empty_meaning or empty_rendering):
        print("No warnings. Clean.")


if __name__ == "__main__":
    main()