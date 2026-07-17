#!/usr/bin/env python3
"""
Direct-only spontaneous drift test.

Question: does the base model, with NO injection and NO KB-CoT mechanism
involved at all, spontaneously produce idiomatic English on a sentence
that is genuinely literal but happens to share vocabulary with a real
Czech idiom?

This tests the model itself, not the injection pipeline — pure lexical-
priming robustness. Only build_direct_prompt() is ever called; KB-CoT
is not used anywhere in this script.

Reuses translate.py's model loading, direct prompt builder, and
generation — no duplicated logic.

Requires (once, before running):
    huggingface-cli login
    -> then visit https://huggingface.co/utter-project/EuroLLM-9B-Instruct
       and accept the model's access conditions (gated repo).

Usage:
    python test_direct_only_drift.py
    python test_direct_only_drift.py --input data/direct_only_drift_test.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

from translate import load_model, build_direct_prompt, generate, MODEL_ID, SEED
from config import RESULTS_DIR

DEFAULT_INPUT = Path("data/direct_only_drift_test.jsonl")
DEFAULT_OUTPUT = RESULTS_DIR / "direct_only_drift_results.jsonl"


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_leakage(text, risk_phrases):
    """Case-insensitive substring check. Returns the list of phrases found."""
    text_lower = text.lower()
    return [p for p in risk_phrases if p.lower() in text_lower]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help=f"Input JSONL with cs_sentence, risk_phrases "
                              f"(default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                         help=f"Where to write results (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    model, tokenizer = load_model()

    results = []
    flagged_count = 0
    control_count = 0

    for i, row in enumerate(rows, start=1):
        cs_sentence = row["cs_sentence"]
        risk_phrases = row.get("risk_phrases", [])
        is_control = len(risk_phrases) == 0

        # ONLY Direct is ever called — no KB-CoT prompt exists in this script at all.
        direct_output = generate(model, tokenizer, build_direct_prompt(cs_sentence))

        found = check_leakage(direct_output, risk_phrases)
        flagged = bool(found) and not is_control

        if is_control:
            control_count += 1
            status = "CONTROL"
        elif flagged:
            flagged_count += 1
            status = "FLAGGED (spontaneous drift)"
        else:
            status = "PASS (stayed literal)"

        result = {
            "id": row["id"],
            "cs_sentence": cs_sentence,
            "direct_output": direct_output,
            "risk_phrases": risk_phrases,
            "phrases_found": found,
            "status": status,
            "note": row.get("note", ""),
            "model_id": MODEL_ID,
            "seed": SEED,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        results.append(result)

        print(f"[{i}/{len(rows)}] id={row['id']} -> {status}")
        print(f"  CS      : {cs_sentence}")
        print(f"  Direct  : {direct_output}")
        if found:
            print(f"  Found   : {found}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tested = len(rows) - control_count
    print(f"\n{'=' * 60}")
    print(f"Wrote {len(results)} results -> {args.output}")
    print(f"Tested (lexical-overlap rows) : {tested}")
    print(f"Controls (no overlap)         : {control_count}")
    print(f"FLAGGED (spontaneous drift)   : {flagged_count}")
    print(f"PASS (stayed literal)         : {tested - flagged_count}")


if __name__ == "__main__":
    main()