#!/usr/bin/env python3
"""
KB-CoT false-positive test.

Question: does injecting a WRONG figurative meaning cause KB-CoT to leak
idiomatic content into a translation of a genuinely literal sentence?

Unlike translate.py's Direct-vs-KB-CoT ablation, this script only cares
about ONE thing per row: does kbcot_output contain any of that row's
risk_phrases? If yes -> FLAGGED (the injection leaked). If no -> PASS
(the model resisted the bad hint).

Reuses translate.py's model loading, prompt builders, and generation —
no duplicated logic, so both scripts stay in sync with the same
determinism/quantization setup.

Requires (once, before running):
    huggingface-cli login
    -> then visit https://huggingface.co/utter-project/EuroLLM-9B-Instruct
       and accept the model's access conditions (gated repo).

Usage:
    python test_kbcot_false_positive.py
    python test_kbcot_false_positive.py --input data/kbcot_false_positive_test.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

from translate import load_model, build_direct_prompt, build_kbcot_prompt, generate, MODEL_ID, SEED
from config import RESULTS_DIR

DEFAULT_INPUT = Path("data/kbcot_false_positive_test.jsonl")
DEFAULT_OUTPUT = RESULTS_DIR / "kbcot_false_positive_results.jsonl"


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
                         help=f"Input JSONL with cs_sentence, figurative_meaning, "
                              f"risk_phrases (default: {DEFAULT_INPUT})")
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
        figurative_meaning = row.get("figurative_meaning", "")
        risk_phrases = row.get("risk_phrases", [])

        direct_output = generate(model, tokenizer, build_direct_prompt(cs_sentence))

        if figurative_meaning.strip():
            kbcot_output = generate(model, tokenizer, build_kbcot_prompt(cs_sentence, figurative_meaning))
        else:
            kbcot_output = direct_output  # control row, no injection

        found = check_leakage(kbcot_output, risk_phrases)
        is_control = not figurative_meaning.strip()
        flagged = bool(found) and not is_control

        if is_control:
            control_count += 1
            status = "CONTROL"
        elif flagged:
            flagged_count += 1
            status = "FLAGGED (leaked)"
        else:
            status = "PASS (resisted)"

        result = {
            "id": row["id"],
            "cs_sentence": cs_sentence,
            "injected_meaning": figurative_meaning,
            "direct_output": direct_output,
            "kbcot_output": kbcot_output,
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
        if figurative_meaning.strip():
            print(f"  Injected: {figurative_meaning}")
        print(f"  Direct  : {direct_output}")
        print(f"  KB-CoT  : {kbcot_output}")
        if found:
            print(f"  Found   : {found}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tested = len(rows) - control_count
    print(f"\n{'=' * 60}")
    print(f"Wrote {len(results)} results -> {args.output}")
    print(f"Tested (injection rows) : {tested}")
    print(f"Controls (no injection) : {control_count}")
    print(f"FLAGGED (leaked)        : {flagged_count}")
    print(f"PASS (resisted)         : {tested - flagged_count}")


if __name__ == "__main__":
    main()