#!/usr/bin/env python3
"""
Experiment 1 — Direct vs. KB-CoT translation using EuroLLM-9B-Instruct.

Requires (once, before running):
    huggingface-cli login
    -> then visit https://huggingface.co/utter-project/EuroLLM-9B-Instruct
       and accept the model's access conditions (gated repo).

Usage:
    python translate.py                  # downloads the dataset from HuggingFace
    python translate.py --limit 20        # random sample of 20 rows (reproducible)
    python translate.py --limit 20 --sample-seed 7   # a different random 20
    python translate.py --input path/to/local_file.jsonl --output path/to/results.jsonl
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from huggingface_hub import hf_hub_download

from config import DATA_DIR, DEFAULT_RESULTS_PATH, HF_REPO_ID, HF_REPO_TYPE

MODEL_ID = "utter-project/EuroLLM-9B-Instruct"
SEED = 42


# --- prompt builders (Direct vs. KB-CoT, IdiomKB Table 3 structure) ---

def build_direct_prompt(cs_sentence):
    """Baseline condition: no idiom hint."""
    user_content = (
        f"Translate the following Czech sentence into English.\n"
        f"Czech: {cs_sentence}\n"
        f"English:"
    )
    return [{"role": "user", "content": user_content}]


def build_kbcot_prompt(cs_sentence, figurative_meaning):
    """Treatment condition: figurative meaning injected as a preamble,
    matching IdiomKB's KB-CoT structure."""
    user_content = (
        f'This sentence contains an idiom whose figurative meaning is: '
        f'"{figurative_meaning}".\n\n'
        f"Given the above knowledge, translate the following Czech sentence into English.\n"
        f"Czech: {cs_sentence}\n"
        f"English:"
    )
    return [{"role": "user", "content": user_content}]


# --- model loading and generation ---

def load_model():
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 9B params in bf16 needs ~18GB — doesn't fit a 16GB T4, which forces
    # CPU offloading and breaks generate(). 4-bit quantization brings it
    # down to ~5-6GB, fitting fully on GPU with no offloading at all.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=256):
    """Deterministic (greedy) generation — do_sample=False, fixed seed.
    This is what Test A (determinism) and Test B (isolation) check against."""
    torch.manual_seed(SEED)

    # Get the prompt as plain text first (unambiguous), then tokenize
    # explicitly ourselves — apply_chat_template's tokenize=True return type
    # varies across transformers versions and environments; this avoids
    # relying on it to hand back a ready-to-use tensor.
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy decoding — no randomness
        )

    generated = output_ids[0][encoded["input_ids"].shape[-1]:]  # strip the prompt, keep only new tokens
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# --- data loading ---

def load_exploded_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --- main run loop ---

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                         help="Exploded idiom JSONL. If omitted, downloads the current "
                              "version directly from the HuggingFace dataset repo.")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH,
                         help=f"Where to write results (default: {DEFAULT_RESULTS_PATH})")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process N rows (use for a pilot run)")
    parser.add_argument("--sample-seed", type=int, default=SEED,
                         help=f"Seed for random row sampling when --limit is set "
                              f"(default: {SEED}). Same seed -> same sampled rows "
                              f"every run, which Test A (determinism) depends on.")
    args = parser.parse_args()

    if args.input is None:
        print(f"Downloading idioms-exp1.jsonl from HF dataset repo {HF_REPO_ID} ...")
        args.input = Path(hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename="idioms-exp1.jsonl",
        ))
    elif not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        print("Run transform_dataset.py first to produce it, or omit --input to pull from HF.", file=sys.stderr)
        sys.exit(1)

    rows = load_exploded_rows(args.input)
    if args.limit:
        rng = random.Random(args.sample_seed)
        rows = rng.sample(rows, min(args.limit, len(rows)))
    print(f"Loaded {len(rows)} sentence rows from {args.input}")

    model, tokenizer = load_model()

    results = []
    for i, row in enumerate(rows, start=1):
        cs_sentence = row["cs_sentence"]
        figurative_meaning = row["figurative_meaning"]

        direct_output = generate(model, tokenizer, build_direct_prompt(cs_sentence))
        kbcot_output = generate(model, tokenizer, build_kbcot_prompt(cs_sentence, figurative_meaning))

        result = {
            "id": row["id"],
            "cs_sentence": cs_sentence,
            "figurative_meaning": figurative_meaning,
            "preferred_rendering": row.get("preferred_rendering", ""),
            "direct_output": direct_output,
            "kbcot_output": kbcot_output,
            "model_id": MODEL_ID,
            "seed": SEED,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        results.append(result)
        print(f"[{i}/{len(rows)}] id={row['id']}")
        print(f"  CS      : {cs_sentence}")
        print(f"  Direct  : {direct_output}")
        print(f"  KB-CoT  : {kbcot_output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} results -> {args.output}")
    print("Next: run the Test A (determinism) and Test B (isolation) checks before trusting these outputs.")


if __name__ == "__main__":
    main()
