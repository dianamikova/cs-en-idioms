"""
Shared configuration for the Czech idiom MT project scripts.

    from config import DEFAULT_INPUT_CSV, DEFAULT_EXPLODED_JSONL, HF_REPO_ID...
"""

from pathlib import Path

# --- filesystem layout ---
# This file lives in scripts/, one level below the repo root.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_INPUT_CSV = DATA_DIR / "cs_en_idioms_raw.csv"
DEFAULT_EXPLODED_JSONL = DATA_DIR / "idioms-exp1.jsonl"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "exp1_results.jsonl"

# --- HuggingFace ---
HF_REPO_ID = "dianamikova/cs-en-idioms"
HF_REPO_TYPE = "dataset"
HF_PRIVATE = True  # switch to False before publishing