#!/usr/bin/env python3
"""
Upload the exploded idiom JSONL to the HuggingFace dataset repo.

Requires prior authentication (run once, outside this script):
    huggingface-cli login

Usage:
    python upload_to_hf.py # uploads DEFAULT_EXPLODED_JSONL
    python upload_to_hf.py path/to/file.jsonl
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

from config import DEFAULT_EXPLODED_JSONL, HF_REPO_ID, HF_REPO_TYPE, HF_PRIVATE


def main():
    local_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXPLODED_JSONL

    if not local_file.exists():
        print(f"ERROR: file not found: {local_file}", file=sys.stderr)
        print("Run transform_dataset.py first to produce it.", file=sys.stderr)
        sys.exit(1)

    api = HfApi()

    # Creates the repo if it doesn't exist yet; does nothing if it already does.
    # HF_PRIVATE comes from config.py — switch to False after experiments are finished
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        private=HF_PRIVATE,
        exist_ok=True,
    )

    repo_filename = local_file.name  # just the filename — where it lands in the repo
    api.upload_file(
        path_or_fileobj=str(local_file),  # local disk path — where the file is
        path_in_repo=repo_filename,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
    )
	# Visibility
    visibility = "private" if HF_PRIVATE else "public"
    print(f"Uploaded {local_file} -> https://huggingface.co/datasets/{HF_REPO_ID}/blob/main/{repo_filename}")
    print(f"Repo visibility: {visibility}")


if __name__ == "__main__":
    main()