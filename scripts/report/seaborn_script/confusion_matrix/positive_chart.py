#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit

# -------- Config (keep in sync with your main script) --------
TREC_DL_YEAR = "2023"
MODEL = "gpt-oss-20b"
LANG  = "raw"  # "raw","eng","vi","fr", etc.

if LANG == "raw":
    suffix = "raw"
else:
    suffix = LANG

LLM_FILE = (
    Path("outputs/llm_label")
    / MODEL
    / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{suffix}.csv"
)

LLM_RELEVANCE_COL = "llm_relevance"  # column name in the CSV
GOLD_RELEVANCE_COL = "relevance"  # column name for gold relevance


def main() -> None:
    bump_field_limit()

    if not LLM_FILE.exists():
        raise FileNotFoundError(f"LLM file not found: {LLM_FILE}")

    # If the file is large, you could use chunksize; for now we just read it fully.
    df = pd.read_csv(LLM_FILE, engine="python", on_bad_lines="skip")

    if LLM_RELEVANCE_COL not in df.columns:
        raise KeyError(
            f"Column '{LLM_RELEVANCE_COL}' not found in {LLM_FILE}. "
            f"Available columns: {list(df.columns)}"
        )

    if GOLD_RELEVANCE_COL not in df.columns:
        raise KeyError(
            f"Column '{GOLD_RELEVANCE_COL}' not found in {LLM_FILE}. "
            f"Available columns: {list(df.columns)}"
        )

    # Loop through each row and print both LLM relevance and gold relevance
    positivity = 0
    print("positivity: ")
    for idx, (gold, llm) in enumerate(
        zip(df[GOLD_RELEVANCE_COL], df[LLM_RELEVANCE_COL]),
        start=1
    ):
        # Treat missing gold/llm as 0 (or whatever default you want)
        if pd.isna(gold):
            gold_val = 0
        else:
            gold_val = int(gold)

        if pd.isna(llm):
            llm_val = 0
        else:
            llm_val = int(llm)

        positivity += llm_val - gold_val
    print(positivity)
if __name__ == "__main__":
    main()
