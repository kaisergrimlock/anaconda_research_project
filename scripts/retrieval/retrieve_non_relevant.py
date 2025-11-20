#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

# =========================
# Path configuration
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]   # because script is in scripts/retrieval/

INPUT_DIR    = PROJECT_ROOT / "retrieved" / "trec_dl_2023" / "judged"
FILE_PATTERN = "*.csv"

LABEL_COL    = "relevance"
LABEL_VALUE  = 0
SAMPLE_SIZE  = 100

PASSAGE_COL  = "passage"   # change if needed
MAX_WORDS    = 1000
MAX_WORD_LEN = 20

# Desired output columns (with query_nr)
OUTPUT_COLS_SRC  = [
    "qid",
    "query",
    "pid_qrels",
    "pid_resolved",
    "passage",
    "relevance",
    "query_fr",         # source column name
    "passage_injected",
]
RENAME_FOR_OUTPUT = {
    "query_fr": "query_nr",  # output header should use query_nr
}

OUTPUT_CSV = PROJECT_ROOT / "sample_label0_judged.csv"


def passage_ok(text: str) -> bool:
    """Return True if passage length is acceptable."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    words = text.split()
    if len(words) > MAX_WORDS:
        return False
    if any(len(w) > MAX_WORD_LEN for w in words):
        return False
    return True


def main() -> None:
    print(f"[INFO] Using INPUT_DIR = {INPUT_DIR}")

    files = sorted(INPUT_DIR.rglob(FILE_PATTERN))
    if not files:
        print(f"[WARN] No files found.")
        return

    print(f"[INFO] Found {len(files)} CSV files, scanning…")

    all_rows = []

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] Could not read {fp}: {e}")
            continue

        # basic column checks
        if LABEL_COL not in df.columns:
            print(f"[WARN] Column {LABEL_COL!r} missing in {fp}, skipping.")
            continue
        if PASSAGE_COL not in df.columns:
            print(f"[WARN] Column {PASSAGE_COL!r} missing in {fp}, skipping.")
            continue

        # only label==0
        sub = df[df[LABEL_COL] == LABEL_VALUE].copy()
        if sub.empty:
            continue

        # filter by passage length / long words
        before = len(sub)
        mask_ok = sub[PASSAGE_COL].apply(passage_ok)
        sub = sub[mask_ok]
        after = len(sub)

        if after == 0:
            continue

        if after < before:
            print(f"[INFO] {fp.name}: filtered out {before - after} long passages, kept {after}")

        all_rows.append(sub)

    if not all_rows:
        print("[INFO] No rows with label 0 and acceptable passage length found.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"[INFO] Total label-0 rows after length filter: {len(combined)}")

    # Ensure all needed columns exist (fill missing ones as empty)
    for col in OUTPUT_COLS_SRC:
        if col not in combined.columns:
            combined[col] = ""

    # sample
    if len(combined) <= SAMPLE_SIZE:
        sample = combined
        print(f"[INFO] Only {len(combined)} rows available; taking all.")
    else:
        sample = combined.sample(n=SAMPLE_SIZE, random_state=123)
        print(f"[INFO] Sampled {SAMPLE_SIZE} rows.")

    # select & reorder columns, then rename query_fr -> query_nr
    sample = sample[OUTPUT_COLS_SRC].rename(columns=RENAME_FOR_OUTPUT)

    # write output
    sample.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Wrote sample to {OUTPUT_CSV}")
    print(f"[INFO] Output columns: {list(sample.columns)}")


if __name__ == "__main__":
    main()
