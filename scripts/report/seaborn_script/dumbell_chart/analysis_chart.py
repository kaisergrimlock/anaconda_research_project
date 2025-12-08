#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# =========================
# Config
# =========================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]  # adjust depth if needed
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Path to the CSV that contains llm_relevance and the criterion scores
INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "llm_label"
    / "trec_dl_2022"
    / "gpt-oss-20b"
    / "gpt-oss-20b_trecdl_2022_raw_crit_labels.csv"
)

# Columns that are individual criterion scores (edit if your names differ)
SCORE_COLS = ["contextuality", "coverage", "exactness", "topicality"]

# Where to write the summary
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "llm_relevance_sum_scores_min_median_max.csv"


def main() -> None:
    # Load data
    df = pd.read_csv(INPUT_CSV)

    # Basic checks
    if "llm_relevance" not in df.columns:
        raise ValueError(
            f"'llm_relevance' column not found in {INPUT_CSV}. "
            f"Available columns: {list(df.columns)}"
        )

    missing_scores = [c for c in SCORE_COLS if c not in df.columns]
    if missing_scores:
        raise ValueError(
            f"These SCORE_COLS are missing in {INPUT_CSV}: {missing_scores}. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure numeric
    df["llm_relevance"] = pd.to_numeric(df["llm_relevance"], errors="coerce")
    for c in SCORE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing llm_relevance or any score
    df_valid = df.dropna(subset=["llm_relevance"] + SCORE_COLS).copy()

    if df_valid.empty:
        raise RuntimeError("No valid rows with llm_relevance and all score columns present.")

    # Sum of all score columns per document
    df_valid["score_sum"] = df_valid[SCORE_COLS].sum(axis=1)

    # Group by llm_relevance and compute min / median / max of the score_sum
    summary = (
        df_valid
        .groupby("llm_relevance")["score_sum"]
        .agg(["min", "median", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "min": "score_sum_min",
                "median": "score_sum_median",
                "max": "score_sum_max",
                "count": "n_docs",
            }
        )
        .sort_values("llm_relevance")
    )

    # Save and print
    summary.to_csv(OUT_CSV, index=False)
    print("Summary by llm_relevance:")
    print(summary.to_string(index=False))
    print(f"\nWrote summary CSV to: {OUT_CSV}")


if __name__ == "__main__":
    main()
