#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# ========= Config =========
TREC_DL_YEAR = "2022"
MODEL = "gpt-oss-20b"
VARIANT = "eng_crit"   # the file we want to augment

LABEL_DIR = Path("outputs") / "llm_label" / f"trec_dl_{TREC_DL_YEAR}" / MODEL

RAW_FILE = LABEL_DIR / f"{MODEL}_trecdl_{TREC_DL_YEAR}_raw_labels.csv"
VAR_FILE = LABEL_DIR / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{VARIANT}_labels.csv"

# Output (change to overwrite if you want)
OUT_FILE = LABEL_DIR / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{VARIANT}_with_relevance_labels.csv"


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw labels file not found: {RAW_FILE}")
    if not VAR_FILE.exists():
        raise FileNotFoundError(f"Variant file not found: {VAR_FILE}")

    # Load CSVs
    df_raw = pd.read_csv(RAW_FILE)
    df_var = pd.read_csv(VAR_FILE)

    # Determine the relevance column name in the raw file
    if "relevance" in df_raw.columns:
        rel_col = "relevance"
    elif "NIST_relevance" in df_raw.columns:
        rel_col = "NIST_relevance"
    else:
        raise ValueError(
            f"No relevance column found in {RAW_FILE} "
            "(expected 'relevance' or 'NIST_relevance')."
        )

    # Keep only the columns needed for the join
    key_cols = ["qid", "pid", "query"]
    for col in key_cols:
        if col not in df_raw.columns or col not in df_var.columns:
            raise ValueError(f"Key column '{col}' missing in one of the files.")

    df_raw_sub = df_raw[key_cols + [rel_col]]

    # Merge relevance into variant file
    merged = df_var.merge(
        df_raw_sub,
        on=key_cols,
        how="left",
        suffixes=("", "_raw"),
    )

    # Rename relevance column to 'relevance' (if it was NIST_relevance)
    if rel_col != "relevance":
        merged = merged.rename(columns={rel_col: "relevance"})

    # Report how many rows did not find a match
    missing = merged["relevance"].isna().sum()
    print(f"Rows with missing relevance after merge: {missing}")

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f"Saved file with relevance column to: {OUT_FILE}")


if __name__ == "__main__":
    main()
