#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit
from helpers.output_writer import (
    write_confusion_outputs,
    write_metrics,
    save_heatmap,
    write_df,
)

from helpers.metrics_llm import (
    compute_mae,
    compute_weighted_kappa_ordinal,
    compute_unweighted_kappa,
    compute_krippendorff_alpha_paired,
    binarize_labels,
)

# -------- Config --------
TREC_DL_YEARS = ["2021", "2022"]   # <-- combined years
MODEL = "llama3-8b-instruct"  # e.g., "qwen3-32b-v1", "gpt-oss-20b", etc.
LANG  = "raw_crit"  # "raw","eng","vi","fr", etc.

LABELS = [0, 1, 2, 3]

# Output naming: "2021_2022"
YEAR_TAG = "_".join(TREC_DL_YEARS)

OUT_DIR = Path("figures") / YEAR_TAG / MODEL / "confusion_matrix" / LANG
OUT_COUNTS       = OUT_DIR / "confusion_matrix_llm_vs_nist.csv"
OUT_PCT          = OUT_DIR / "confusion_matrix_llm_vs_nist_pct.csv"
OUT_SVG          = OUT_DIR / "confusion_matrix_llm_vs_nist.svg"
OUT_INVALID_ROWS = OUT_DIR / "rows_with_missing_or_invalid_labels.csv"


def _llm_file_for_year(year: str) -> Path:
    return (
        Path("outputs/llm_label")
        / f"trec_dl_{year}"
        / MODEL
        / f"{MODEL}_trecdl_{year}_{LANG}_labels.csv"
    )


def load_and_prepare() -> pd.DataFrame:
    """
    Load CSVs for all years in TREC_DL_YEARS. Each CSV must contain:
      - relevance (NIST)
      - llm_relevance (LLM)
    Concatenate them into one DataFrame and standardize columns (NIST, LLM).
    """
    bump_field_limit()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for year in TREC_DL_YEARS:
        llm_file = _llm_file_for_year(year)
        if not llm_file.exists():
            raise FileNotFoundError(f"Missing LLM label file for year {year}: {llm_file}")

        df_y = pd.read_csv(llm_file)

        if "relevance" not in df_y.columns or "llm_relevance" not in df_y.columns:
            raise ValueError(
                f"Expected columns 'relevance' and 'llm_relevance' in {llm_file}, "
                f"but got: {list(df_y.columns)}"
            )

        # keep provenance
        df_y["trec_dl_year"] = year

        # Coerce to numeric (invalid parses => NaN)
        df_y["NIST"] = pd.to_numeric(df_y["relevance"], errors="coerce")
        df_y["LLM"]  = pd.to_numeric(df_y["llm_relevance"], errors="coerce")

        frames.append(df_y)

    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def split_valid_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate rows with valid labels from rows with missing/invalid labels.

    Valid rows:
        - NIST and LLM in LABELS
    Everything else is invalid and written to a separate CSV.
    """
    valid_mask = df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)

    valid_df   = df[valid_mask].copy()
    invalid_df = df[~valid_mask].copy()

    print("Total rows:", len(df))
    print("Valid rows:", valid_mask.sum())
    print("Invalid rows:", (~valid_mask).sum())

    # Keep the original data for debugging, drop computed columns if you want
    invalid_out = invalid_df.drop(columns=["NIST", "LLM"], errors="ignore")
    write_df(invalid_out, OUT_INVALID_ROWS)

    return valid_df, invalid_df


def compute_and_save_confusion_and_metrics(paired: pd.DataFrame) -> None:
    """
    Compute confusion matrix + metrics for the concatenated (multi-year) data.
    """
    cm = pd.crosstab(
        index=pd.Categorical(paired["NIST"], categories=LABELS, ordered=True),
        columns=pd.Categorical(paired["LLM"], categories=LABELS, ordered=True),
        dropna=False,
    )
    cm.index.name = "NIST"
    cm.columns.name = "LLM"

    cm_pct = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) * 100.0

    mae = compute_mae(paired["NIST"], paired["LLM"])
    kappa_weighted = compute_weighted_kappa_ordinal(cm)

    alpha_4pt = compute_krippendorff_alpha_paired(
        paired["NIST"],
        paired["LLM"],
        level="ordinal",
    )

    paired_bin = paired.copy()
    paired_bin["NIST_bin"] = binarize_labels(paired_bin["NIST"])
    paired_bin["LLM_bin"]  = binarize_labels(paired_bin["LLM"])

    cm_bin = pd.crosstab(
        index=pd.Categorical(paired_bin["NIST_bin"], categories=[0, 1], ordered=True),
        columns=pd.Categorical(paired_bin["LLM_bin"],  categories=[0, 1], ordered=True),
        dropna=False,
    )

    kappa_binary = compute_unweighted_kappa(cm_bin)
    mae_binary = compute_mae(paired_bin["NIST_bin"], paired_bin["LLM_bin"])

    alpha_2pt = compute_krippendorff_alpha_paired(
        paired_bin["NIST_bin"],
        paired_bin["LLM_bin"],
        level="nominal",
    )

    write_confusion_outputs(cm, cm_pct, OUT_COUNTS, OUT_PCT)

    metrics_df = pd.DataFrame(
        [
            {"metric": "mae",                    "value": float(mae)},
            {"metric": "mae_binary_2pt",         "value": float(mae_binary)},
            {"metric": "kappa_weighted_4pt",     "value": float(kappa_weighted)},
            {"metric": "kappa_binary_2pt",       "value": float(kappa_binary)},
            {"metric": "alpha_4pt_ordinal",      "value": float(alpha_4pt)},
            {"metric": "alpha_binary_nominal_2pt","value": float(alpha_2pt)},
            {"metric": "pairs",                  "value": float(len(paired))},
        ]
    )
    write_metrics(metrics_df, OUT_DIR / "metrics_llm_vs_nist.csv")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cbar=True)
    plt.title(f"Confusion Matrix: NIST vs LLM — {MODEL} {YEAR_TAG} {LANG}")
    plt.ylabel("NIST label")
    plt.xlabel("LLM label")

    save_heatmap(plt, OUT_SVG, dpi=200, tight=True, show=True)


def main():
    df = load_and_prepare()

    paired, _invalid_df = split_valid_invalid(df)
    if paired.empty:
        raise RuntimeError(
            "No valid (NIST, LLM) label pairs found after filtering. "
            f"Check {OUT_INVALID_ROWS} for details."
        )

    compute_and_save_confusion_and_metrics(paired)


if __name__ == "__main__":
    main()
