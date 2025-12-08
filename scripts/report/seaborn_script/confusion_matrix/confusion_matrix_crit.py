#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
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
    binarize_labels,
)

# -------- Config --------
TREC_DL_YEAR = "2022"
MODEL = "gpt-oss-20b"  # e.g., "qwen3-32b-v1", "gpt-oss-20b", etc.
LANG  = "raw_crit_2"     # "raw","eng","vi","fr", etc.

# LLM CSV is assumed to contain:
#   - qid
#   - pid
#   - llm_relevance
LLM_FILE = (
    Path("outputs/llm_label")
    / f"trec_dl_{TREC_DL_YEAR}"
    / MODEL
    / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{LANG}_labels.csv"
)

# Directory containing NIST label CSVs with header:
# qid,query,pid,passage,relevance
# (Adjust this to your actual layout if needed.)
NIST_DIR = PROJECT_ROOT / "retrieved" / f"trec_dl_{TREC_DL_YEAR}" / "judged"

OUT_DIR          = Path("outputs/baseline") / TREC_DL_YEAR / MODEL / LANG
OUT_COUNTS       = OUT_DIR / "confusion_matrix_llm_vs_nist.csv"
OUT_PCT          = OUT_DIR / "confusion_matrix_llm_vs_nist_pct.csv"
OUT_SVG          = OUT_DIR / "confusion_matrix_llm_vs_nist.svg"
OUT_INVALID_ROWS = OUT_DIR / "rows_with_missing_or_invalid_labels.csv"

LABELS = [0, 1, 2, 3]


def load_nist_labels() -> pd.DataFrame:
    """
    Load all NIST CSVs from NIST_DIR.

    Each file is expected to have at least:
        qid, query, pid, passage, relevance

    We return a DataFrame with:
        qid, pid, NIST
    """
    if not NIST_DIR.exists():
        raise FileNotFoundError(
            f"NIST_DIR does not exist: {NIST_DIR}. "
            "Update NIST_DIR in this script to point to your NIST/qrels CSV files."
        )

    frames: list[pd.DataFrame] = []

    # Only look at CSVs; tweak glob if your files differ
    for path in sorted(NIST_DIR.glob("*.csv")):
        if not path.is_file():
            continue

        df_raw = pd.read_csv(path)

        required_cols = {"qid", "pid", "relevance"}
        missing = required_cols - set(df_raw.columns)
        if missing:
            raise ValueError(
                f"NIST file {path} is missing required columns {sorted(missing)}. "
                f"Found columns: {list(df_raw.columns)}"
            )

        df = df_raw[["qid", "pid", "relevance"]].copy()
        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No usable NIST CSV files found in {NIST_DIR}. "
            "Check that your NIST files are there and named *.csv"
        )

    nist_all = pd.concat(frames, ignore_index=True)

    # Convert 'relevance' to numeric and rename to NIST
    nist_all["NIST"] = pd.to_numeric(nist_all["relevance"], errors="coerce")
    nist_all = nist_all.drop(columns=["relevance"])

    # Drop duplicate (qid, pid) pairs if any
    nist_all = nist_all.drop_duplicates(subset=["qid", "pid"], keep="last")

    return nist_all  # columns: qid, pid, NIST


def load_and_prepare() -> pd.DataFrame:
    """
    Load the LLM CSV (with qid, pid, llm_relevance),
    join with NIST labels on (qid, pid), and standardize to NIST + LLM columns.
    """
    bump_field_limit()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load LLM labels
    df_llm = pd.read_csv(LLM_FILE)

    required_cols = {"qid", "pid", "llm_relevance"}
    missing = required_cols - set(df_llm.columns)
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required_cols)} in {LLM_FILE}, "
            f"but missing: {sorted(missing)}. Got columns: {list(df_llm.columns)}"
        )

    # Load NIST labels from official files
    nist_df = load_nist_labels()

    # Merge on (qid, pid)
    df = df_llm.merge(
        nist_df,
        on=["qid", "pid"],
        how="left",
        validate="m:1",  # each (qid,pid) from LLM maps to at most one NIST row
    )

    # Coerce to numeric; invalid parses become NaN
    df["NIST"] = pd.to_numeric(df["NIST"], errors="coerce")
    df["LLM"]  = pd.to_numeric(df["llm_relevance"], errors="coerce")

    return df


def split_valid_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate rows with valid labels from rows with missing/invalid labels.

    Valid rows:
        - NIST and LLM are not NaN
        - And lie within LABELS
    Everything else is treated as invalid and written to a separate CSV.
    """
    valid_mask = df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)

    valid_df   = df[valid_mask].copy()
    invalid_df = df[~valid_mask].copy()

    # ----- Missing / invalid checking section -----
    if not invalid_df.empty:
        write_df(invalid_df, OUT_INVALID_ROWS)

    return valid_df, invalid_df


def compute_and_save_confusion_and_metrics(paired: pd.DataFrame) -> None:
    """
    Given a DataFrame with columns NIST and LLM (already validated),
    compute the confusion matrix and metrics, then write outputs + heatmap.
    """
    # 1) Confusion matrix (4-point)
    cm = pd.crosstab(
        index=pd.Categorical(paired["NIST"], categories=LABELS, ordered=True),
        columns=pd.Categorical(paired["LLM"], categories=LABELS, ordered=True),
        dropna=False,
    )
    cm.index.name = "NIST"
    cm.columns.name = "LLM"

    cm_pct = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) * 100.0

    # 2) Metrics
    mae = compute_mae(paired["NIST"], paired["LLM"])
    kappa_weighted = compute_weighted_kappa_ordinal(cm)

    # Binary version: collapse labels into [0, 1]
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

    # 3) Write outputs
    write_confusion_outputs(cm, cm_pct, OUT_COUNTS, OUT_PCT)

    metrics_df = pd.DataFrame(
        [
            {"metric": "mae",                "value": float(mae)},
            {"metric": "mae_binary_2pt",     "value": float(mae_binary)},
            {"metric": "kappa_weighted_4pt", "value": float(kappa_weighted)},
            {"metric": "kappa_binary_2pt",   "value": float(kappa_binary)},
            {"metric": "pairs",              "value": float(len(paired))},
        ]
    )
    write_metrics(metrics_df, OUT_DIR / "metrics_llm_vs_nist.csv")

    # 4) Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cbar=True)
    plt.title(f"Confusion Matrix: NIST vs LLM — {MODEL} {TREC_DL_YEAR} {LANG}")
    plt.ylabel("NIST label")
    plt.xlabel("LLM label")

    save_heatmap(plt, OUT_SVG, dpi=200, tight=True, show=True)


def main():
    # Load + standardize (using NIST files for NIST labels)
    df = load_and_prepare()

    # ---- Section: missing/invalid label checking ----
    paired, invalid_df = split_valid_invalid(df)

    # ---- Section: confusion matrix + metrics ----
    if paired.empty:
        raise RuntimeError(
            "No valid (NIST, LLM) label pairs found after filtering. "
            f"Check {OUT_INVALID_ROWS} for details."
        )

    compute_and_save_confusion_and_metrics(paired)


if __name__ == "__main__":
    main()
