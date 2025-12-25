#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
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
MODEL = "qwen3-32b-v1"  # e.g., "qwen3-32b-v1", "gpt-oss-20b", etc.

# Two variants / filenames to compare (the middle part in your naming convention)
# Example: "raw", "eng", "vi", "ga", "ga_word", "vi_corrected", etc.
VAR_A = "vi"
VAR_B = "vi_2"

LABELS = [0, 1, 2, 3]

# Output naming: "2021_2022"
YEAR_TAG = "_".join(TREC_DL_YEARS)

OUT_DIR = Path("figures") / YEAR_TAG / MODEL / "confusion_matrix" / f"{VAR_A}_vs_{VAR_B}"
OUT_COUNTS       = OUT_DIR / "confusion_matrix_llmA_vs_llmB.csv"
OUT_PCT          = OUT_DIR / "confusion_matrix_llmA_vs_llmB_pct.csv"
OUT_SVG          = OUT_DIR / "confusion_matrix_llmA_vs_llmB.svg"
OUT_INVALID_ROWS = OUT_DIR / "rows_with_missing_or_invalid_labels.csv"
OUT_UNMATCHED    = OUT_DIR / "rows_unmatched_between_files.csv"


def _llm_file_for_year(year: str, variant: str) -> Path:
    return (
        Path("outputs/llm_label")
        / f"trec_dl_{year}"
        / MODEL
        / f"{MODEL}_trecdl_{year}_{variant}_labels.csv"
    )


def _infer_merge_keys(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    """
    Prefer stable IDs if present.
    Falls back to query/passage if no ids exist (less ideal but workable).
    """
    preferred = [
        ["qid", "pid"],
        ["query_id", "pid"],
        ["qid", "passage_id"],
        ["query_id", "passage_id"],
    ]
    for keys in preferred:
        if all(k in df_a.columns for k in keys) and all(k in df_b.columns for k in keys):
            return keys

    fallback = ["query", "passage"]
    if all(k in df_a.columns for k in fallback) and all(k in df_b.columns for k in fallback):
        return fallback

    # last resort: intersecting columns that look like ids
    candidates = [c for c in ["qid", "query_id", "pid", "passage_id"] if c in df_a.columns and c in df_b.columns]
    if len(candidates) >= 2:
        return candidates[:2]

    raise ValueError(
        "Could not infer merge keys. Please ensure both files share identifiers "
        "like (qid,pid) or at least (query,passage)."
    )


def load_and_prepare() -> pd.DataFrame:
    """
    Load CSVs for VAR_A and VAR_B for all years and pair rows by a merge key.
    Produces a single DataFrame with:
      - LLM_A, LLM_B (numeric)
      - trec_dl_year provenance
      - original columns from both sides (with suffixes)
    """
    bump_field_limit()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    unmatched_frames: List[pd.DataFrame] = []

    for year in TREC_DL_YEARS:
        f_a = _llm_file_for_year(year, VAR_A)
        f_b = _llm_file_for_year(year, VAR_B)

        if not f_a.exists():
            raise FileNotFoundError(f"Missing file A for year {year}: {f_a}")
        if not f_b.exists():
            raise FileNotFoundError(f"Missing file B for year {year}: {f_b}")

        df_a = pd.read_csv(f_a)
        df_b = pd.read_csv(f_b)

        if "llm_relevance" not in df_a.columns:
            raise ValueError(f"Expected column 'llm_relevance' in {f_a}, got: {list(df_a.columns)}")
        if "llm_relevance" not in df_b.columns:
            raise ValueError(f"Expected column 'llm_relevance' in {f_b}, got: {list(df_b.columns)}")

        merge_keys = _infer_merge_keys(df_a, df_b)

        # Keep provenance + reduce chance of accidental duplicate column collisions
        df_a = df_a.copy()
        df_b = df_b.copy()
        df_a["trec_dl_year"] = year
        df_b["trec_dl_year"] = year

        # Merge within year (must match same qid/pid etc.)
        merged = df_a.merge(
            df_b,
            on=(merge_keys + ["trec_dl_year"]),
            how="outer",
            suffixes=("_A", "_B"),
            indicator=True,
        )

        # Save unmatched rows for debugging
        unmatched = merged[merged["_merge"] != "both"].copy()
        if not unmatched.empty:
            unmatched_frames.append(unmatched)

        # Keep only matched pairs
        paired = merged[merged["_merge"] == "both"].copy()
        paired.drop(columns=["_merge"], inplace=True)

        # Coerce to numeric
        paired["LLM_A"] = pd.to_numeric(paired["llm_relevance_A"], errors="coerce")
        paired["LLM_B"] = pd.to_numeric(paired["llm_relevance_B"], errors="coerce")

        frames.append(paired)

    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Write unmatched if any
    if unmatched_frames:
        df_unmatched = pd.concat(unmatched_frames, ignore_index=True)
        write_df(df_unmatched, OUT_UNMATCHED)

    return df_all


def split_valid_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Valid rows:
        - LLM_A and LLM_B in LABELS
    Everything else is invalid and written to a separate CSV.
    """
    valid_mask = df["LLM_A"].isin(LABELS) & df["LLM_B"].isin(LABELS)

    valid_df   = df[valid_mask].copy()
    invalid_df = df[~valid_mask].copy()

    print("Total paired rows:", len(df))
    print("Valid rows:", valid_mask.sum())
    print("Invalid rows:", (~valid_mask).sum())

    # Keep the original data for debugging, drop computed columns if you want
    invalid_out = invalid_df.drop(columns=["LLM_A", "LLM_B"], errors="ignore")
    write_df(invalid_out, OUT_INVALID_ROWS)

    return valid_df, invalid_df


def latex_metrics_row(
    mae_4pt: float,
    mae_2pt: float,
    kappa_4pt: float,
    kappa_2pt: float,
) -> str:
    return (
        f"& \\num{{{mae_4pt}}}  %MAE_4pt\n"
        f"& \\num{{{mae_2pt}}} %MAE_2pt\n"
        f"& \\num{{{kappa_4pt}}}   %kappa_4pt\n"
        f"& \\num{{{kappa_2pt}}} \\\\ %kappa_2pt \n"
    )


def compute_and_save_confusion_and_metrics(paired: pd.DataFrame) -> None:
    """
    Compute confusion matrix + metrics for LLM_A vs LLM_B.
    Treat A as the "reference" axis (rows), B as "predicted" axis (cols).
    """
    cm = pd.crosstab(
        index=pd.Categorical(paired["LLM_A"], categories=LABELS, ordered=True),
        columns=pd.Categorical(paired["LLM_B"], categories=LABELS, ordered=True),
        dropna=False,
    )
    cm.index.name = f"LLM_A ({VAR_A})"
    cm.columns.name = f"LLM_B ({VAR_B})"

    cm_pct = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) * 100.0

    mae_4pt = compute_mae(paired["LLM_A"], paired["LLM_B"])
    kappa_4pt_weighted = compute_weighted_kappa_ordinal(cm)

    alpha_4pt = compute_krippendorff_alpha_paired(
        paired["LLM_A"],
        paired["LLM_B"],
        level="ordinal",
    )

    paired_bin = paired.copy()
    paired_bin["A_bin"] = binarize_labels(paired_bin["LLM_A"])
    paired_bin["B_bin"] = binarize_labels(paired_bin["LLM_B"])

    cm_bin = pd.crosstab(
        index=pd.Categorical(paired_bin["A_bin"], categories=[0, 1], ordered=True),
        columns=pd.Categorical(paired_bin["B_bin"], categories=[0, 1], ordered=True),
        dropna=False,
    )

    kappa_2pt = compute_unweighted_kappa(cm_bin)
    mae_2pt = compute_mae(paired_bin["A_bin"], paired_bin["B_bin"])

    alpha_2pt = compute_krippendorff_alpha_paired(
        paired_bin["A_bin"],
        paired_bin["B_bin"],
        level="nominal",
    )

    write_confusion_outputs(cm, cm_pct, OUT_COUNTS, OUT_PCT)

    metrics_df = pd.DataFrame(
        [
            {"metric": "mae_4pt",                   "value": float(mae_4pt)},
            {"metric": "mae_binary_2pt",            "value": float(mae_2pt)},
            {"metric": "kappa_weighted_4pt",        "value": float(kappa_4pt_weighted)},
            {"metric": "kappa_binary_2pt",          "value": float(kappa_2pt)},
            {"metric": "alpha_4pt_ordinal",         "value": float(alpha_4pt)},
            {"metric": "alpha_binary_nominal_2pt",  "value": float(alpha_2pt)},
            {"metric": "pairs",                     "value": float(len(paired))},
        ]
    )
    write_metrics(metrics_df, OUT_DIR / "metrics_llmA_vs_llmB.csv")

    latex_row = latex_metrics_row(
        mae_4pt=float(mae_4pt),
        mae_2pt=float(mae_2pt),
        kappa_4pt=float(kappa_4pt_weighted),
        kappa_2pt=float(kappa_2pt),
    )
    print("\n[LaTeX row]\n" + latex_row)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cbar=True)
    plt.title(f"Confusion Matrix: {MODEL} {YEAR_TAG} — {VAR_A} vs {VAR_B}")
    plt.ylabel(f"LLM label ({VAR_A})")
    plt.xlabel(f"LLM label ({VAR_B})")

    save_heatmap(plt, OUT_SVG, dpi=200, tight=True, show=True)


def main():
    df = load_and_prepare()
    if df.empty:
        raise RuntimeError("No paired rows loaded. Check filenames and merge keys.")

    paired, _invalid_df = split_valid_invalid(df)
    if paired.empty:
        raise RuntimeError(
            "No valid (LLM_A, LLM_B) label pairs found after filtering. "
            f"Check {OUT_INVALID_ROWS} for details."
        )

    compute_and_save_confusion_and_metrics(paired)


if __name__ == "__main__":
    main()
