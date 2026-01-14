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
    compute_krippendorff_alpha_paired,
    binarize_labels,
)

# -------- Config --------
TREC_DL_YEAR = "2021"
MODEL = "gpt-oss-20b"  # e.g., "qwen3-32b-v1", "gpt-oss-20b", etc.
LANG = "th_first" # "raw","eng","vi","fr", etc.

# This CSV is now assumed to already contain:
#   - relevance
#   - llm_relevance
LLM_FILE = (
    Path("outputs/llm_label")
    / f"trec_dl_{TREC_DL_YEAR}"
    / MODEL
    / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{LANG}_labels.csv"
)

OUT_DIR = Path("figures") / TREC_DL_YEAR / MODEL / "confusion_matrix" / LANG
OUT_COUNTS = OUT_DIR / "confusion_matrix_llm_vs_nist.csv"
OUT_PCT = OUT_DIR / "confusion_matrix_llm_vs_nist_pct.csv"
OUT_SVG = OUT_DIR / "confusion_matrix_llm_vs_nist.svg"
OUT_INVALID_ROWS = OUT_DIR / "rows_with_missing_or_invalid_labels.csv"
OUT_LATEX = OUT_DIR / "metrics_llm_vs_nist_row.tex"

LABELS = [0, 1, 2, 3]

# how many example disagreements to print (kept as-is; your function prints 1 per bucket)
DISAGREE_EXAMPLES = 1

OUT_MISSING_PART0 = (
    Path("retrieved")
    / f"trec_dl_{TREC_DL_YEAR}"
    / LANG
    / f"all_topics_trecdl_{TREC_DL_YEAR}_part0.csv"
)


def load_and_prepare() -> pd.DataFrame:
    """
    Load the combined CSV that already contains both relevance and llm_relevance.
    Coerce them to numeric and standardize column names (NIST, LLM).
    """
    bump_field_limit()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LLM_FILE)

    if "relevance" not in df.columns or "llm_relevance" not in df.columns:
        raise ValueError(
            f"Expected columns 'relevance' and 'llm_relevance' in {LLM_FILE}, "
            f"but got: {list(df.columns)}"
        )

    # Coerce to numeric (in case they are strings); invalid parses become NaN
    df["NIST"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["LLM"] = pd.to_numeric(df["llm_relevance"], errors="coerce")

    return df


def split_valid_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_mask = df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)

    valid_df = df[valid_mask].copy()
    invalid_df = df[~valid_mask].copy()

    print("Total rows:", len(df))
    print("Valid rows:", valid_mask.sum())
    print("Invalid rows:", (~valid_mask).sum())

    # Keep your existing debug dump (optional)
    # Keep debug columns so we can see WHY it was marked invalid
    debug_cols = []
    for c in ["qid", "pid", "relevance", "llm_relevance", "NIST", "LLM"]:
        if c in invalid_df.columns:
            debug_cols.append(c)

    # Add a reason column to make it obvious
    invalid_dbg = invalid_df.copy()
    invalid_dbg["reason"] = ""
    invalid_dbg.loc[invalid_dbg["NIST"].isna(), "reason"] += "NIST_NaN;"
    invalid_dbg.loc[~invalid_dbg["NIST"].isin(LABELS) & invalid_dbg["NIST"].notna(), "reason"] += "NIST_out_of_range;"
    invalid_dbg.loc[invalid_dbg["LLM"].isna(), "reason"] += "LLM_NaN;"
    invalid_dbg.loc[~invalid_dbg["LLM"].isin(LABELS) & invalid_dbg["LLM"].notna(), "reason"] += "LLM_out_of_range;"

    write_df(invalid_dbg[debug_cols + ["reason"]], OUT_INVALID_ROWS)


    # Write ALL invalid rows as "part0" exactly as-is (no renaming, no new cols)
    # Write ONLY rows where llm_relevance is NaN after coercion ("LLM" column)
    OUT_MISSING_PART0.parent.mkdir(parents=True, exist_ok=True)

    missing_llm_df = df[df["LLM"].isna()].copy()

    missing_llm_df.drop(columns=["NIST", "LLM"], errors="ignore").to_csv(
        OUT_MISSING_PART0,
        index=False,
        encoding="utf-8",
    )
    print(f"[Missing->Part0] Wrote {len(missing_llm_df)} rows (LLM is NaN) to: {OUT_MISSING_PART0}")


    return valid_df, invalid_df


def latex_metrics_row(
    mae_4pt: float,
    mae_2pt: float,
    kappa_4pt: float,
    kappa_2pt: float,
) -> str:
    """
    Return a LaTeX table row fragment in exactly this style:

    & \\num{...}  %MAE_4pt
    & \\num{...}  %MAE_2pt
    & \\num{...}  %kappa_4pt
    & \\num{...}  %kappa_2pt \\\\
    """
    # Keep full precision (like your example). If you want rounding, change here.
    return (
        f"& \\num{{{mae_4pt}}}  %MAE_4pt\n"
        f"& \\num{{{mae_2pt}}} %MAE_2pt\n"
        f"& \\num{{{kappa_4pt}}}   %kappa_4pt\n"
        f"& \\num{{{kappa_2pt}}} \\\\ %kappa_2pt \n"
    )


def save_latex_row(text: str, path: Path) -> None:
    """Write LaTeX row fragment to disk (overwrites)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def compute_and_save_confusion_and_metrics(paired: pd.DataFrame) -> None:
    """
    Given a DataFrame with columns NIST and LLM (already validated),
    compute the confusion matrix and metrics, then write outputs + heatmap + LaTeX row.
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

    # 2) Metrics (4-point / graded)
    mae_4pt = compute_mae(paired["NIST"], paired["LLM"])
    kappa_4pt_weighted = compute_weighted_kappa_ordinal(cm)

    # Krippendorff's alpha (4-point, ordinal) (kept for CSV outputs)
    alpha_4pt = compute_krippendorff_alpha_paired(
        paired["NIST"],
        paired["LLM"],
        level="ordinal",
    )

    # Binary version: collapse labels into [0, 1]
    paired_bin = paired.copy()
    paired_bin["NIST_bin"] = binarize_labels(paired_bin["NIST"])
    paired_bin["LLM_bin"] = binarize_labels(paired_bin["LLM"])

    cm_bin = pd.crosstab(
        index=pd.Categorical(paired_bin["NIST_bin"], categories=[0, 1], ordered=True),
        columns=pd.Categorical(paired_bin["LLM_bin"], categories=[0, 1], ordered=True),
        dropna=False,
    )

    kappa_2pt = compute_unweighted_kappa(cm_bin)
    mae_2pt = compute_mae(paired_bin["NIST_bin"], paired_bin["LLM_bin"])

    # Krippendorff's alpha (binary, treat as nominal) (kept for CSV outputs)
    alpha_2pt = compute_krippendorff_alpha_paired(
        paired_bin["NIST_bin"],
        paired_bin["LLM_bin"],
        level="nominal",
    )

    # 3) Write confusion outputs
    write_confusion_outputs(cm, cm_pct, OUT_COUNTS, OUT_PCT)

    # 4) Metrics CSV (kept)
    metrics_df = pd.DataFrame(
        [
            {"metric": "mae_4pt", "value": float(mae_4pt)},
            {"metric": "mae_binary_2pt", "value": float(mae_2pt)},
            {"metric": "kappa_weighted_4pt", "value": float(kappa_4pt_weighted)},
            {"metric": "kappa_binary_2pt", "value": float(kappa_2pt)},
            {"metric": "alpha_ordinal_4pt", "value": float(alpha_4pt)},
            {"metric": "alpha_binary_nominal_2pt", "value": float(alpha_2pt)},
            {"metric": "pairs", "value": float(len(paired))},
        ]
    )
    write_metrics(metrics_df, OUT_DIR / "metrics_llm_vs_nist.csv")

    # 5) LaTeX row fragment
    latex_row = latex_metrics_row(
        mae_4pt=float(mae_4pt),
        mae_2pt=float(mae_2pt),
        kappa_4pt=float(kappa_4pt_weighted),
        kappa_2pt=float(kappa_2pt),
    )
    print("\n[LaTeX row]\n" + latex_row)
    save_latex_row(latex_row, OUT_LATEX)

    # 6) Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=0.5, cbar=True)
    plt.title(f"Confusion Matrix: NIST vs LLM — {MODEL} {TREC_DL_YEAR} {LANG}")
    plt.ylabel("NIST label")
    plt.xlabel("LLM label")

    save_heatmap(plt, OUT_SVG, dpi=200, tight=True, show=True)


def print_false_positive_examples(paired: pd.DataFrame) -> None:
    """
    Print 1 example for each case where:
      - NIST (relevance) == 0
      - LLM (llm_relevance) == 1, 2, or 3
    Uses only valid pairs (paired).
    """
    fp = paired[(paired["NIST"] == 0) & (paired["LLM"].isin([1, 2, 3]))].copy()

    if fp.empty:
        print("[FP] No false positives found (NIST==0 but LLM in {1,2,3}).")
        return

    cols_prefer = [
        "qid",
        "docid",
        "pid",
        "query_id",
        "passage_id",
        "relevance",
        "llm_relevance",
        "NIST",
        "LLM",
        "query",
        "passage",
    ]

    for llm_label in [1, 2, 3]:
        bucket = fp[fp["LLM"] == llm_label]
        if bucket.empty:
            print(f"[FP] No examples where NIST=0 and LLM={llm_label}.")
            continue

        row = bucket.sample(n=1, random_state=42).iloc[0]
        cols = [c for c in cols_prefer if c in bucket.columns]

        print(f"[FP] Example where NIST=0 and LLM={llm_label}:")
        if cols:
            kv = ", ".join([f"{c}={repr(row[c])}" for c in cols])
            print("  -", kv)
        else:
            print("  -", row.to_dict())


def main() -> None:
    df = load_and_prepare()
    paired, _invalid_df = split_valid_invalid(df)

    # print_false_positive_examples(paired)

    if paired.empty:
        raise RuntimeError(
            "No valid (NIST, LLM) label pairs found after filtering. "
            f"Check {OUT_INVALID_ROWS} for details."
        )

    compute_and_save_confusion_and_metrics(paired)


if __name__ == "__main__":
    main()
