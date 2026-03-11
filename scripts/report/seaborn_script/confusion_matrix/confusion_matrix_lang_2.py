#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# =========================
# Path setup (MUST be before helpers imports)
# =========================
THIS_FILE = Path(__file__).resolve()

# /scripts/report/seaborn_script
SEABORN_SCRIPT_DIR = THIS_FILE.parents[1]

# repo root (same as your current pattern)
PROJECT_ROOT = THIS_FILE.parents[4]

# Ensure global seaborn_script/helpers wins over local subfolder helpers
if str(SEABORN_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SEABORN_SCRIPT_DIR))

# Keep repo root for scripts.* imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================
# Imports
# =========================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


from helpers.draw import apply_paper_fmt
# -------- Config --------
TREC_DL_YEAR = "2021"
MODEL = "llama3-8b-instruct"
LANG = "ko"

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
    print("Valid rows:", int(valid_mask.sum()))
    print("Invalid rows:", int((~valid_mask).sum()))

    # Keep your existing debug dump: show why invalid
    debug_cols: list[str] = []
    for c in ["qid", "pid", "relevance", "llm_relevance", "NIST", "LLM"]:
        if c in invalid_df.columns:
            debug_cols.append(c)

    invalid_dbg = invalid_df.copy()
    invalid_dbg["reason"] = ""
    invalid_dbg.loc[invalid_dbg["NIST"].isna(), "reason"] += "NIST_NaN;"
    invalid_dbg.loc[
        ~invalid_dbg["NIST"].isin(LABELS) & invalid_dbg["NIST"].notna(),
        "reason",
    ] += "NIST_out_of_range;"
    invalid_dbg.loc[invalid_dbg["LLM"].isna(), "reason"] += "LLM_NaN;"
    invalid_dbg.loc[
        ~invalid_dbg["LLM"].isin(LABELS) & invalid_dbg["LLM"].notna(),
        "reason",
    ] += "LLM_out_of_range;"

    write_df(invalid_dbg[debug_cols + ["reason"]], OUT_INVALID_ROWS)

    # Write ONLY rows where llm_relevance is NaN after coercion ("LLM" column)
    OUT_MISSING_PART0.parent.mkdir(parents=True, exist_ok=True)
    missing_llm_df = df[df["LLM"].isna()].copy()

    missing_llm_df.drop(columns=["NIST", "LLM"], errors="ignore").to_csv(
        OUT_MISSING_PART0,
        index=False,
        encoding="utf-8",
    )
    print(
        f"[Missing->Part0] Wrote {len(missing_llm_df)} rows (LLM is NaN) to: {OUT_MISSING_PART0}"
    )

    return valid_df, invalid_df


def latex_metrics_row(
    mae_4pt: float,
    mae_2pt: float,
    kappa_4pt: float,
    kappa_2pt: float,
    alpha_4pt: float,
    alpha_2pt: float,
) -> str:
    return (
        f"& \\num{{{mae_4pt}}}  %MAE_4pt\n"
        f"& \\num{{{mae_2pt}}} %MAE_2pt\n"
        f"& \\num{{{kappa_4pt}}}   %kappa_4pt\n"
        f"& \\num{{{kappa_2pt}}}   %kappa_2pt\n"
        f"& \\num{{{alpha_4pt}}}   %alpha_4pt\n"
        f"& \\num{{{alpha_2pt}}} \\\\ %alpha_2pt\n"
    )


def save_latex_row(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def plot_confusion_heatmap(
    cm: pd.DataFrame,
    *,
    title: str,
    out_svg: Path,
    dpi: int = 200,
) -> None:
    """
    Confusion heatmap using repo style defaults.
    We call apply_paper_fmt() (from helpers.draw) once per plot to ensure
    consistent styling regardless of import order.
    """
    apply_paper_fmt()

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        linewidths=0.5,
        cbar=True,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylabel("NIST label")
    ax.set_xlabel("LLM label")

    save_heatmap(plt, out_svg, dpi=dpi, tight=True, show=True)
    plt.close(fig)


def compute_and_save_confusion_and_metrics(paired: pd.DataFrame) -> None:
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
    alpha_4pt = compute_krippendorff_alpha_paired(
        paired["NIST"],
        paired["LLM"],
        level="ordinal",
    )

    # Binary version
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
    alpha_2pt = compute_krippendorff_alpha_paired(
        paired_bin["NIST_bin"],
        paired_bin["LLM_bin"],
        level="nominal",
    )

    # 3) Write confusion outputs
    write_confusion_outputs(cm, cm_pct, OUT_COUNTS, OUT_PCT)

    # 4) Metrics CSV
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
        alpha_4pt=float(alpha_4pt),
        alpha_2pt=float(alpha_2pt),
    )
    print("\n[LaTeX row]\n" + latex_row)
    save_latex_row(latex_row, OUT_LATEX)

    # 6) Heatmap
    plot_confusion_heatmap(
        cm,
        title=f"Confusion Matrix: {MODEL} {TREC_DL_YEAR} {LANG}",
        out_svg=OUT_SVG,
        dpi=200,
    )


def main() -> None:
    # Apply style once at entry (draw.py -> settings.py), but we also re-apply in plots
    # to avoid surprises if other scripts modify rcParams.
    apply_paper_fmt()

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
