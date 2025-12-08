#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Project root setup ----------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit
from helpers.output_writer import write_df, save_heatmap

# ---------- Config ----------
TREC_DL_YEAR = "2022"
MODEL        = "gpt-oss-20b"
LANG         = "raw_crit_2"

# LLM + criterion file (your raw_crit_2 CSV)
CRIT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "llm_label"
    / f"trec_dl_{TREC_DL_YEAR}"
    / MODEL
    / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{LANG}_labels.csv"
)

# NIST judged folder (same structure as your confusion-matrix script)
NIST_DIR = PROJECT_ROOT / "retrieved" / f"trec_dl_{TREC_DL_YEAR}" / "judged"

# Short IDs used in the figure:
# T = Topicality, F = Contextual Fit, E = Exactness, C = Coverage,
# J = Ground-truth relevance, L = LLM relevance
COL_MAP = {
    "T": "topicality",
    "F": "contextuality",
    "E": "exactness",
    "C": "coverage",
    "J": "NIST",           # will be created after merging
    "L": "llm_relevance",
}

# Output locations
OUT_DIR      = PROJECT_ROOT / "figures" / "criterion_corr" / TREC_DL_YEAR / MODEL / LANG
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CORR_CSV = OUT_DIR / "criterion_grade_correlation.csv"
OUT_FIG_SVG  = OUT_DIR / "criterion_grade_correlation.svg"


# ---------- Helpers to load NIST ----------
def load_nist_labels() -> pd.DataFrame:
    """
    Load all NIST CSVs from NIST_DIR, return columns: qid, pid, NIST
    """
    if not NIST_DIR.exists():
        raise FileNotFoundError(
            f"NIST_DIR does not exist: {NIST_DIR}"
        )

    frames: list[pd.DataFrame] = []
    for path in sorted(NIST_DIR.glob("*.csv")):
        if not path.is_file():
            continue

        df_raw = pd.read_csv(path)
        required_cols = {"qid", "pid", "relevance"}
        missing = required_cols - set(df_raw.columns)
        if missing:
            raise ValueError(
                f"NIST file {path} missing {sorted(missing)}. "
                f"Got columns: {list(df_raw.columns)}"
            )

        df = df_raw[["qid", "pid", "relevance"]].copy()
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No NIST CSV files found in {NIST_DIR}")

    nist_all = pd.concat(frames, ignore_index=True)

    nist_all["NIST"] = pd.to_numeric(nist_all["relevance"], errors="coerce")
    nist_all = nist_all.drop(columns=["relevance"])
    nist_all = nist_all.drop_duplicates(subset=["qid", "pid"], keep="last")

    return nist_all  # qid, pid, NIST


def load_and_prepare() -> pd.DataFrame:
    """
    Load raw_crit_2 CSV + NIST and return a DataFrame with:
    qid, pid, contextuality, coverage, exactness, topicality,
    llm_relevance, NIST
    """
    bump_field_limit()

    if not CRIT_FILE.exists():
        raise FileNotFoundError(f"CRIT_FILE not found: {CRIT_FILE}")

    df_llm = pd.read_csv(CRIT_FILE)

    required_cols = {
        "qid",
        "pid",
        "contextuality",
        "coverage",
        "exactness",
        "topicality",
        "llm_relevance",
    }
    missing = required_cols - set(df_llm.columns)
    if missing:
        raise ValueError(
            f"raw_crit_2 file missing columns {sorted(missing)}. "
            f"Got: {list(df_llm.columns)}"
        )

    nist_df = load_nist_labels()

    df = df_llm.merge(
        nist_df,
        on=["qid", "pid"],
        how="left",
        validate="m:1",
    )

    # Coerce all relevant columns to numeric 0–3
    for short, col in COL_MAP.items():
        if col == "NIST":
            continue  # handled by NIST loader
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["NIST"] = pd.to_numeric(df["NIST"], errors="coerce")

    # Keep only rows where all variables are in {0,1,2,3}
    from numpy import isin

    def valid_0_3(series: pd.Series) -> pd.Series:
        return series.isin([0, 1, 2, 3])

    mask = (
        valid_0_3(df["topicality"])
        & valid_0_3(df["contextuality"])
        & valid_0_3(df["exactness"])
        & valid_0_3(df["coverage"])
        & valid_0_3(df["llm_relevance"])
        & valid_0_3(df["NIST"])
    )
    df_valid = df[mask].copy()

    return df_valid


# ---------- Indicator + correlation ----------
def build_indicator_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each variable (T,F,E,C,J,L) and grade (3,2,1,0) create a binary column:

        T3, F3, E3, C3, J3, L3, T2, ..., L0

    matching the order in the paper figure.
    """
    grades = [3, 2, 1, 0]
    short_order = ["T", "F", "E", "C", "J", "L"]

    indicator_cols = []
    out = pd.DataFrame(index=df.index)

    for g in grades:
        for short in short_order:
            col_name = COL_MAP[short]
            new_col = f"{short}{g}"
            out[new_col] = (df[col_name] == g).astype(int)
            indicator_cols.append(new_col)

    return out[indicator_cols]


def compute_and_plot_corr(indicators: pd.DataFrame) -> None:
    corr = indicators.corr()

    # Save numeric correlation matrix
    write_df(corr, OUT_CORR_CSV)

    # Plot heatmap
    plt.figure(figsize=(8, 10))
    ax = sns.heatmap(
        corr,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar=True,
        cbar_kws={"label": "Pearson correlation"},
        annot_kws={"size": 6},   # <-- smaller numbers inside cells
    )

    # Smaller tick-label fonts
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

    # Smaller colorbar tick labels
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)

    # Slightly smaller title font too, if you like
    ax.set_title(
        f"Criterion Grade-Level Correlations with Relevance Labels\n"
        f"{MODEL}, TREC-DL {TREC_DL_YEAR}, {LANG}",
        fontsize=8,
    )

    save_heatmap(plt, OUT_FIG_SVG, dpi=300, tight=True, show=True)

def main():
    df = load_and_prepare()
    indicators = build_indicator_matrix(df)
    compute_and_plot_corr(indicators)


if __name__ == "__main__":
    main()
