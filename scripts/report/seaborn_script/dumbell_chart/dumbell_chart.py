#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit

# -------- Config --------
TREC_DL_YEAR = "2022"
MODEL = "gpt-oss-20b"   # e.g., "qwen3-32b-v1", "gpt-oss-20b", etc.
LANG  = "raw_crit"      # "raw","eng","vi","fr", etc.

# LLM CSV is assumed to contain:
#   - qid
#   - pid
#   - llm_relevance   (this is the crit label for raw_crit)
LLM_FILE = (
    Path("outputs/llm_label")
    / f"trec_dl_{TREC_DL_YEAR}"
    / MODEL
    / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{LANG}_labels.csv"
)

# Directory containing NIST label CSVs with header:
# qid,query,pid,passage,relevance
NIST_DIR = PROJECT_ROOT / "retrieved" / f"trec_dl_{TREC_DL_YEAR}" / "judged"

# Where to save the dumbbell figure
FIG_DIR = PROJECT_ROOT / "figures" / "dumbbell"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = FIG_DIR / f"dumbbell_{MODEL}_{TREC_DL_YEAR}_{LANG}.svg"

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
            "Update NIST_DIR in this script to point to your NIST/judged CSV files."
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
    join with NIST labels on (qid, pid), and return only valid rows with:

        - NIST in LABELS
        - crit (LLM) in LABELS
        - at most FIRST 5 documents per qid
    """
    bump_field_limit()

    df_llm = pd.read_csv(LLM_FILE)

    required_cols = {"qid", "pid", "llm_relevance"}
    missing = required_cols - set(df_llm.columns)
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required_cols)} in {LLM_FILE}, "
            f"but missing: {sorted(missing)}. Got columns: {list(df_llm.columns)}"
        )

    nist_df = load_nist_labels()

    df = df_llm.merge(
        nist_df,
        on=["qid", "pid"],
        how="left",
        validate="m:1",
    )

    df["NIST"] = pd.to_numeric(df["NIST"], errors="coerce")
    df["crit"] = pd.to_numeric(df["llm_relevance"], errors="coerce")

    valid_mask = df["NIST"].isin(LABELS) & df["crit"].isin(LABELS)
    df_valid = df[valid_mask].copy()

    if df_valid.empty:
        raise RuntimeError("No valid rows with NIST and crit labels in LABELS.")

    # Order within each qid
    df_valid = df_valid.sort_values(
        by=["qid", "NIST", "crit", "pid"],
        ignore_index=True,
    )

    # Keep only the first 5 documents per qid
    df_valid = (
        df_valid
        .groupby("qid", as_index=False, group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )

    return df_valid


def plot_dumbbell(df: pd.DataFrame) -> None:
    """
    Dumbbell chart of NIST vs crit for each (qid, pid).

    - Blue dot: crit (raw_crit / LLM)
    - Red dot:  NIST
    - Line: green if crit - NIST > 0, else red.
    No y-axis labels; only dots and lines.
    """
    df = df.reset_index(drop=True)
    df["y"] = df.index

    n_docs = len(df)
    height = max(6, min(0.12 * n_docs, 40))

    fig, ax = plt.subplots(figsize=(8, height))

    for _, row in df.iterrows():
        y = row["y"]
        nist = row["NIST"]
        crit = row["crit"]

        diff = crit - nist
        line_color = "green" if diff > 0 else "red"

        ax.hlines(
            y=y,
            xmin=min(nist, crit),
            xmax=max(nist, crit),
            color=line_color,
            linewidth=2,
            alpha=0.9,
        )

        ax.scatter(nist, y, color="red",  s=40, zorder=3)
        ax.scatter(crit, y, color="blue", s=40, zorder=3)

    # No y labels, no ticks
    ax.set_yticks([])
    ax.set_ylabel("Documents")

    ax.set_xticks(LABELS)
    ax.set_xlim(min(LABELS) - 0.5, max(LABELS) + 0.5)
    ax.set_xlabel("Relevance label (0–3)")
    ax.set_title(f"NIST vs {LANG} dumbbell — {MODEL} {TREC_DL_YEAR}")

    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.grid(axis="y", visible=False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="NIST",
               markerfacecolor="red", markersize=7),
        Line2D([0], [0], marker="o", color="w", label=LANG,
               markerfacecolor="blue", markersize=7),
        Line2D([0], [0], color="green", lw=2, label="crit > NIST"),
        Line2D([0], [0], color="red",   lw=2, label="crit ≤ NIST"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=300)
    print(f"Saved dumbbell chart to: {FIG_PATH}")


def main():
    df_valid = load_and_prepare()
    plot_dumbbell(df_valid)


if __name__ == "__main__":
    main()
