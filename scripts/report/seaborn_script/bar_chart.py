#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]   # adjust if script lives elsewhere

INPUT_CSV  = PROJECT_ROOT / "outputs" / "llm_label" / "gpt-oss-20b" / "gpt-oss-20b_trec_dl_2023_leet.csv"
LABEL_COL  = "llm_relevance"   # change if your column uses a different name

OUTPUT_NAME = INPUT_CSV.stem.split("_")[-1]
OUTPUT_PNG = PROJECT_ROOT / "figures" / f"llm_relevance_bar_{OUTPUT_NAME}.png"

# one color per label (0,1,2,3)
palette = {
    "0": "#1f77b4",  # blue
    "1": "#ff7f0e",  # orange
    "2": "#2ca02c",  # green
    "3": "#d62728",  # red
}

def main() -> None:
    print(f"[INFO] Reading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Column {LABEL_COL!r} not found in {INPUT_CSV}")

    # keep only labels 0–3, drop NaNs
    labels = df[LABEL_COL].dropna().astype(int)
    labels = labels[labels.isin([0, 1, 2, 3])]

    counts = labels.value_counts().reindex([0, 1, 2, 3], fill_value=0)
    plot_df = counts.reset_index()
    plot_df.columns = ["label", "count"]

    print("[INFO] Counts by label:")
    print(plot_df)

    # =========================
    # Plot
    # =========================
    sns.set(style="whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=plot_df, x="label", y="count", palette=palette)
    ax.set_xlabel("LLM relevance label")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of LLM relevance labels (0–3)")

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300)
    print(f"[INFO] Saved figure to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
