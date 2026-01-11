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

YEAR = "2022"
LANGS = ["eng", "eng_word"]
INPUT_DIR = PROJECT_ROOT / "outputs"

SUFFIX = "_vs_".join(LANGS)

OUTPUT_PNG = PROJECT_ROOT / "figures" / f"perplexity_delta_mean_{YEAR}_.pdf"
DELTA_COL = "perplexity_delta"


def load_mean_delta(lang: str) -> float:
    csv_path = INPUT_DIR / f"textdescriptives_perplexity_{YEAR}_{lang}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV for {lang}: {csv_path}")

    df = pd.read_csv(csv_path)
    if DELTA_COL not in df.columns:
        raise ValueError(f"Column {DELTA_COL!r} not found in {csv_path}")
    return float(pd.to_numeric(df[DELTA_COL], errors="coerce").mean())


def main() -> None:
    rows = []
    for lang in LANGS:
        rows.append({"lang": lang, "mean_delta": load_mean_delta(lang)})

    plot_df = pd.DataFrame(rows)

    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=plot_df, x="lang", y="mean_delta", color="#4c72b0")
    ymin = plot_df["mean_delta"].min()
    ymax = plot_df["mean_delta"].max()
    pad = (ymax - ymin) * 0.1 if ymax > ymin else 0.01
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("Language")
    ax.set_ylabel("Mean perplexity delta")
    ax.set_title(f"Mean perplexity delta ({YEAR})")

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300)
    print(f"[INFO] Saved figure to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
