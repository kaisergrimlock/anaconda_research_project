#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]   # adjust if script lives elsewhere

YEAR = "2022"
LANGS = ["eng", "eng_word"]
INPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / f"perplexity_delta_mean_{YEAR}.csv"
OUTPUT_PDF = PROJECT_ROOT / "figures" / f"perplexity_delta_mean_{YEAR}.pdf"
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

    table_df = pd.DataFrame(rows).sort_values("lang").reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved CSV to {OUTPUT_CSV}")

    fig, ax = plt.subplots(figsize=(6, 0.4 + 0.3 * len(table_df)))
    ax.axis("off")
    ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    fig.tight_layout()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    print(f"[INFO] Saved PDF to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
