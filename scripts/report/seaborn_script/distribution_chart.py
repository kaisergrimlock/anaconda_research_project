#!/usr/bin/env python3
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Config
# =========================
BASE_DIR = Path("outputs") / "baseline" / "2022" / "gpt-oss-20b"  # change as needed
CONFUSION_NAME = "confusion_matrix_llm_vs_nist_pct.csv"
PRED_SCORES: List[str] = ["0", "1", "2", "3"]  # LLM labels (columns)

# figures/nonrel is a sibling of outputs/
PROJECT_ROOT = BASE_DIR.parents[3]           # .../<project_root>/outputs/...
FIG_DIR = PROJECT_ROOT / "figures" / "nonrel"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = FIG_DIR / "nonrel_basic_gpt-oss-20b.png"


# =========================
# Data loading
# =========================
def load_nonrel_distributions(base_dir: Path) -> pd.DataFrame:
    """
    For each language subfolder under base_dir, read confusion_matrix_llm_vs_nist_pct.csv
    and take the NIST=0 row (non-relevant). Returns a long-form dataframe:

        variant  : label for x-axis (NonRelP, NonRelP+eng, ...)
        lang     : raw folder name (raw, eng, fr, ...)
        score    : LLM relevance label (0..3)
        prop     : proportion in [0, 1]
    """
    records = []

    # sort with "raw" first, then alphabetically
    lang_dirs = sorted(
        (p for p in base_dir.iterdir() if p.is_dir()),
        key=lambda p: (p.name != "raw", p.name),
    )

    for lang_dir in lang_dirs:
        lang = lang_dir.name
        csv_path = lang_dir / CONFUSION_NAME
        if not csv_path.exists():
            print(f"[WARN] Missing confusion matrix: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Get the non-relevant row: NIST == 0
        if "NIST" in df.columns:
            nonrel_rows = df[df["NIST"] == 0]
            if nonrel_rows.empty:
                print(f"[WARN] No NIST=0 row in {csv_path}, using first data row.")
                row = df.iloc[0]
            else:
                row = nonrel_rows.iloc[0]
        else:
            # fallback: just take the first row after header
            print(f"[WARN] No 'NIST' column in {csv_path}, using first row.")
            row = df.iloc[0]

        # Pretty label for x-axis
        if lang == "raw":
            variant_name = "NonRelP"
        else:
            variant_name = f"NonRelP+{lang}"

        for score_str in PRED_SCORES:
            pct = float(row[score_str])   # e.g. 27.92
            prop = pct / 100.0            # convert to 0–1

            records.append(
                {
                    "lang": lang,
                    "variant": variant_name,
                    "score": int(score_str),
                    "prop": prop,
                }
            )

    return pd.DataFrame.from_records(records)


# =========================
# Plotting
# =========================
def plot_nonrel_distribution(df: pd.DataFrame, title: str, out_path: Path) -> None:
    sns.set_theme(style="darkgrid")
    plt.style.use("dark_background")

    # Colors for scores 0–3
    palette = {
        0: "#111111",  # dark
        1: "#8f6b32",  # brown
        2: "#5b7f24",  # dark olive green
        3: "#9ad000",  # bright green
    }

    variants = df["variant"].unique()      # already ordered raw → others
    scores = sorted(df["score"].unique())

    fig, ax = plt.subplots(figsize=(5, 6))
    x_positions = range(len(variants))

    for i, variant in enumerate(variants):
        subset = (
            df[df["variant"] == variant]
            .set_index("score")
            .reindex(scores)
            .fillna(0.0)
        )

        bottom = 0.0
        for s in scores:
            height = subset.loc[s, "prop"]
            if height <= 0:
                continue

            ax.bar(
                i,
                height,
                bottom=bottom,
                color=palette[s],
                edgecolor="black",
                linewidth=0.5,
            )

            # label segment if big enough
            if height >= 0.04:
                ax.text(
                    i,
                    bottom + height / 2,
                    f"{height:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

            bottom += height

    # Axes & title
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score Distribution")
    ax.set_ylim(0, 1.01)
    ax.set_title(title, fontsize=12)

    # Legend on the right so it doesn't overlap x-axis
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[s]) for s in scores]
    labels = [str(s) for s in scores]
    ax.legend(
        handles,
        labels,
        title="Relevance Scores",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=1.0,
        edgecolor="white",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
if __name__ == "__main__":
    df_nonrel = load_nonrel_distributions(BASE_DIR)
    if df_nonrel.empty:
        print("[ERROR] No data found; check BASE_DIR and CSV paths.")
    else:
        plot_nonrel_distribution(df_nonrel, title="Basic", out_path=FIG_PATH)
        print(f"[OK] Saved figure to {FIG_PATH}")
