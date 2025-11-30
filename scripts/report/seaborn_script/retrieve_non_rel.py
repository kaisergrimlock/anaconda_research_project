#!/usr/bin/env python3
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Config
# =========================
TREC_DL_YEAR = "2022"
MODEL = "qwen3-32b-v1"  # e.g., "qwen3-32b-v1", "gpt-oss-20b", etc.

# Baseline dir is only used to find PROJECT_ROOT / figs
BASELINE_DIR = Path("outputs") / "baseline" / TREC_DL_YEAR / MODEL
PROJECT_ROOT = BASELINE_DIR.parents[3]           # .../<project_root>/outputs/...
FIG_DIR = PROJECT_ROOT / "figures" / "nonrel"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = FIG_DIR / f"nonrelpairs_{MODEL}_{TREC_DL_YEAR}.png"

# Where the LLM label CSVs live
LABEL_DIR = Path("outputs") / "llm_label" / f"trec_dl_{TREC_DL_YEAR}" / MODEL

# Optional: limit to some langs
TARGET_LANGS: List[str] = ["eng", "vi", "fr", "th", "ru"]  # edit or remove filter

# =========================

def load_nonrel_from_llm_labels(label_dir: Path) -> pd.DataFrame:
    """
    Build a long-form DataFrame with the same schema as the confusion-matrix
    version, but computed directly from LLM label rows:

        variant : "NonRelP" for raw, "NonRelP+<lang>" otherwise
        lang    : raw, eng, vi, ...
        score   : llm_relevance (0..3)
        prop    : proportion in [0, 1] among non-rel pairs for that variant
    """
    records = []

    raw_file = label_dir / f"{MODEL}_trecdl_{TREC_DL_YEAR}_raw_labels.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing raw labels file: {raw_file}")

    df_raw = pd.read_csv(raw_file)

    # Make sure llm_relevance is numeric
    df_raw["llm_relevance"] = pd.to_numeric(df_raw["llm_relevance"], errors="coerce")

    # Non-rel pairs in raw
    non_rel_raw = df_raw[df_raw["llm_relevance"] == 0].copy()
    if non_rel_raw.empty:
        print("[WARN] No llm_relevance == 0 rows in RAW.")
        return pd.DataFrame()

    key_cols = ["qid", "pid"]
    nonrel_keys = non_rel_raw[key_cols].drop_duplicates()

    scores = [0, 1, 2, 3]

    # ---- RAW variant distribution (will be all zeros, but keep for completeness) ----
    counts_raw = non_rel_raw["llm_relevance"].value_counts().to_dict()
    total_raw = len(non_rel_raw)

    for s in scores:
        prop = counts_raw.get(s, 0) / total_raw
        records.append(
            {
                "lang": "raw",
                "variant": "NonRelP",
                "score": s,
                "prop": prop,
            }
        )

    # ---- Other language files ----
    for file_path in label_dir.glob(f"{MODEL}_trecdl_{TREC_DL_YEAR}_*_labels.csv"):
        if "raw" in file_path.name:
            continue  # already handled

        # Extract lang from filename: {MODEL}_trecdl_{YEAR}_{lang}_labels.csv
        stem_parts = file_path.stem.split("_")
        lang = stem_parts[-2] if len(stem_parts) >= 2 else "unknown"

        # If you only want some langs
        if TARGET_LANGS and (lang not in TARGET_LANGS):
            continue

        df_other = pd.read_csv(file_path)
        df_other["llm_relevance"] = pd.to_numeric(
            df_other["llm_relevance"], errors="coerce"
        )

        # keep only rows whose (qid, pid) are in the non-rel raw set
        df_match = df_other.merge(nonrel_keys, on=key_cols, how="inner")
        if df_match.empty:
            print(f"[INFO] No matching non-rel pairs in {file_path.name}")
            continue

        counts = df_match["llm_relevance"].value_counts().to_dict()
        total = len(df_match)

        variant_name = f"NonRelP+{lang}"

        for s in scores:
            prop = counts.get(s, 0) / total
            records.append(
                {
                    "lang": lang,
                    "variant": variant_name,
                    "score": s,
                    "prop": prop,
                }
            )

    return pd.DataFrame.from_records(records)



# =========================
# Plotting (unchanged)
# =========================
def plot_nonrel_distribution(df: pd.DataFrame, title: str, out_path: Path) -> None:
    sns.set_theme(style="darkgrid")
    plt.style.use("dark_background")

    palette = {
        0: "#111111",  # dark
        1: "#8f6b32",  # brown
        2: "#5b7f24",  # dark olive green
        3: "#9ad000",  # bright green
    }

    variants = df["variant"].unique()
    scores = sorted(df["score"].unique())

    fig, ax = plt.subplots(figsize=(5, 6))
    x_positions = range(len(variants))

    for i, variant in enumerate(variants):
        # Take only this variant and aggregate in case there are duplicates per score
        subset = (
            df[df["variant"] == variant]
            .groupby("score", dropna=False)["prop"]
            .sum()                # Series: index = score, value = summed prop
        )

        # Ensure all scores exist; fill missing with 0.0
        subset = subset.reindex(scores).fillna(0.0)

        bottom = 0.0
        for s in scores:
            height = subset.loc[s]
            if height <= 0:
                continue

            ax.bar(
                i,
                height,
                bottom=bottom,
                color=palette.get(s, "#444444"),
                edgecolor="black",
                linewidth=0.5,
            )

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

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score Distribution")
    ax.set_ylim(0, 1.01)
    ax.set_title(title, fontsize=12)

    handles = [plt.Rectangle((0, 0), 1, 1, color=palette.get(s, "#444444")) for s in scores]
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



# ========================= # Main # =========================
if __name__ == "__main__":
    df_nonrel = load_nonrel_from_llm_labels(LABEL_DIR)
    print(df_nonrel.groupby("variant")["prop"].sum())
    if df_nonrel.empty:
        print("[ERROR] No data found; check LABEL_DIR and raw llm_relevance == 0.")
    else:
        plot_nonrel_distribution(
            df_nonrel,
            title="NonRelP pairs – LLM labels across variants",
            out_path=FIG_PATH,
        )
        print(f"[OK] Saved figure to {FIG_PATH}")