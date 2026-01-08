#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Config
# =========================
# Take 2 years (or more if you want)
TREC_DL_YEARS: List[str] = ["2021", "2022"]
MODEL = "llama3-8b-instruct"    # e.g. "gpt-oss-20b", "qwen3-32b-v1", ...

# Where the baseline figs live (just to get project root)
BASELINE_DIR = Path("outputs") / "baseline" / TREC_DL_YEARS[0] / MODEL
PROJECT_ROOT = BASELINE_DIR.parents[3]  # .../<project_root>/outputs/...

# Output figure directory
FIG_ROOT = PROJECT_ROOT / "figures" / "2021_2022" / MODEL / "nonrel_nist"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
CHART_TYPE = "word"

YEARS_TAG = "_".join(TREC_DL_YEARS)
GROUPED_FIG_PATH = FIG_ROOT / CHART_TYPE / f"nonreloverall_{MODEL}_{YEARS_TAG}_{CHART_TYPE}_grouped"
STACKED_FIG_PATH = FIG_ROOT / CHART_TYPE / f"nonreloverall_{MODEL}_{YEARS_TAG}_{CHART_TYPE}_stacked"

# Where the LLM label CSVs live (base dir; year/model appended under it)
LABEL_BASE_DIR = Path("outputs") / "llm_label"

# Column order you want (and ONLY these variants will be plotted by default)
#TARGET_LANGS: List[str] = ["eng", "eng_crit", "fr", "fr_crit", "ru", "ru_crit", "vi", "vi_crit", "th", "th_crit", "sw", "sw_crit", "ga", "ga_crit"]
#TARGET_LANGS: List[str] = ["eng", "ar", "ru", "fr", "vi", "th", "sw", "ga",]
TARGET_LANGS: List[str] = ["eng", "eng_word", "ar", "ar_word", "ru", "ru_word", "fr", "fr_word", "vi", "vi_word", "th", "th_word", "sw", "sw_word", "ga", "ga_word",]


# Relevance scores used by the models
SCORES: List[int] = [0, 1, 2, 3]

# This defines the plot order exactly
VARIANT_ORDER: List[str] = ["baseline"] + TARGET_LANGS


label_map = {
    "baseline": "Baseline",
    "eng": "Baseline + EnQP",
    "vi": "Baseline + ViQP",
    "th": "Baseline + ThQP",
    "ru": "Baseline + RuQP",
    "fr": "Baseline + FrQP",
    "sw": "Baseline + SwQP",
    "ga": "Baseline + GaQP",
    "eng_word": "Baseline + EnWP",
    "vi_word": "Baseline + ViWP",
    "th_word": "Baseline + ThWP",
    "ru_word": "Baseline + RuWP",
    "fr_word": "Baseline + FrWP",
    "sw_word": "Baseline + SwWP",
    "ga_word": "Baseline + GaWP",
    "ar_word": "Baseline + ArWP",
    "ar": "Baseline + ArQP",
    "raw_crit": "Baseline Crit",
    "eng_crit": "Baseline Crit + EnQP",
    "vi_crit": "Base Crit + ViQP",
}


# =========================
# Helpers
# =========================
def pick_gold_col(df: pd.DataFrame) -> Optional[str]:
    """Pick the NIST / gold relevance column, if present."""
    if "relevance" in df.columns:
        return "relevance"
    return None


def pick_id_col(df: pd.DataFrame) -> Optional[str]:
    """
    Pick a document id column (used together with qid to form the key).
    Tries common names in order.
    """
    for col in ["pid_qrels", "pid", "docid", "pid_resolved", "passage_id"]:
        if col in df.columns:
            return col
    return None


def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None


def build_raw_zero_keyset(raw_file: Path) -> tuple[pd.DataFrame, Set[str], str]:
    """
    Load the RAW file, filter to rows where:
      - gold relevance == 0
      - AND llm_relevance == 0
    Return:
      (filtered_df, base_key_set, id_col_name)
    """
    df_raw = pd.read_csv(raw_file)

    gold_col = pick_gold_col(df_raw)
    if gold_col is None:
        print(f"[ERROR] RAW file {raw_file.name}: missing gold relevance column ('relevance').")
        return df_raw.iloc[0:0], set(), ""

    if "llm_relevance" not in df_raw.columns:
        print(f"[ERROR] RAW file {raw_file.name}: missing 'llm_relevance' column.")
        return df_raw.iloc[0:0], set(), ""

    if "qid" not in df_raw.columns:
        print(f"[ERROR] RAW file {raw_file.name}: missing 'qid' column.")
        return df_raw.iloc[0:0], set(), ""

    id_col = pick_id_col(df_raw)
    if id_col is None:
        print(f"[ERROR] RAW file {raw_file.name}: could not find any pid/docid column.")
        return df_raw.iloc[0:0], set(), ""

    df_raw["llm_relevance"] = pd.to_numeric(df_raw["llm_relevance"], errors="coerce")

    # RAW cohort: NIST==0 AND RAW_LLM==0
    df_zero = df_raw[(df_raw[gold_col] == 0)]

    if df_zero.empty:
        print(f"[WARN] RAW file {raw_file.name}: no rows with {gold_col}==0 and llm_relevance==0.")
        return df_zero, set(), id_col

    base_keys: Set[str] = set(df_zero["qid"].astype(str) + "|" + df_zero[id_col].astype(str))
    print(f"[INFO] RAW base cohort size in {raw_file.name}: {len(base_keys)}")

    return df_zero, base_keys, id_col


def compute_distribution_for_variant(
    df_variant: pd.DataFrame,
    base_keys: Set[str],
    *,
    year: str,
    variant_name: str,
) -> List[Dict]:
    """
    Restrict df_variant to base_keys cohort, then compute llm_relevance distribution.
    Returns long-form records: year, variant, score, prop
    """
    if "qid" not in df_variant.columns:
        print(f"[WARN] {variant_name} ({year}): missing 'qid'; skipping.")
        return []

    id_col = pick_id_col(df_variant)
    if id_col is None:
        print(f"[WARN] {variant_name} ({year}): missing pid/docid column; skipping.")
        return []

    if "llm_relevance" not in df_variant.columns:
        print(f"[WARN] {variant_name} ({year}): missing 'llm_relevance'; skipping.")
        return []

    df_variant["llm_relevance"] = pd.to_numeric(df_variant["llm_relevance"], errors="coerce")

    df_variant["__pair_key"] = df_variant["qid"].astype(str) + "|" + df_variant[id_col].astype(str)
    df_cohort = df_variant[df_variant["__pair_key"].isin(base_keys)].drop(columns=["__pair_key"])

    df_valid = df_cohort.dropna(subset=["llm_relevance"])
    if df_valid.empty:
        print(f"[INFO] {variant_name} ({year}): no valid rows in RAW cohort; skipping.")
        return []

    counts = df_valid["llm_relevance"].value_counts().to_dict()
    total = len(df_valid)

    out: List[Dict] = []
    for s in SCORES:
        out.append(
            {
                "year": year,
                "variant": variant_name,
                "score": s,
                "prop": (counts.get(s, 0) / total) if total > 0 else 0.0,
            }
        )
    return out


# =========================
# Data loading (multi-year; ordered variants)
# =========================
def load_label_distributions_multi_year(
    label_base_dir: Path,
    years: Sequence[str],
) -> pd.DataFrame:
    """
    For each year:
      - Build RAW cohort keys: (NIST==0 AND RAW llm==0)
      - Compute distributions for:
          baseline (RAW itself) + each lang in TARGET_LANGS, in list order.

    Then:
      - concatenate per-year distributions
      - average props across years per (variant, score)
    """
    per_year_records: List[Dict] = []

    for year in years:
        label_dir = label_base_dir / f"trec_dl_{year}" / MODEL
        if not label_dir.exists():
            print(f"[ERROR] Missing label directory for year {year}: {label_dir}")
            continue

        raw_file = label_dir / f"{MODEL}_trecdl_{year}_raw_labels.csv"
        if not raw_file.exists():
            print(f"[ERROR] Missing raw labels file for year {year}: {raw_file}")
            continue

        df_raw_zero, base_keys, _raw_id_col = build_raw_zero_keyset(raw_file)
        if not base_keys:
            print(f"[ERROR] Year {year}: empty RAW cohort; skipping year.")
            continue

        # Baseline distribution = RAW distribution over the cohort
        # (df_raw_zero is already the cohort subset for RAW)
        counts_raw = df_raw_zero["llm_relevance"].value_counts().to_dict()
        total_raw = len(df_raw_zero)
        for s in SCORES:
            per_year_records.append(
                {
                    "year": year,
                    "variant": "baseline",
                    "score": s,
                    "prop": (counts_raw.get(s, 0) / total_raw) if total_raw > 0 else 0.0,
                }
            )

        # Now loop langs one by one in TARGET_LANGS order (this guarantees order later)
        for lang in TARGET_LANGS:
            file_path = label_dir / f"{MODEL}_trecdl_{year}_{lang}_labels.csv"
            if not file_path.exists():
                print(f"[WARN] Year {year}: missing {lang} file: {file_path.name} (skipping)")
                continue

            df_lang = read_csv_safe(file_path)
            if df_lang is None:
                continue

            per_year_records.extend(
                compute_distribution_for_variant(
                    df_lang,
                    base_keys,
                    year=year,
                    variant_name=lang,
                )
            )

    df_years = pd.DataFrame.from_records(per_year_records)
    if df_years.empty:
        return df_years

    # Debug: per-year sums should be ~1 for each (year, variant)
    print("\n[DEBUG] Per-year sum of props per (year, variant) (should be ~1.0):")
    print(df_years.groupby(["year", "variant"])["prop"].sum())

    # Combine years: mean prop per (variant, score)
    df_combined = df_years.groupby(["variant", "score"], as_index=False)["prop"].mean()

    print("\n[DEBUG] Combined sum of props per variant (should be ~1.0):")
    print(df_combined.groupby("variant")["prop"].sum())

    return df_combined


# =========================
# Plotting – grouped bars (ordered variants)
# =========================
def plot_grouped_distribution(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    sns.set_theme(style="whitegrid")
    plt.style.use("default")

    df = df.copy()

    # Ensure we only plot variants in our desired order (but keep any extras at the end if they exist)
    variants_present = df["variant"].unique().tolist()
    variants: List[str] = [v for v in VARIANT_ORDER if v in variants_present]
    variants += [v for v in variants_present if v not in set(variants)]

    scores = sorted(df["score"].unique())

    grouped = df.groupby(["score", "variant"], as_index=False)["prop"].sum()

    # Fast lookup: (variant, score) -> prop
    prop_map: Dict[Tuple[str, int], float] = {}
    for _, row in grouped.iterrows():
        prop_map[(str(row["variant"]), int(row["score"]))] = float(row["prop"])

    n_scores = len(scores)
    n_variants = len(variants)
    x_base = list(range(n_scores))

    group_width = 0.8
    bar_width = group_width / max(1, n_variants)

    fig, ax = plt.subplots(figsize=(6, 6))

    colors = sns.color_palette("muted", n_colors=n_variants)
    variant_colors: Dict[str, tuple] = {v: colors[i] for i, v in enumerate(variants)}

    for j, variant in enumerate(variants):
        heights = [prop_map.get((variant, s), 0.0) for s in scores]
        offsets = [x + (j - (n_variants - 0.5) / 2) * bar_width for x in x_base]

        ax.bar(
            offsets,
            heights,
            width=bar_width,
            label=variant,
            color=variant_colors[variant],
            edgecolor="black",
            linewidth=0.5,
        )

        ax.grid(False)

        for x, h in zip(offsets, heights):
            if h >= 0.04:
                ax.text(
                    x,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    color="white",
                )

    ax.set_xticks(x_base)
    ax.set_ylabel("Score Distribution")
    ax.set_ylim(0, 1.05)
    #ax.set_title(title, fontsize=12)

    ax.legend(
        title="Variant",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=1.0,
        edgecolor="white",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# Plotting – stacked bars (ordered variants)
# =========================
def plot_stacked_distribution(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    sns.set_theme(style="whitegrid")
    plt.style.use("default")

    df = df.copy()

    grouped = df.groupby(["variant", "score"], as_index=False)["prop"].sum()
    pivot = grouped.pivot(index="variant", columns="score", values="prop").fillna(0.0)

    # Order variants exactly as requested
    ordered: List[str] = [v for v in VARIANT_ORDER if v in pivot.index]
    ordered += [v for v in pivot.index if v not in set(ordered)]  # append any extras
    pivot = pivot.loc[ordered]
    variants = ordered

    scores = sorted(pivot.columns)
    x = list(range(len(variants)))

    fig, ax = plt.subplots(figsize=(6, 6))

    default_palette = {
        0: "#000000",  # black
        1: "#a6761d",  # muted brown
        2: "#6b8e23",  # olive/green
        3: "#b8de6f",  # light yellow-green
    }
    fallback_colors = sns.color_palette("muted", n_colors=len(scores))
    score_colors: Dict[int, tuple] = {s: default_palette.get(s, fallback_colors[i]) for i, s in enumerate(scores)}

    bottom = [0.0] * len(variants)
    for s in scores:
        heights = pivot[s].values.tolist()

        ax.bar(
            x,
            heights,
            bottom=bottom,
            label=str(s),
            color=score_colors[s],
            edgecolor="black",
            linewidth=0.5,
        )

        for xi, h, b in zip(x, heights, bottom):
            if h >= 0.04:
                ax.text(
                    xi,
                    b + h / 2.0,
                    f"{h:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                )

        bottom = [b + h for b, h in zip(bottom, heights)]

    ax.grid(False)
    pretty_labels = [label_map.get(v, v) for v in variants]

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Relevance Label Score Distribution")
    ax.set_ylim(0, 1.05)
    #ax.set_title(title, fontsize=12)

    ax.legend(
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
    df_labels = load_label_distributions_multi_year(LABEL_BASE_DIR, TREC_DL_YEARS)

    if df_labels.empty:
        print("[ERROR] No data found across years; check LABEL_BASE_DIR, years, and columns.")
    else:
        years_txt = ", ".join(TREC_DL_YEARS)
        title = (
            "Relevance Label Distribution for Non-Relevant Passages "
            "(NIST = 0 & Baseline LLM = 0) "
            f"— Years: {years_txt}"
        )

        # Grouped bar chart
        plot_grouped_distribution(
            df_labels,
            title=title,
            out_path=GROUPED_FIG_PATH.with_suffix(".png"),
        )
        print(f"[OK] Saved grouped figure to {GROUPED_FIG_PATH.with_suffix('.png')}")

        plot_grouped_distribution(
            df_labels,
            title=title,
            out_path=GROUPED_FIG_PATH.with_suffix(".svg"),
        )
        print(f"[OK] Saved grouped figure to {GROUPED_FIG_PATH.with_suffix('.svg')}")

        # Stacked bar chart
        plot_stacked_distribution(
            df_labels,
            title=title,
            out_path=STACKED_FIG_PATH.with_suffix(".png"),
        )
        print(f"[OK] Saved stacked figure to {STACKED_FIG_PATH.with_suffix('.png')}")

        plot_stacked_distribution(df_labels, title=title, out_path=STACKED_FIG_PATH.with_suffix(".pdf"))
        print(f"[OK] Saved stacked figure to {STACKED_FIG_PATH.with_suffix('.pdf')}")

        plot_stacked_distribution(
            df_labels,
            title="",
            out_path=STACKED_FIG_PATH.with_suffix(".svg"),
        )
        print(f"[OK] Saved stacked figure to {STACKED_FIG_PATH.with_suffix('.svg')}")
