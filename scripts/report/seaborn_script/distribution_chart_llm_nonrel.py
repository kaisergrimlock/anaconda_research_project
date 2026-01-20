#!/usr/bin/env python3
from pathlib import Path
from typing import List, Dict, Set, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Config
# =========================
TREC_DL_YEAR = "2021"
MODEL = "gpt-oss-20b"    # e.g. "gpt-oss-20b", "qwen3-32b-v1", ...

# Where the baseline figs live (just to get project root)
BASELINE_DIR = Path("outputs") / "baseline" / TREC_DL_YEAR / MODEL
PROJECT_ROOT = BASELINE_DIR.parents[3]           # .../<project_root>/outputs/...

# Output figure directory
FIG_ROOT = PROJECT_ROOT / "figures" / TREC_DL_YEAR / MODEL / "nonrel_llm"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
CHART_TYPE = "corrected"
GROUPED_FIG_PATH = FIG_ROOT / CHART_TYPE / f"nonreloverall_{MODEL}_{TREC_DL_YEAR}_{CHART_TYPE}_grouped"
STACKED_FIG_PATH = FIG_ROOT / CHART_TYPE / f"nonreloverall_{MODEL}_{TREC_DL_YEAR}_{CHART_TYPE}_stacked"

# Where the LLM label CSVs live
LABEL_DIR = Path("outputs") / "llm_label" / f"trec_dl_{TREC_DL_YEAR}" / MODEL

# Only keep these “language” variants.-
# Set to [] or None to include all non-raw files.
TARGET_LANGS: Optional[List[str]] = ["eng", "ar", "ru", "fr", "zh", "vi", "he", "hi", "th", "sw", "ga"]
#TARGET_LANGS: List[str] = ["eng", "eng_word", "fr", "fr_word", "ru", "ru_word", "vi", "vi_word", "th", "th_word", "sw", "sw_word", "ga", "ga_word"]

# Relevance scores used by the models
SCORES: List[int] = [0, 1, 2, 3]


# =========================
# Helpers
# =========================
def parse_lang_from_filename(path: Path) -> str:
    """
    Given a filename like:
        gpt-oss-20b_trecdl_2022_eng_mult_labels.csv
        gpt-oss-20b_trecdl_2022_eng_vi_between_labels.csv
    return:
        "eng_mult"
        "eng_vi_between"

    Pattern is assumed:
        {MODEL}_trecdl_{YEAR}_{lang}_labels.csv
    where {lang} may itself contain underscores.
    """
    parts = path.stem.split("_")
    # Expect at least: [MODEL, 'trecdl', YEAR, <lang...>, 'labels']
    if len(parts) >= 5 and parts[1] == "trecdl":
        # join everything between YEAR and 'labels'
        lang = "_".join(parts[3:-1])
    else:
        # fallback: second last token
        lang = parts[-2] if len(parts) >= 2 else "unknown"
    return lang


def pick_gold_col(df: pd.DataFrame) -> Optional[str]:
    """Pick the NIST / gold relevance column, if present."""
    if "relevance" in df.columns:
        return "relevance"
    else:
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


def build_raw_zero_keyset(raw_file: Path) -> tuple[pd.DataFrame, Set[str], str]:
    """
    Load the RAW file, filter to rows where:
      - gold relevance == 0
      - AND llm_relevance == 0
    Return:
      (filtered_df, base_key_set, id_col_name)

    base_key_set contains keys of the form "qid|doc_id".
    """
    df_raw = pd.read_csv(raw_file)

    gold_col = pick_gold_col(df_raw)
    if gold_col is None:
        print(f"[ERROR] RAW file {raw_file.name}: no gold relevance column ('relevance' or 'NIST_relevance').")
        return df_raw.iloc[0:0], set(), ""

    if "llm_relevance" not in df_raw.columns:
        print(f"[ERROR] RAW file {raw_file.name}: no 'llm_relevance' column.")
        return df_raw.iloc[0:0], set(), ""

    if "qid" not in df_raw.columns:
        print(f"[ERROR] RAW file {raw_file.name}: no 'qid' column.")
        return df_raw.iloc[0:0], set(), ""

    id_col = pick_id_col(df_raw)
    if id_col is None:
        print(f"[ERROR] RAW file {raw_file.name}: could not find any pid/docid column.")
        return df_raw.iloc[0:0], set(), ""

    # Ensure numeric llm_relevance
    df_raw["llm_relevance"] = pd.to_numeric(df_raw["llm_relevance"], errors="coerce")

    # Filter: NIST==0 AND LLM==0
    df_zero = df_raw[(df_raw[gold_col] == 0) & (df_raw["llm_relevance"] == 0)]
    df_zero = df_zero.dropna(subset=["llm_relevance"])

    if df_zero.empty:
        print(f"[WARN] RAW file {raw_file.name}: no rows with {gold_col}==0 and llm_relevance==0.")
        return df_zero, set(), id_col

    # Build key set
    base_keys: Set[str] = set(
        df_zero["qid"].astype(str) + "|" + df_zero[id_col].astype(str)
    )

    print(f"[INFO] RAW base cohort size (NIST==0 & LLM==0): {len(base_keys)}")

    return df_zero, base_keys, id_col


label_map = {
    "baseline": "Baseline",
    "eng": "Baseline + EnQP",
    "vi": "Baseline + ViQP",
    "th": "Baseline + ThQP",
    "ru": "Baseline + RuQP",
    "fr": "Baseline + FrQP",
    "er": "Baseline + ErQP",
    "eng_word": "Baseline + EnWP",
    "vi_word": "Baseline + ViWP",
    "raw_crit": "Baseline Crit",
    "eng_crit": "Baseline Crit + EnQP",
    "vi_crit": "Base Crit + ViQP",
    "eng_last": "Baseline + EnQP (Last)",
    "eng_first": "Baseline + EnQP (First)",
}

# =========================
# Data loading
# =========================
def load_label_distributions(label_dir: Path) -> pd.DataFrame:
    """
    Build a long-form DataFrame using a *fixed cohort*:

      Cohort = all (qid, doc_id) pairs in the RAW file such that
               gold relevance == 0 AND llm_relevance == 0.

    For that cohort, we compute, for each variant (raw, eng, vi, th, ...):

        variant : 'raw' for raw file, otherwise the parsed lang (e.g. 'eng', 'vi_trans_q')
        lang    : same as variant
        score   : llm_relevance (0..3)
        prop    : proportion in [0, 1] of each score within the cohort
    """
    records: List[Dict] = []

    # ---------- RAW: define the cohort ----------
    raw_file = label_dir / f"{MODEL}_trecdl_{TREC_DL_YEAR}_raw_labels.csv"
    if not raw_file.exists():
        print(f"[ERROR] Missing raw labels file: {raw_file}")
        return pd.DataFrame()

    df_raw_zero, base_keys, raw_id_col = build_raw_zero_keyset(raw_file)
    if not base_keys:
        print("[ERROR] Empty cohort after NIST==0 & LLM==0 filtering on RAW; nothing to plot.")
        return pd.DataFrame()

    # Distribution for RAW itself (over the same cohort)
    counts_raw = df_raw_zero["llm_relevance"].value_counts().to_dict()
    total_raw = len(df_raw_zero)

    for s in SCORES:
        prop = counts_raw.get(s, 0) / total_raw if total_raw > 0 else 0.0
        records.append(
            {
                "lang": "raw",
                "variant": "raw",
                "score": s,
                "prop": prop,
            }
        )

    # ---------- OTHER VARIANTS ----------
    seen_langs: set[str] = set()
    pattern = f"{MODEL}_trecdl_{TREC_DL_YEAR}_*_labels.csv"

    for file_path in label_dir.glob(pattern):
        # skip raw (already handled)
        if file_path.name.endswith("_raw_labels.csv"):
            continue

        lang = parse_lang_from_filename(file_path)

        # optionally restrict to selected variants
        if TARGET_LANGS and (lang not in TARGET_LANGS):
            continue

        # avoid accidental duplicates for the same lang
        if lang in seen_langs:
            print(f"[SKIP] Duplicate file for lang '{lang}': {file_path.name}")
            continue
        seen_langs.add(lang)

        df_other = pd.read_csv(file_path)

        if "qid" not in df_other.columns:
            print(f"[WARN] {file_path.name}: missing 'qid' column; skipping.")
            continue

        id_col_other = pick_id_col(df_other)
        if id_col_other is None:
            print(f"[WARN] {file_path.name}: could not find any pid/docid column; skipping.")
            continue

        if "llm_relevance" not in df_other.columns:
            print(f"[WARN] {file_path.name}: no 'llm_relevance' column; skipping.")
            continue

        df_other["llm_relevance"] = pd.to_numeric(df_other["llm_relevance"], errors="coerce")

        # Build keys and restrict to the base cohort
        df_other["__pair_key"] = df_other["qid"].astype(str) + "|" + df_other[id_col_other].astype(str)
        df_cohort = df_other[df_other["__pair_key"].isin(base_keys)].drop(columns=["__pair_key"])

        if df_cohort.empty:
            print(f"[INFO] {file_path.name}: no rows matching RAW cohort keys; skipping.")
            continue

        df_valid = df_cohort.dropna(subset=["llm_relevance"])
        if df_valid.empty:
            print(f"[INFO] {file_path.name}: no valid llm_relevance in cohort subset; skipping.")
            continue

        counts = df_valid["llm_relevance"].value_counts().to_dict()
        total = len(df_valid)
        variant_name = lang

        for s in SCORES:
            prop = counts.get(s, 0) / total if total > 0 else 0.0
            records.append(
                {
                    "lang": lang,
                    "variant": variant_name,
                    "score": s,
                    "prop": prop,
                }
            )

    df = pd.DataFrame.from_records(records)

    # Debug: ensure each variant's props sum to ~1
    if not df.empty:
        print("\n[DEBUG] Sum of props per variant (should be ~1.0 each, over RAW zero cohort):")
        print(df.groupby("variant")["prop"].sum())

    return df

def display_variant_name(v: str) -> str:
    return "baseline" if v == "raw" else v

# =========================
# Plotting – grouped bars (muted palette)
# =========================
def plot_grouped_distribution(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    sns.set_theme(style="whitegrid")
    plt.style.use("default")
    df = df.copy()
    df["variant"] = df["variant"].replace({"raw": "baseline"})
    # Order: raw first, then others alphabetically
    variants = sorted(
        df["variant"].unique(),
        key=lambda v: (v != "raw", v),
    )
    scores = sorted(df["score"].unique())

    # Aggregate just in case there are duplicates
    grouped = df.groupby(["score", "variant"], as_index=False)["prop"].sum()

    # X-axis = scores; for each score, we show one bar per variant
    n_scores = len(scores)
    n_variants = len(variants)
    x_base = list(range(n_scores))

    # Total width of each group (per score)
    group_width = 0.8
    bar_width = group_width / max(1, n_variants)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Muted palette for variants
    colors = sns.color_palette("muted", n_colors=n_variants)
    variant_colors: Dict[str, tuple] = {
        v: colors[i] for i, v in enumerate(variants)
    }

    for j, variant in enumerate(variants):
        var_data = grouped[grouped["variant"] == variant].set_index("score")["prop"]
        heights = [float(var_data.get(s, 0.0)) for s in scores]

        # Shift bars within each group
        offsets = [
            x + (j - (n_variants - 0.5) / 2) * bar_width for x in x_base
        ]

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

        # Optional: label bars if tall enough
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
# Plotting – stacked bars (muted palette matching screenshot)
# =========================

def plot_stacked_distribution(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    """
    Stacked bar chart:
      x-axis  : variants (raw, eng, vi, th, ...)
      stacks  : scores (0,1,2,3)
      height  : proportion per score, so each bar sums ~1.0
      labels  : numeric proportions on each segment (0–1, formatted to 2 d.p.)
    """
    sns.set_theme(style="whitegrid")
    plt.style.use("default")

    df = df.copy()
    df["variant"] = df["variant"].replace({"raw": "baseline"})

    # Aggregate and pivot to variant x score
    grouped = df.groupby(["variant", "score"], as_index=False)["prop"].sum()
    pivot = grouped.pivot(index="variant", columns="score", values="prop").fillna(0.0)

    # Order variants: raw first, then alphabetically
    variants = sorted(pivot.index, key=lambda v: (v != "raw", v))
    pivot = pivot.loc[variants]

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
    score_colors: Dict[int, tuple] = {}
    for i, s in enumerate(scores):
        score_colors[s] = default_palette.get(s, fallback_colors[i])

    # Stack bars
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

        # Add labels in the middle of each segment (if big enough)
        for xi, h, b in zip(x, heights, bottom):
            if h >= 0.04:   # skip tiny slivers
                y = b + h / 2.0
                ax.text(
                    xi,
                    y,
                    f"{h:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                )

        # Update bottom for next stack
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
    df_labels = load_label_distributions(LABEL_DIR)

    if df_labels.empty:
        print("[ERROR] No data found for RAW NIST=0 & LLM=0 cohort; check LABEL_DIR and columns.")
    else:
        title = "Relevance Label Distribution for Non-Relevant Passages (NIST = 0 & Baseline LLM = 0)"

        # Grouped bar chart
        plot_grouped_distribution(
            df_labels,
            title=title,
            out_path=GROUPED_FIG_PATH.with_suffix(".png"),
        )
        print(f"[OK] Saved grouped figure to {GROUPED_FIG_PATH}")

        plot_grouped_distribution(
            df_labels,
            out_path=GROUPED_FIG_PATH.with_suffix(".svg"),
        )
        print(f"[OK] Saved grouped figure to {GROUPED_FIG_PATH}")

        # Stacked bar chart
        plot_stacked_distribution(
            df_labels,
            title=title,
            out_path=STACKED_FIG_PATH.with_suffix(".png"),
        )
        print(f"[OK] Saved stacked figure to {STACKED_FIG_PATH}")

        plot_stacked_distribution(
            df_labels,
            out_path=STACKED_FIG_PATH.with_suffix(".svg"),
        )
        print(f"[OK] Saved stacked figure to {STACKED_FIG_PATH}")
