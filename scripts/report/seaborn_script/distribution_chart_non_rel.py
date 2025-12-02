#!/usr/bin/env python3
from pathlib import Path
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Config
# =========================
TREC_DL_YEAR = "2022"
MODEL = "gpt-oss-20b"   # e.g. "gpt-oss-20b", "qwen3-32b-v1", ...

# Where the baseline figs live (just to get project root)
BASELINE_DIR = Path("outputs") / "baseline" / TREC_DL_YEAR / MODEL
PROJECT_ROOT = BASELINE_DIR.parents[3]           # .../<project_root>/outputs/...

# Output figure
FIG_DIR = PROJECT_ROOT / "figures" / "nonrel"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = FIG_DIR / f"nonrelpairs_{MODEL}_{TREC_DL_YEAR}_trans_q.png"

# Where the LLM label CSVs live
LABEL_DIR = Path("outputs") / "llm_label" / f"trec_dl_{TREC_DL_YEAR}" / MODEL

# Only keep these “language” variants.
# Set to [] or None to include all non-raw files.
TARGET_LANGS: List[str] = ["eng", "vi_trans_q", "vi"]

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


# =========================
# Data loading
# =========================
def load_nonrel_from_llm_labels(label_dir: Path) -> pd.DataFrame:
    """
    Build a long-form DataFrame:

        variant : "NonRelP" for raw, "NonRelP+<lang>" otherwise
        lang    : raw, eng, eng_mult, eng_vi_between, ...
        score   : llm_relevance (0..3)
        prop    : proportion in [0, 1] among *raw non-rel pairs* for that variant
    """
    records: List[Dict] = []

    # ---------- RAW ----------
    raw_file = label_dir / f"{MODEL}_trecdl_{TREC_DL_YEAR}_raw_labels.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Missing raw labels file: {raw_file}")

    df_raw = pd.read_csv(raw_file)
    df_raw["llm_relevance"] = pd.to_numeric(df_raw["llm_relevance"], errors="coerce")

    # rows where raw judged 0
    non_rel_raw = df_raw[df_raw["llm_relevance"] == 0].copy()
    if non_rel_raw.empty:
        print("[WARN] No llm_relevance == 0 rows in RAW.")
        return pd.DataFrame()

    key_cols = ["qid", "pid"]
    nonrel_keys = non_rel_raw[key_cols].drop_duplicates()

    # distribution in RAW (will be all zeros except score 0)
    counts_raw = non_rel_raw["llm_relevance"].value_counts().to_dict()
    total_raw = len(non_rel_raw)

    for s in SCORES:
        prop = counts_raw.get(s, 0) / total_raw
        records.append(
            {
                "lang": "raw",
                "variant": "NonRelP",
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
        df_other["llm_relevance"] = pd.to_numeric(
            df_other["llm_relevance"], errors="coerce"
        )

        # keep only the (qid, pid) pairs that were 0 in raw
        df_match = df_other.merge(nonrel_keys, on=key_cols, how="inner")
        if df_match.empty:
            print(f"[INFO] No matching non-rel pairs in {file_path.name}")
            continue

        counts = df_match["llm_relevance"].value_counts().to_dict()
        total = len(df_match)
        variant_name = f"NonRelP+{lang}"

        for s in SCORES:
            prop = counts.get(s, 0) / total
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
        print("\n[DEBUG] Sum of props per variant (should be 1.0 each):")
        print(df.groupby("variant")["prop"].sum())

    return df


# =========================
# Plotting
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

    # order: NonRelP first, then others alphabetically
    variants = sorted(
        df["variant"].unique(),
        key=lambda v: (v != "NonRelP", v),
    )
    scores = sorted(df["score"].unique())

    fig, ax = plt.subplots(figsize=(6, 6))
    x_positions = range(len(variants))

    for i, variant in enumerate(variants):
        subset = df[df["variant"] == variant]

        # aggregate just in case there are duplicates
        series = subset.groupby("score", dropna=False)["prop"].sum()
        series = series.reindex(scores).fillna(0.0)

        # normalise defensively so each stack sums to 1.0
        total = series.sum()
        if total > 0:
            series = series / total

        bottom = 0.0
        for s in scores:
            height = series.loc[s]
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


# =========================
# Main
# =========================
if __name__ == "__main__":
    df_nonrel = load_nonrel_from_llm_labels(LABEL_DIR)

    if df_nonrel.empty:
        print("[ERROR] No data found; check LABEL_DIR and raw llm_relevance == 0.")
    else:
        plot_nonrel_distribution(
            df_nonrel,
            title="NonRelP pairs – LLM labels across variants",
            out_path=FIG_PATH,
        )
        print(f"[OK] Saved figure to {FIG_PATH}")