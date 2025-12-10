#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========= Config you can edit =========
TREC_DL_YEAR = "2022"
MODEL        = "gpt-oss-20b"

# Languages to compare (matching filenames: ..._<lang>_<criterion>_labels.csv)
LANGS        = ["raw", "eng", "eng_word"]

# Which criterion to plot: "contextuality", "coverage", "exactness", or "topicality"
CRITERION    = "exactness"

THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

CRITERION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "llm_label"
    / f"trec_dl_{TREC_DL_YEAR}"
    / MODEL
    / "criterion"
)

# Where to save the figure
FIG_DIR  = PROJECT_ROOT / "figures" / "criterion_compare"
FIG_DIR.mkdir(parents=True, exist_ok=True)
LANG_PART = "_".join(LANGS)
FIG_PATH = FIG_DIR / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{CRITERION}_{LANG_PART}.png"
# ======================================


def load_lang_series(lang: str, criterion: str) -> pd.Series:
    """
    Load one language's criterion file and return a Series of criterion scores.
    """
    fname = f"{MODEL}_trecdl_{TREC_DL_YEAR}_{lang}_{criterion}_labels.csv"
    path = CRITERION_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"File not found for lang={lang}, criterion={criterion}: {path}")

    print(f"[INFO] Loading {path.name}")
    df = pd.read_csv(path)

    if criterion not in df.columns:
        raise KeyError(
            f"Column '{criterion}' not found in {path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    s = df[criterion].astype(str).str.strip()
    s = s[s != ""]
    s = pd.to_numeric(s, errors="coerce").dropna().astype(int)
    return s


def build_label_distribution(langs: List[str], criterion: str) -> pd.DataFrame:
    """
    Build a tidy DataFrame with columns: score, language, count
    from the per-language criterion label files.
    """
    rows: List[Dict[str, object]] = []
    labels = [0, 1, 2, 3]

    for lang in langs:
        series = load_lang_series(lang, criterion)
        value_counts = series.value_counts().to_dict()

        for lab in labels:
            rows.append(
                {
                    "score": lab,
                    "language": lang,
                    "count": int(value_counts.get(lab, 0)),
                }
            )

    return pd.DataFrame(rows)


def plot_grouped_bars(dist_df: pd.DataFrame, out_path: Path, criterion: str) -> None:
    """
    Plot grouped bar chart:
      x = score, y = count, hue = language
    """
    if dist_df.empty:
        print("[WARN] No data to plot.")
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))

    # Ensure score is categorical and ordered
    dist_df["score"] = dist_df["score"].astype(int).astype(str)

    sns.barplot(
        data=dist_df,
        x="score",
        y="count",
        hue="language",
    )

    plt.xlabel(f"{criterion.capitalize()} score")
    plt.ylabel("Count of (qid, pid) pairs")
    plt.title(
        f"{criterion.capitalize()} distribution by language\n"
        f"{MODEL}, trec_dl_{TREC_DL_YEAR}"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[DONE] Saved figure to {out_path}")


def main() -> None:
    if not CRITERION_DIR.exists():
        print(f"[FATAL] Criterion dir not found: {CRITERION_DIR}")
        sys.exit(1)

    dist_df = build_label_distribution(LANGS, CRITERION)
    print(
        "[INFO] Distribution table:\n",
        dist_df.pivot(index="score", columns="language", values="count")
    )

    plot_grouped_bars(dist_df, FIG_PATH, CRITERION)


if __name__ == "__main__":
    main()
