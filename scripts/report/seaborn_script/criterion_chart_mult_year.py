#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========= Config you can edit =========
TREC_DL_YEARS = ["2021", "2022"]   # <-- now supports multiple years
MODEL         = "gpt-oss-20b"

# Languages to compare (matching filenames: ..._<lang>_<criterion>_labels.csv)
LANGS         = ["raw", "eng", "ru", "fr", "vi"]

# Which criterion to plot: "contextuality", "coverage", "exactness", or "topicality"
CRITERION     = "coverage"

THIS_FILE     = Path(__file__).resolve()
PROJECT_ROOT  = THIS_FILE.parents[3]
# ======================================


def criterion_dir_for_year(year: str) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "llm_label"
        / f"trec_dl_{year}"
        / MODEL
        / "criterion"
    )


def load_lang_series(year: str, lang: str, criterion: str) -> pd.Series:
    """
    Load one (year, language)'s criterion file and return a Series of criterion scores.
    """
    crit_dir = criterion_dir_for_year(year)
    fname = f"{MODEL}_trecdl_{year}_{lang}_{criterion}_labels.csv"
    path = crit_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"File not found for year={year}, lang={lang}, criterion={criterion}: {path}"
        )

    print(f"[INFO] Loading {path}")
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


def build_label_distribution(years: List[str], langs: List[str], criterion: str) -> pd.DataFrame:
    """
    Build a tidy DataFrame with columns: score, year, language, year_lang, count
    """
    rows: List[Dict[str, object]] = []
    labels = [0, 1, 2, 3]

    for year in years:
        for lang in langs:
            series = load_lang_series(year, lang, criterion)
            value_counts = series.value_counts().to_dict()

            for lab in labels:
                rows.append(
                    {
                        "score": lab,
                        "year": year,
                        "language": lang,
                        "year_lang": f"{year}-{lang}",
                        "count": int(value_counts.get(lab, 0)),
                    }
                )

    return pd.DataFrame(rows)


def plot_grouped_bars(dist_df: pd.DataFrame, out_png: Path, out_svg: Path, criterion: str) -> None:
    """
    Plot grouped bar chart:
      x = score, y = count, hue = year_lang
    Save both PNG and SVG.
    """
    if dist_df.empty:
        print("[WARN] No data to plot.")
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))

    # Ensure score is categorical and ordered
    dist_df["score"] = dist_df["score"].astype(int).astype(str)

    # Deterministic hue order: year1-langs..., year2-langs...
    hue_order = [f"{y}-{l}" for y in sorted(dist_df["year"].unique()) for l in LANGS]

    sns.barplot(
        data=dist_df,
        x="score",
        y="count",
        hue="year_lang",
        hue_order=[h for h in hue_order if h in set(dist_df["year_lang"])],
    )

    plt.xlabel(f"{criterion.capitalize()} score")
    plt.ylabel("Count of (qid, pid) pairs")
    plt.title(
        f"{criterion.capitalize()} distribution by year+language\n"
        f"{MODEL}, trec_dl_{', '.join(TREC_DL_YEARS)}"
    )
    plt.tight_layout()

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()

    print(f"[DONE] Saved PNG to {out_png}")
    print(f"[DONE] Saved SVG to {out_svg}")


def main() -> None:
    # Verify each year's criterion dir exists
    for y in TREC_DL_YEARS:
        d = criterion_dir_for_year(y)
        if not d.exists():
            print(f"[FATAL] Criterion dir not found for year {y}: {d}")
            sys.exit(1)

    dist_df = build_label_distribution(TREC_DL_YEARS, LANGS, CRITERION)

    # Print a pivot per year (easier to read than one giant table)
    for y in TREC_DL_YEARS:
        sub = dist_df[dist_df["year"] == y]
        print(f"\n[INFO] Distribution table for {y}:\n",
              sub.pivot(index="score", columns="language", values="count"))

    # Output dir includes both years + langs
    year_part = "_".join(TREC_DL_YEARS)
    lang_part = "_".join(LANGS)
    fig_dir = PROJECT_ROOT / "figures" / f"{year_part}" / MODEL / "criterion_compare" / lang_part
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_base = fig_dir / f"{MODEL}_trecdl_{year_part}_{CRITERION}"
    out_png = fig_base.with_suffix(".png")
    out_svg = fig_base.with_suffix(".svg")

    plot_grouped_bars(dist_df, out_png, out_svg, CRITERION)


if __name__ == "__main__":
    main()
