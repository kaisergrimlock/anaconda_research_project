#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Set

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========= Config =========
TREC_DL_YEAR = "2021"
MODEL = "gpt-oss-20b"

LANGS = ["raw", "eng", "vi_corrected", "he_corrected", "fr", "zh"]
CRITERION = "contextuality"

VALID_LABELS = {0, 1, 2, 3}

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

CRITERION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "llm_label"
    / f"trec_dl_{TREC_DL_YEAR}"
    / MODEL
    / "criterion"
)

# Plot output
LANG_PART = "_".join(LANGS)
FIG_DIR = PROJECT_ROOT / "figures" / TREC_DL_YEAR / MODEL / "criterion_compare" / LANG_PART
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIG_BASE = FIG_DIR / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{CRITERION}"
FIG_PATH_PNG = FIG_BASE.with_suffix(".png")
FIG_PATH_SVG = FIG_BASE.with_suffix(".svg")

# Invalid rows output (VERBATIM)
INVALID_DIR = FIG_DIR / "invalid_labels"
INVALID_DIR.mkdir(parents=True, exist_ok=True)
# ======================================


def load_df_for_lang(lang: str, criterion: str) -> pd.DataFrame:
    fname = f"{MODEL}_trecdl_{TREC_DL_YEAR}_{lang}_{criterion}_labels.csv"
    path = CRITERION_DIR / fname
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[INFO] Loading {path.name}")
    return pd.read_csv(path)


def invalid_llm_mask(series: pd.Series) -> pd.Series:
    """
    Boolean mask for invalid llm_relevance labels.
    """
    raw = series.astype(str).str.strip()

    is_empty = raw.eq("") | raw.isna()

    numeric = pd.to_numeric(raw, errors="coerce")
    is_non_numeric = (~is_empty) & numeric.isna()

    is_non_integer = numeric.notna() & (numeric % 1 != 0)

    int_like = numeric.notna() & (numeric % 1 == 0)
    numeric_int = numeric.where(int_like).astype("Int64")
    is_out_of_range = int_like & (~numeric_int.isin(VALID_LABELS))

    return is_empty | is_non_numeric | is_non_integer | is_out_of_range


def dump_invalid_rows(langs: List[str], criterion: str) -> None:
    """
    Dump original CSV rows with invalid llm_relevance labels.
    """
    for lang in langs:
        df = load_df_for_lang(lang, criterion)

        if "llm_relevance" not in df.columns:
            print(f"[WARN] llm_relevance missing for {lang}, skipping")
            continue

        mask = invalid_llm_mask(df["llm_relevance"])
        if not mask.any():
            print(f"[INFO] No invalid rows for {lang}")
            continue

        out_name = f"invalid_{criterion}_{lang}.csv"
        out_path = INVALID_DIR / out_name

        df.loc[mask].to_csv(out_path, index=False, encoding="utf-8")
        print(f"[DUMPED] {mask.sum()} rows → {out_path.name}")


def load_lang_series(lang: str, criterion: str) -> pd.Series:
    """
    Valid numeric labels only (used for plotting).
    """
    df = load_df_for_lang(lang, criterion)

    s = df[criterion].astype(str).str.strip()
    s = s[s != ""]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[(s % 1 == 0)].astype(int)
    s = s[s.isin(VALID_LABELS)]
    return s


def build_label_distribution(langs: List[str], criterion: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    labels = sorted(VALID_LABELS)

    for lang in langs:
        series = load_lang_series(lang, criterion)
        vc = series.value_counts().to_dict()
        for lab in labels:
            rows.append(
                {"score": lab, "language": lang, "count": int(vc.get(lab, 0))}
            )

    return pd.DataFrame(rows)


def plot_grouped_bars(dist_df: pd.DataFrame) -> None:
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))

    dist_df["score"] = dist_df["score"].astype(str)

    sns.barplot(
        data=dist_df,
        x="score",
        y="count",
        hue="language",
    )

    plt.xlabel(f"{CRITERION.capitalize()} score")
    plt.ylabel("Count of (qid, pid) pairs")
    plt.title(f"{CRITERION.capitalize()} distribution by language\n{MODEL}, trec_dl_{TREC_DL_YEAR}")
    plt.tight_layout()

    plt.savefig(FIG_PATH_PNG, dpi=300)
    plt.savefig(FIG_PATH_SVG)
    plt.close()

    print(f"[DONE] Saved PNG → {FIG_PATH_PNG}")
    print(f"[DONE] Saved SVG → {FIG_PATH_SVG}")


def main() -> None:
    if not CRITERION_DIR.exists():
        print(f"[FATAL] Criterion dir not found: {CRITERION_DIR}")
        sys.exit(1)

    # 1) Dump original invalid rows (verbatim)
    dump_invalid_rows(LANGS, CRITERION)

    # 2) Plot valid-label distribution
    dist_df = build_label_distribution(LANGS, CRITERION)
    print(
        "[INFO] Distribution table:\n",
        dist_df.pivot(index="score", columns="language", values="count"),
    )

    plot_grouped_bars(dist_df)


if __name__ == "__main__":
    main()
