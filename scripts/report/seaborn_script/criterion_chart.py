#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========= Config you can edit =========
TREC_DL_YEAR = "2021"
MODEL        = "gpt-oss-20b"  # e.g. "gpt-oss-20b", "qwen3-32b-v1", etc.

# Languages to compare (matching filenames: ..._<lang>_<criterion>_labels.csv)
LANGS        = ["raw", "eng", "vi_corrected", "he_corrected", "fr", "zh"]

# Which criterion to plot: "contextuality", "coverage", "exactness", or "topicality"
CRITERION    = "contextuality"

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

# Where to save the figure (into a folder named by the combined langs)
LANG_PART = "_".join(LANGS)
FIG_DIR  = PROJECT_ROOT / "figures" / TREC_DL_YEAR / MODEL / "criterion_compare" / LANG_PART
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Save base name once; we’ll write both .png and .svg
FIG_BASE = FIG_DIR / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{CRITERION}"
FIG_PATH_PNG = FIG_BASE.with_suffix(".png")
FIG_PATH_SVG = FIG_BASE.with_suffix(".svg")

# Invalid labels outputs
INVALID_DIR = FIG_DIR / "invalid_labels"
INVALID_DIR.mkdir(parents=True, exist_ok=True)
INVALID_CSV = INVALID_DIR / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{CRITERION}_invalid_labels.csv"
INVALID_TXT = INVALID_DIR / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{CRITERION}_invalid_summary.txt"
# ======================================


def load_df_for_lang(lang: str, criterion: str) -> Tuple[pd.DataFrame, Path]:
    """
    Load one language's criterion file as a DataFrame.
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

    return df, path


def load_lang_series(lang: str, criterion: str) -> pd.Series:
    """
    Load one language's criterion file and return a Series of VALID numeric labels (0..3).
    """
    df, _ = load_df_for_lang(lang, criterion)

    s = df[criterion].astype(str).str.strip()
    s = s[s != ""]
    s = pd.to_numeric(s, errors="coerce").dropna()

    # keep integer-like only, then cast
    s = s[s % 1 == 0].astype(int)

    # keep expected label set
    s = s[s.isin([0, 1, 2, 3])]
    return s


def collect_invalid_labels(
    langs: List[str],
    criterion: str,
    valid_labels: Set[int] = {0, 1, 2, 3},
) -> pd.DataFrame:
    """
    Collect rows where the criterion label is invalid:
      - empty/whitespace
      - non-numeric
      - numeric but non-integer (e.g., 2.5)
      - integer but out of range (not in valid_labels)

    Output DF includes: language, file, row_index, raw_value, reason (+ qid/pid/query/passage if present)
    """
    invalid_rows: List[Dict[str, object]] = []
    keep_if_exists = ["qid", "pid", "query", "passage"]

    for lang in langs:
        df, path = load_df_for_lang(lang, criterion)

        raw = df[criterion]
        raw_str = raw.astype(str)
        trimmed = raw_str.str.strip()

        # classify invalids
        is_empty = trimmed.eq("") | raw.isna()

        numeric = pd.to_numeric(trimmed, errors="coerce")
        is_non_numeric = (~is_empty) & numeric.isna()

        is_non_integer_numeric = numeric.notna() & (numeric % 1 != 0)

        int_like = numeric.notna() & (numeric % 1 == 0)
        numeric_int = numeric.where(int_like).astype("Int64")  # safe NA-capable integer dtype
        is_out_of_range = int_like & (~numeric_int.isin(list(valid_labels)))

        is_invalid = is_empty | is_non_numeric | is_non_integer_numeric | is_out_of_range

        n_bad = int(is_invalid.sum())
        if n_bad == 0:
            print(f"[INFO] No invalid labels found for {lang}.")
            continue

        print(f"[INFO] Invalid labels for {lang}: {n_bad}")
        sub = df.loc[is_invalid].copy()

        for idx, row in sub.iterrows():
            raw_val = row.get(criterion)
            raw_val_str = "" if pd.isna(raw_val) else str(raw_val)

            # determine reason(s)
            reasons: List[str] = []
            t = raw_val_str.strip()

            if t == "" or pd.isna(raw_val):
                reasons.append("empty")
            else:
                num = pd.to_numeric(pd.Series([t]), errors="coerce").iloc[0]
                if pd.isna(num):
                    reasons.append("non_numeric")
                else:
                    if num % 1 != 0:
                        reasons.append("non_integer_numeric")
                    else:
                        as_int = int(num)
                        if as_int not in valid_labels:
                            reasons.append(f"out_of_range({as_int})")

            out: Dict[str, object] = {
                "language": lang,
                "file": path.name,
                "row_index": int(idx),
                "criterion": criterion,
                "raw_value": raw_val_str,
                "reason": "|".join(reasons) if reasons else "unknown",
            }

            for k in keep_if_exists:
                if k in row.index:
                    out[k] = row.get(k)

            invalid_rows.append(out)

    if not invalid_rows:
        return pd.DataFrame(
            columns=["language", "file", "row_index", "criterion", "raw_value", "reason", "qid", "pid", "query", "passage"]
        )

    df_out = pd.DataFrame(invalid_rows)

    core = ["language", "file", "row_index", "criterion", "raw_value", "reason"]
    rest = [c for c in df_out.columns if c not in core]
    return df_out[core + rest]


def write_invalid_outputs(invalid_df: pd.DataFrame, out_csv: Path, out_txt: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # always write CSV (even if empty) for reproducibility
    invalid_df.to_csv(out_csv, index=False, encoding="utf-8")

    if invalid_df.empty:
        out_txt.write_text("No invalid labels found.\n", encoding="utf-8")
        print(f"[DONE] No invalid labels. Wrote empty CSV → {out_csv}")
        print(f"[DONE] Wrote summary → {out_txt}")
        return

    summary_lines: List[str] = []
    summary_lines.append(f"Model={MODEL}  Year={TREC_DL_YEAR}  Criterion={CRITERION}")
    summary_lines.append(f"Total invalid rows: {len(invalid_df)}")
    summary_lines.append("")

    summary_lines.append("Invalid rows by language:")
    by_lang = invalid_df.groupby("language").size().sort_values(ascending=False)
    for lang, n in by_lang.items():
        summary_lines.append(f"  {lang}: {n}")
    summary_lines.append("")

    summary_lines.append("Invalid rows by reason:")
    by_reason = invalid_df.groupby("reason").size().sort_values(ascending=False)
    for reason, n in by_reason.items():
        summary_lines.append(f"  {reason}: {n}")
    summary_lines.append("")

    out_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[DONE] Wrote invalid labels CSV → {out_csv}")
    print(f"[DONE] Wrote summary TXT → {out_txt}")


def build_label_distribution(langs: List[str], criterion: str) -> pd.DataFrame:
    """
    Build a tidy DataFrame with columns: score, language, count
    from the per-language criterion label files (VALID labels only).
    """
    rows: List[Dict[str, object]] = []
    labels = [0, 1, 2, 3]

    for lang in langs:
        series = load_lang_series(lang, criterion)
        value_counts = series.value_counts().to_dict()

        for lab in labels:
            rows.append({"score": lab, "language": lang, "count": int(value_counts.get(lab, 0))})

    return pd.DataFrame(rows)


def plot_grouped_bars(dist_df: pd.DataFrame, out_png: Path, out_svg: Path, criterion: str) -> None:
    """
    Plot grouped bar chart:
      x = score, y = count, hue = language
    Save both PNG and SVG.
    """
    if dist_df.empty:
        print("[WARN] No data to plot.")
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))

    dist_df["score"] = dist_df["score"].astype(int).astype(str)

    sns.barplot(
        data=dist_df,
        x="score",
        y="count",
        hue="language",
    )

    plt.xlabel(f"{criterion.capitalize()} score")
    plt.ylabel("Count of (qid, pid) pairs")
    plt.title(f"{criterion.capitalize()} distribution by language\n{MODEL}, trec_dl_{TREC_DL_YEAR}")
    plt.tight_layout()

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    plt.close()

    print(f"[DONE] Saved PNG to {out_png}")
    print(f"[DONE] Saved SVG to {out_svg}")


def main() -> None:
    if not CRITERION_DIR.exists():
        print(f"[FATAL] Criterion dir not found: {CRITERION_DIR}")
        sys.exit(1)

    # 1) Dump invalid labels
    invalid_df = collect_invalid_labels(LANGS, CRITERION, valid_labels={0, 1, 2, 3})
    write_invalid_outputs(invalid_df, INVALID_CSV, INVALID_TXT)

    # 2) Normal distribution plot (valid labels only)
    dist_df = build_label_distribution(LANGS, CRITERION)
    print("[INFO] Plotting criterion:", CRITERION)
    print(
        "[INFO] Distribution table:\n",
        dist_df.pivot(index="score", columns="language", values="count")
    )

    plot_grouped_bars(dist_df, FIG_PATH_PNG, FIG_PATH_SVG, CRITERION)


if __name__ == "__main__":
    main()
