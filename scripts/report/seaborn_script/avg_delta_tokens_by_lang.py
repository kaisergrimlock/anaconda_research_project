#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]   # adjust if script lives elsewhere

YEAR = "2022"
LANGS = [
    "eng",
    "ga",
    "vi",
    "fr",
    "ru",
    "th",
    "sw",
    "he",
]
AUTO_DISCOVER = True
SKIP_MISSING = True
TAXONOMY_CSV = PROJECT_ROOT / "scripts" / "report" / "seaborn_script" / "lang.csv"

INPUT_ROOT = PROJECT_ROOT / "outputs" / "token"
OUTPUT_DELTA_PNG = PROJECT_ROOT / "figures" / f"avg_delta_token_{YEAR}.png"
OUTPUT_FERTILITY_PNG = PROJECT_ROOT / "figures" / f"avg_fertility_{YEAR}.png"

DELTA_COL = "delta_token"
FERTILITY_COL = "fertility_score"


def load_avg_delta(lang: str, column: str) -> float | None:
    csv_path = INPUT_ROOT / f"passage_tokens_{YEAR}_{lang}.csv"
    if not csv_path.exists():
        if SKIP_MISSING:
            print(f"[WARN] Missing CSV for {lang}: {csv_path}")
            return None
        raise FileNotFoundError(f"Missing CSV for {lang}: {csv_path}")

    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not found in {csv_path}")

    return float(df[column].mean())


def get_langs() -> list[str]:
    if not AUTO_DISCOVER:
        return LANGS

    pattern = f"passage_tokens_{YEAR}_*.csv"
    langs = []
    for csv_path in INPUT_ROOT.glob(pattern):
        name = csv_path.stem
        lang = name.replace(f"passage_tokens_{YEAR}_", "", 1)
        if lang:
            langs.append(lang)
    return sorted(set(langs)) or LANGS


def load_taxonomy() -> pd.DataFrame:
    if not TAXONOMY_CSV.exists():
        raise FileNotFoundError(f"Missing taxonomy file: {TAXONOMY_CSV}")

    tax_df = pd.read_csv(TAXONOMY_CSV)
    tax_df["lang"] = tax_df["lang"].astype(str).str.strip()
    tax_df["taxonomy"] = tax_df["taxonomy"].astype(str).str.strip()
    tax_df["taxonomy"] = pd.to_numeric(tax_df["taxonomy"], errors="coerce")
    tax_df = tax_df.dropna(subset=["taxonomy"])
    tax_df["taxonomy"] = tax_df["taxonomy"].astype(int)
    return tax_df[["lang", "taxonomy"]]


def build_plot_df(column: str, value_name: str) -> pd.DataFrame:
    rows = []
    for lang in get_langs():
        avg_value = load_avg_delta(lang, column)
        if avg_value is None:
            continue
        rows.append({"lang": lang, value_name: avg_value})

    plot_df = pd.DataFrame(rows)
    tax_df = load_taxonomy()
    plot_df = plot_df.merge(tax_df, on="lang", how="left")
    if plot_df["taxonomy"].isna().any():
        missing = plot_df.loc[plot_df["taxonomy"].isna(), "lang"].tolist()
        raise ValueError(f"Missing taxonomy for languages: {missing}")

    plot_df = plot_df.sort_values(["taxonomy", "lang"]).reset_index(drop=True)
    return plot_df


def plot_bars(plot_df: pd.DataFrame, value_col: str, output_png: Path, title: str, y_label: str) -> None:
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4))
    palette = {
        2: "#4c72b0",
        3: "#55a868",
        4: "#c44e52",
        5: "#8172b2",
    }
    ax = sns.barplot(
        data=plot_df,
        x="lang",
        y=value_col,
        hue="taxonomy",
        palette=palette,
        dodge=False,
    )
    ax.set_xlabel("Language")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title="Taxonomy", loc="best")

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    print(f"[INFO] Saved figure to {output_png}")


def main() -> None:
    delta_df = build_plot_df(DELTA_COL, "avg_delta")
    plot_bars(
        delta_df,
        "avg_delta",
        OUTPUT_DELTA_PNG,
        f"Average delta tokens by language ({YEAR})",
        "Average delta tokens",
    )

    fertility_df = build_plot_df(FERTILITY_COL, "avg_fertility")
    plot_bars(
        fertility_df,
        "avg_fertility",
        OUTPUT_FERTILITY_PNG,
        f"Average fertility by language ({YEAR})",
        "Average fertility score",
    )


if __name__ == "__main__":
    main()
