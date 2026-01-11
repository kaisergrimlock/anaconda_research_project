#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

YEAR = "2022"
INPUT_ROOT = PROJECT_ROOT / "outputs" / "token"
OUTPUT_PNG = PROJECT_ROOT / "figures" / f"avg_fertility_by_lang_model_{YEAR}.png"

AUTO_DISCOVER_MODELS = True
MODELS = ["gpt", "qwen", "llama"]

# Exclude derived language variants by default.
EXCLUDE_SUFFIXES = ["_word", "_crit", "_first", "_corrected"]

FERTILITY_COL = "fertility_score"
TAXONOMY_CSV = PROJECT_ROOT / "scripts" / "report" / "seaborn_script" / "lang.csv"


def discover_models() -> list[str]:
    if not AUTO_DISCOVER_MODELS:
        return MODELS

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Missing input root: {INPUT_ROOT}")

    discovered = [p.name for p in INPUT_ROOT.iterdir() if p.is_dir()]
    if MODELS:
        return [m for m in MODELS if m in discovered]
    return sorted(discovered)


def extract_lang(csv_path: Path) -> str | None:
    match = re.match(rf"passage_tokens_{re.escape(YEAR)}_(.+)\.csv$", csv_path.name)
    if not match:
        return None
    lang = match.group(1)
    if any(lang.endswith(suffix) for suffix in EXCLUDE_SUFFIXES):
        return None
    return lang


def load_mean(csv_path: Path) -> float | None:
    df = pd.read_csv(csv_path)
    if FERTILITY_COL not in df.columns:
        raise ValueError(f"Column {FERTILITY_COL!r} not found in {csv_path}")

    series = pd.to_numeric(df[FERTILITY_COL], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def load_taxonomy_order(langs: list[str]) -> list[str] | None:
    if not TAXONOMY_CSV.exists():
        return None

    tax_df = pd.read_csv(TAXONOMY_CSV)
    tax_df["lang"] = tax_df["lang"].astype(str).str.strip()
    tax_df["taxonomy"] = pd.to_numeric(tax_df["taxonomy"], errors="coerce")
    tax_df = tax_df.dropna(subset=["taxonomy"])
    tax_df["taxonomy"] = tax_df["taxonomy"].astype(int)

    lang_set = set(langs)
    tax_df = tax_df[tax_df["lang"].isin(lang_set)]
    if tax_df.empty:
        return None

    ordered = tax_df.sort_values(["taxonomy", "lang"])["lang"].tolist()
    missing = [lang for lang in langs if lang not in set(ordered)]
    return ordered + sorted(missing)


def build_plot_df() -> pd.DataFrame:
    rows = []
    for model in discover_models():
        model_root = INPUT_ROOT / model
        pattern = f"passage_tokens_{YEAR}_*.csv"
        for csv_path in model_root.glob(pattern):
            lang = extract_lang(csv_path)
            if not lang:
                continue
            mean = load_mean(csv_path)
            if mean is None:
                continue
            rows.append({"lang": lang, "model": model, "avg_fertility": mean})

    if not rows:
        raise ValueError("No matching CSVs found with fertility scores.")

    return pd.DataFrame(rows)


def plot_bars(plot_df: pd.DataFrame) -> None:
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 4))

    lang_order = load_taxonomy_order(sorted(plot_df["lang"].unique().tolist()))
    ax = sns.barplot(
        data=plot_df,
        x="lang",
        y="avg_fertility",
        hue="model",
        order=lang_order,
        dodge=True,
    )
    ax.set_xlabel("Language")
    ax.set_ylabel("Mean fertility score")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=max(1, len(labels)),
        frameon=False,
    )

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved figure to {OUTPUT_PNG}")


def main() -> None:
    plot_df = build_plot_df()
    plot_bars(plot_df)


if __name__ == "__main__":
    main()
