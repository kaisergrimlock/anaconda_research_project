#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm

from settings import apply_paper_fmt

# ============================================================
# Config
# ============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

# Use one or more years to combine.
YEARS = ["2022", "2021"]
INPUT_ROOT = PROJECT_ROOT / "outputs" / "token"

YEAR_LABEL = "_".join(YEARS)
OUTPUT_PDF = (
    PROJECT_ROOT
    / "figures"
    / YEAR_LABEL
    / f"avg_fertility_by_lang_model_base_{YEAR_LABEL}_sd_only.pdf"
)
OUTPUT_PNG = (
    PROJECT_ROOT
    / "figures"
    / f"avg_fertility_by_lang_model_{YEAR_LABEL}_sd_only.png"
)

AUTO_DISCOVER_MODELS = True
MODELS = ["gpt", "qwen", "llama"]

FERTILITY_COL = "fertility_score"
TAXONOMY_CSV = PROJECT_ROOT / "scripts" / "report" / "seaborn_script" / "lang.csv"
EXCLUDE_LANGS = {"All"}
EXCLUDE_LANGS_NORM = {l.strip().casefold() for l in EXCLUDE_LANGS}

# Plot settings
Y_MIN: float | None = 1.0  # set to None to auto-scale
FIGSIZE = (12, 4)
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 15

# Per-model colors (Matplotlib accepts hex). If missing, seaborn uses defaults.
MODEL_COLORS: dict[str, str] = {
    "gpt": "#4c72b0",
    "qwen": "#55a868",
    "llama": "#c44e52",
}

# Vertical separators between x categories
DRAW_LANG_SEPARATORS = True
SEPARATOR_COLOR = "0.85"
SEPARATOR_LINEWIDTH = 0.8

# ------------------------------------------------------------
# Variant control
# ------------------------------------------------------------
# Baseline: lang (e.g., "ga")
# Derived variants: "<lang>_word", "<lang>_first" (also supports "_crit", "_corrected", "_last" if present)
INCLUDE_VARIANTS = ["base"]  # e.g., ["base","word","first"]

# Display strategy:
# - True: x-axis categories are (lang,variant) e.g. "ga", "ga_word", "ga_first"
#         and models are dodged within each category.
# - False: x-axis is just lang; still dodged by model (variant not visualized).
VARIANT_ON_X_AXIS = True

# Optional: how to order variants within each language
VARIANT_ORDER = ["base", "word", "first", "crit", "corrected", "last"]


# ============================================================
# Helpers
# ============================================================
def discover_models() -> list[str]:
    """
    Return models to plot.
    If AUTO_DISCOVER_MODELS=True, scan INPUT_ROOT for subfolders and (if MODELS
    is non-empty) intersect with MODELS.
    """
    if not AUTO_DISCOVER_MODELS:
        return MODELS

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Missing input root: {INPUT_ROOT}")

    discovered = sorted([p.name for p in INPUT_ROOT.iterdir() if p.is_dir()])
    if MODELS:
        return [m for m in MODELS if m in discovered]
    return discovered


def parse_lang_and_variant(csv_path: Path, year: str) -> tuple[str, str] | None:
    """
    Parse filename:
      passage_tokens_<YEAR>_<lang>.csv

    Where <lang> may end with:
      _word, _first, _crit, _corrected, _last

    Returns (base_lang, variant) where variant in:
      base, word, first, crit, corrected, last
    """
    m = re.match(rf"passage_tokens_{re.escape(year)}_(.+)\.csv$", csv_path.name)
    if not m:
        return None

    name = m.group(1)

    for suf, var in [
        ("_word", "word"),
        ("_first", "first"),
        ("_crit", "crit"),
        ("_corrected", "corrected"),
        ("_last", "last"),
    ]:
        if name.endswith(suf):
            return (name[: -len(suf)], var)

    return (name, "base")


def load_taxonomy_order(langs: list[str]) -> list[str] | None:
    """
    If TAXONOMY_CSV exists (columns: lang,taxonomy), order languages by:
      taxonomy asc, then lang asc
    Any missing langs go last (sorted).
    """
    if not TAXONOMY_CSV.exists():
        return None

    tax_df = pd.read_csv(TAXONOMY_CSV)
    if "lang" not in tax_df.columns or "taxonomy" not in tax_df.columns:
        return None

    tax_df["lang"] = tax_df["lang"].astype(str).str.strip()
    tax_df["taxonomy"] = pd.to_numeric(tax_df["taxonomy"], errors="coerce")
    tax_df = tax_df.dropna(subset=["taxonomy"])
    tax_df["taxonomy"] = tax_df["taxonomy"].astype(int)

    lang_set = set(langs)
    tax_df = tax_df[tax_df["lang"].isin(lang_set)]
    if tax_df.empty:
        return None

    ordered = tax_df.sort_values(["taxonomy", "lang"])["lang"].tolist()
    missing = [l for l in langs if l not in set(ordered)]
    return ordered + sorted(missing)


def read_fertility_values(csv_path: Path) -> np.ndarray:
    """
    Read fertility_score column as numeric array (drop NaNs).
    """
    df = pd.read_csv(csv_path)
    if FERTILITY_COL not in df.columns:
        raise ValueError(f"Column {FERTILITY_COL!r} not found in {csv_path}")

    s = pd.to_numeric(df[FERTILITY_COL], errors="coerce").dropna()
    return s.to_numpy(dtype=float)


def build_raw_df() -> pd.DataFrame:
    """
    Build long-form dataframe with raw values:
      lang | variant | model | fertility_score
    """
    frames: list[pd.DataFrame] = []
    allowed = set(INCLUDE_VARIANTS)

    for model in discover_models():
        model_root = INPUT_ROOT / model
        if not model_root.exists():
            continue

        for year in YEARS:
            for csv_path in model_root.glob(f"passage_tokens_{year}_*.csv"):
                parsed = parse_lang_and_variant(csv_path, year)
                if parsed is None:
                    continue

                lang, variant = parsed
                if variant not in allowed:
                    continue

                vals = read_fertility_values(csv_path)
                if vals.size == 0:
                    continue

                frames.append(
                    pd.DataFrame(
                        {
                            "lang": lang,
                            "variant": variant,
                            "model": model,
                            FERTILITY_COL: vals,
                        }
                    )
                )

    if not frames:
        raise ValueError(
            "No matching CSVs found. Check YEAR, INPUT_ROOT, INCLUDE_VARIANTS, and filenames."
        )

    raw_df = pd.concat(frames, ignore_index=True)

    if EXCLUDE_LANGS_NORM:
        lang_norm = raw_df["lang"].astype(str).str.strip().str.casefold()
        raw_df = raw_df[~lang_norm.isin(EXCLUDE_LANGS_NORM)]

    return raw_df


def add_language_separators(ax: plt.Axes, n_cats: int) -> None:
    """
    Draw vertical separators between x categories at x = i + 0.5.
    Assumes ticks at 0..n_cats-1.
    """
    if n_cats <= 1:
        return
    for i in range(n_cats - 1):
        ax.axvline(
            i + 0.5,
            color=SEPARATOR_COLOR,
            linewidth=SEPARATOR_LINEWIDTH,
            zorder=0,
        )


def make_x_category(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'x_cat' column for seaborn x-axis labeling.
    """
    out = raw_df.copy()
    if VARIANT_ON_X_AXIS:
        out["x_cat"] = np.where(out["variant"].eq("base"), out["lang"], out["lang"] + "_" + out["variant"])
    else:
        out["x_cat"] = out["lang"]
    return out


def build_x_order(raw_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    Returns (x_order, lang_order, variants_present)
    """
    langs = sorted(raw_df["lang"].unique().tolist())
    lang_order = load_taxonomy_order(langs) or langs

    variants_present = [v for v in VARIANT_ORDER if v in set(raw_df["variant"].unique())]
    if not variants_present:
        variants_present = sorted(raw_df["variant"].unique().tolist())

    if VARIANT_ON_X_AXIS:
        x_order: list[str] = []
        for lang in lang_order:
            for v in variants_present:
                x_order.append(lang if v == "base" else f"{lang}_{v}")
    else:
        x_order = lang_order

    return x_order, lang_order, variants_present


def plot_sd_bars_only_from_raw(raw_df: pd.DataFrame) -> None:
    """
    Seaborn-first plot from RAW observations:
      - estimator=np.mean
      - errorbar=("sd", 1) (or ci="sd" for older seaborn)
      - hollow diamond markers
      - no connecting lines
    """
    sns.set_theme(style="whitegrid")
    plt.style.use("default")
    apply_paper_fmt()

    # Models present / ordering
    models = discover_models()
    models_present = [m for m in models if m in set(raw_df["model"].unique())]
    if not models_present:
        models_present = sorted(raw_df["model"].unique().tolist())

    # Build ordered x categories
    x_order, _lang_order, variants_present = build_x_order(raw_df)

    plot_df = make_x_category(raw_df)
    plot_df["x_cat"] = pd.Categorical(plot_df["x_cat"], categories=x_order, ordered=True)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    pointplot_kwargs = dict(
        data=plot_df,
        x="x_cat",
        y=FERTILITY_COL,
        hue="model",
        hue_order=models_present,
        order=x_order,
        palette=MODEL_COLORS,
        estimator=np.mean,
        dodge=0.5,
        linestyle="none",       # seaborn >=0.12
        markers="D",
        err_kws={"linewidth": 1.2},
        capsize=0.15,
        ax=ax,
    )

    # Seaborn compatibility: 0.12+ uses errorbar, older uses ci
    try:
        sns.pointplot(**pointplot_kwargs, errorbar=("sd", 1))
    except TypeError:
        sns.pointplot(**pointplot_kwargs, ci="sd")

    MARKERSIZE = 6
    # Hollow diamonds (seaborn uses Line2D artists)
    for line in ax.lines:
        if line.get_marker() == "D":
            line.set_markersize(MARKERSIZE)
            line.set_markerfacecolor("none")
            line.set_markeredgewidth(1.0)

    # Labels / ticks
    ax.set_xlabel(
        "Language / Variant" if (VARIANT_ON_X_AXIS and len(variants_present) > 1) else "Language",
        fontsize=LABEL_FONTSIZE,
    )
    ax.set_ylabel("Mean fertility score", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

    if Y_MIN is not None:
        ax.set_ylim(bottom=Y_MIN)

    # Optional separators between x categories
    if DRAW_LANG_SEPARATORS:
        add_language_separators(ax, len(x_order))

    # Grid tweaks
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=False)

    # Legend placement
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=max(1, min(6, len(models_present))),
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
        frameon=False,
        title=None,
    )

    plt.tight_layout()

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved figure to {OUTPUT_PDF}")
    print(f"[INFO] Saved figure to {OUTPUT_PNG}")


def main() -> None:
    raw_df = build_raw_df()
    plot_sd_bars_only_from_raw(raw_df)


if __name__ == "__main__":
    main()
