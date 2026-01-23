#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm

from settings import apply_paper_fmt, paper_fmt

# =========================
# Config
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

# Use one or more years to combine.
YEARS = ["2022", "2021"]
INPUT_ROOT = PROJECT_ROOT / "outputs" / "token"

YEAR_LABEL = "_".join(YEARS)
OUTPUT_PDF = PROJECT_ROOT / "figures" / YEAR_LABEL / f"avg_fertility_by_lang_model_base_{YEAR_LABEL}_sd_only.pdf"
OUTPUT_PNG = PROJECT_ROOT / "figures" / f"avg_fertility_by_lang_model_{YEAR_LABEL}_sd_only.png"

AUTO_DISCOVER_MODELS = True
MODELS = ["gpt", "qwen", "llama"]

FERTILITY_COL = "fertility_score"
TAXONOMY_CSV = PROJECT_ROOT / "scripts" / "report" / "seaborn_script" / "lang.csv"
EXCLUDE_LANGS = {"All"}
EXCLUDE_LANGS_NORM = {l.strip().casefold() for l in EXCLUDE_LANGS}

# Plot settings
Y_MIN: float | None = 1.0  # set to None to auto-scale
FIGSIZE = (12, 4)
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 11

# Per-model colors (Matplotlib accepts hex). If missing, falls back to default.
MODEL_COLORS = {
    "gpt": "#4c72b0",
    "qwen": "#55a868",
    "llama": "#c44e52",
}

# Vertical separators between language categories
DRAW_LANG_SEPARATORS = True
SEPARATOR_COLOR = "0.85"
SEPARATOR_LINEWIDTH = 0.8

# -------------------------
# NEW: derived variants control
# -------------------------
# Baseline: lang (e.g., "ga")
# Derived variants: "<lang>_word", "<lang>_first" (also supports "_crit", "_corrected", "_last" if present)
#
# Examples:
#   - Only baseline: ["base"]
#   - Compare base vs word: ["base", "word"]
#   - Compare base vs first: ["base", "first"]
#   - Compare base vs word vs first: ["base", "word", "first"]
INCLUDE_VARIANTS = ["base"]  # change to ["base","word","first"] when needed

# Display strategy:
# - True: x-axis categories are (lang,variant) e.g. "ga", "ga_word", "ga_first"
#         and models are dodged within each category (cleaner for comparison).
# - False: x-axis is just lang; both variant and model are dodged (can get busy).
VARIANT_ON_X_AXIS = True

# Optional: how to order variants within each language
VARIANT_ORDER = ["base", "word", "first", "crit", "corrected", "last"]


# =========================
# Helpers
# =========================
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


def warn_if_fonts_missing(fonts: list[str]) -> None:
    missing: list[str] = []
    for font in fonts:
        try:
            fm.findfont(fm.FontProperties(family=font), fallback_to_default=False)
        except Exception:
            missing.append(font)
    if missing:
        print(f"[WARN] Missing fonts: {', '.join(missing)}")


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


def summarize_sd(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize per (lang, variant, model) mean and standard deviation.
    Output columns:
      lang, variant, model, mean, sd, n
    """
    rows: list[dict[str, object]] = []

    for (lang, variant, model), g in raw_df.groupby(["lang", "variant", "model"], sort=False):
        x = g[FERTILITY_COL].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue

        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if x.size >= 2 else 0.0

        rows.append(
            {
                "lang": lang,
                "variant": variant,
                "model": model,
                "mean": mean,
                "sd": sd,
                "n": int(x.size),
            }
        )

    if not rows:
        raise ValueError("After cleaning, no data remained to summarize.")

    return pd.DataFrame(rows)


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


def plot_sd_bars_only(summary_df: pd.DataFrame) -> None:
    """
    Plot ONLY error bars representing ±1 SD around the mean (no markers).
    Supports comparing variants like base vs word vs first.
    """
    sns.set_theme(style="whitegrid")
    plt.style.use("default")
    apply_paper_fmt()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    mean_handle = Line2D(
        [0],
        [0],
        marker="D",
        linestyle="none",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor="0.2",
        label="Mean",
    )

    # base language order
    langs = sorted(summary_df["lang"].unique().tolist())
    lang_order = load_taxonomy_order(langs) or langs

    # which models/variants are present
    models = discover_models()
    models_present = [m for m in models if m in set(summary_df["model"].unique())]
    if not models_present:
        models_present = sorted(summary_df["model"].unique().tolist())

    variants_present = [v for v in VARIANT_ORDER if v in set(summary_df["variant"].unique())]
    if not variants_present:
        variants_present = sorted(summary_df["variant"].unique().tolist())

    # Build x categories
    if VARIANT_ON_X_AXIS:
        x_pairs: list[tuple[str, str]] = []
        for lang in lang_order:
            for v in variants_present:
                x_pairs.append((lang, v))

        x_base = np.arange(len(x_pairs), dtype=float)
        xticklabels = [
            f"{lang}_{v}" if v != "base" else lang
            for (lang, v) in x_pairs
        ]

        # dodge by model within each (lang,variant)
        n_models = max(1, len(models_present))
        dodge = 0.25
        offsets = np.linspace(-dodge, dodge, n_models) if n_models > 1 else np.array([0.0])

        key_to_row = {
            (r["lang"], r["variant"], r["model"]): r
            for r in summary_df.to_dict(orient="records")
        }

        for mi, model in enumerate(models_present):
            xs: list[float] = []
            ys: list[float] = []
            yerr: list[float] = []

            for xi, (lang, variant) in enumerate(x_pairs):
                r = key_to_row.get((lang, variant, model))
                if r is None:
                    continue

                mean = float(r["mean"])
                sd = float(r["sd"])

                xs.append(float(x_base[xi] + offsets[mi]))
                ys.append(mean)
                yerr.append(sd)

            if not xs:
                continue

            ax.errorbar(
                xs,
                ys,
                yerr=[yerr, yerr],
                fmt="none",
                capsize=3,
                elinewidth=1.2,
                color=MODEL_COLORS.get(model),
                label=model,
                zorder=2,
            )
            ax.scatter(
                xs,
                ys,
                marker="D",
                s=18,
                facecolors="none",
                edgecolors=MODEL_COLORS.get(model),
                linewidths=1.0,
                zorder=3,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(xticklabels, fontsize=TICK_FONTSIZE)
    ax.set_xlabel(
        "Language / Variant" if len(variants_present) > 1 else "Language",
        fontsize=LABEL_FONTSIZE,
    )

    # Optional: separators between every x category
    if DRAW_LANG_SEPARATORS:
        add_language_separators(ax, len(x_pairs))

    else:
        # x-axis is base language only; dodge by (variant, model) within each language
        x_base = np.arange(len(lang_order), dtype=float)
        xticklabels = lang_order

        combos = [(v, m) for v in variants_present for m in models_present]
        n = max(1, len(combos))
        dodge = 0.35
        offsets = np.linspace(-dodge, dodge, n) if n > 1 else np.array([0.0])

        key_to_row = {
            (r["lang"], r["variant"], r["model"]): r
            for r in summary_df.to_dict(orient="records")
        }

        for ci, (variant, model) in enumerate(combos):
            xs: list[float] = []
            ys: list[float] = []
            yerr: list[float] = []

            for li, lang in enumerate(lang_order):
                r = key_to_row.get((lang, variant, model))
                if r is None:
                    continue

                mean = float(r["mean"])
                sd = float(r["sd"])

                xs.append(float(x_base[li] + offsets[ci]))
                ys.append(mean)
                yerr.append(sd)

            if not xs:
                continue

            ax.errorbar(
                xs,
                ys,
                yerr=[yerr, yerr],
                fmt="none",
                capsize=3,
                elinewidth=1.2,
                color=MODEL_COLORS.get(model),
                label=f"{variant}/{model}",
                zorder=2,
            )
            ax.scatter(
                xs,
                ys,
                marker="D",
                s=18,
                facecolors="none",
                edgecolors=MODEL_COLORS.get(model),
                linewidths=1.0,
                zorder=3,
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels(xticklabels, fontsize=TICK_FONTSIZE)
        ax.set_xlabel("Language", fontsize=LABEL_FONTSIZE)

        if DRAW_LANG_SEPARATORS:
            add_language_separators(ax, len(lang_order))

    ax.set_ylabel("Mean fertility score", fontsize=LABEL_FONTSIZE)

    if Y_MIN is not None:
        ax.set_ylim(bottom=Y_MIN)

    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=False)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mean_handle)
    labels.append("Mean")
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=max(1, min(6, len(labels))),
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )

    plt.tight_layout()

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved figure to {OUTPUT_PDF}")
    print(f"[INFO] Saved figure to {OUTPUT_PNG}")


def main() -> None:
    warn_if_fonts_missing(paper_fmt.get("font.serif", []))
    raw_df = build_raw_df()
    summary_df = summarize_sd(raw_df)
    plot_sd_bars_only(summary_df)


if __name__ == "__main__":
    main()
