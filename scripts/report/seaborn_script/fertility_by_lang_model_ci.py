#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Config
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

# Use one or more years to combine.
YEARS = ["2022", "2021"]
INCLUDE_VARIANTS = ["base"]
INPUT_ROOT = PROJECT_ROOT / "outputs" / "token"
YEAR_LABEL = "_".join(YEARS)
OUTPUT_PDF = PROJECT_ROOT / "figures" / YEAR_LABEL / f"avg_fertility_by_lang_model_{INCLUDE_VARIANTS[0]}_{YEAR_LABEL}_ci95_only.pdf"
OUTPUT_PNG = PROJECT_ROOT / "figures" / f"avg_fertility_by_lang_model_{YEAR_LABEL}_ci95_only.png"

AUTO_DISCOVER_MODELS = True
MODELS = ["llama", "qwen", "gpt"]

FERTILITY_COL = "fertility_score"
TAXONOMY_CSV = PROJECT_ROOT / "scripts" / "report" / "seaborn_script" / "lang.csv"
EXCLUDE_LANGS = {"All"}
EXCLUDE_LANGS_NORM = {l.strip().casefold() for l in EXCLUDE_LANGS}

# CI settings (bootstrap percentile CI for the mean)
CI_LEVEL = 95
N_BOOT = 2000
RNG_SEED = 7

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
# Variants are derived from filename suffixes: "_word", "_first", "_crit", "_corrected"
# Set which variants you want to INCLUDE in the plot.
#
# Examples:
#   - Only baseline: ["base"]
#   - Compare base vs word: ["base", "word"]
#   - Compare base vs first: ["base", "first"]
#   - Compare base vs word vs first: ["base", "word", "first"]


# How to display variants
# - If True: treat variant as its own x-category: (lang, variant) on x-axis, models dodged within.
# - If False: keep x as language, and variant becomes an additional dodge level (more clutter).
VARIANT_ON_X_AXIS = True

# Variant colors (optional): if set, CI bars get colored by variant, and model is indicated by linestyle.
# If False: keep your old behavior (color by model).
COLOR_BY_VARIANT = False

VARIANT_COLORS = {
    "base": "#000000",
    "word": "#8c8c8c",
    "first": "#b2b2b2",
    "crit": "#d0d0d0",
    "corrected": "#e0e0e0",
}

MODEL_LINESTYLES = {
    "gpt": "-",
    "qwen": "--",
    "llama": ":",
}

# =========================
# Helpers
# =========================
def discover_models() -> list[str]:
    """
    Return the list of models to include.
    If AUTO_DISCOVER_MODELS=True, uses the subfolders of INPUT_ROOT and
    intersects with MODELS (if MODELS is non-empty).
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
    where <lang> may have a derived suffix:
      <base>_word, <base>_first, <base>_crit, <base>_corrected

    Returns: (base_lang, variant) where variant in {"base","word","first","crit","corrected"}
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
    ]:
        if name.endswith(suf):
            return (name[: -len(suf)], var)

    return (name, "base")


def load_taxonomy_order(langs: list[str]) -> list[str] | None:
    """
    If TAXONOMY_CSV exists and has columns (lang,taxonomy), order langs by
    taxonomy asc then lang asc; append missing langs at the end.
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
    Read fertility_score column as a numeric array, dropping NaNs.
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


def bootstrap_ci_mean(
    x: np.ndarray,
    *,
    ci_level: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for the mean.
    Returns (mean, lo, hi).
    """
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    mean = float(x.mean())
    if x.size == 1:
        return (mean, mean, mean)

    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot_means = x[idx].mean(axis=1)

    alpha = 100 - ci_level
    lo = float(np.percentile(boot_means, alpha / 2))
    hi = float(np.percentile(boot_means, 100 - alpha / 2))
    return (mean, lo, hi)


def summarize_ci(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize per (lang, variant, model) mean + bootstrap CI.
    Output columns:
      lang, variant, model, mean, lo, hi, n
    """
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, object]] = []

    for (lang, variant, model), g in raw_df.groupby(["lang", "variant", "model"], sort=False):
        x = g[FERTILITY_COL].to_numpy(dtype=float)
        mean, lo, hi = bootstrap_ci_mean(x, ci_level=CI_LEVEL, n_boot=N_BOOT, rng=rng)
        if np.isnan(mean):
            continue
        rows.append(
            {
                "lang": lang,
                "variant": variant,
                "model": model,
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "n": int(np.isfinite(x).sum()),
            }
        )

    if not rows:
        raise ValueError("After cleaning, no data remained to summarize.")

    return pd.DataFrame(rows)


def add_language_separators(ax: plt.Axes, n_cats: int) -> None:
    """
    Draw vertical separators between x categories at x = i + 0.5.
    Assumes x ticks are at 0..n_cats-1.
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


def make_x_order(summary_df: pd.DataFrame) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Returns:
      - ordered base langs
      - ordered x-categories as (lang, variant)
    """
    langs = sorted(summary_df["lang"].unique().tolist())
    lang_order = load_taxonomy_order(langs) or langs

    variants_present = [v for v in INCLUDE_VARIANTS if v in set(summary_df["variant"].unique())]

    x_pairs: list[tuple[str, str]] = []
    if VARIANT_ON_X_AXIS:
        for lang in lang_order:
            for v in variants_present:
                x_pairs.append((lang, v))
    else:
        # x only uses lang, but we still return pairs in a consistent way (lang, "")
        x_pairs = [(lang, "") for lang in lang_order]

    return lang_order, x_pairs


def plot_ci_bars_only(summary_df: pd.DataFrame) -> None:
    """
    Plot ONLY confidence interval bars (no markers).
    Supports comparing variants like base vs word vs first.
    """
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))

    models = discover_models()
    models_present = [m for m in models if m in set(summary_df["model"].unique())]
    if not models_present:
        models_present = sorted(summary_df["model"].unique().tolist())

    lang_order, x_pairs = make_x_order(summary_df)

    # x positions
    x_base = np.arange(len(x_pairs), dtype=float)

    # label each x tick
    if VARIANT_ON_X_AXIS:
        tick_labels = [f"{lang}_{variant}" if variant != "base" else lang for lang, variant in x_pairs]
    else:
        tick_labels = lang_order

    # dodging:
    # - if VARIANT_ON_X_AXIS: dodge by model only
    # - else: dodge by (variant, model)
    if VARIANT_ON_X_AXIS:
        n_dodge = max(1, len(models_present))
        dodge = 0.25
        offsets = np.linspace(-dodge, dodge, n_dodge) if n_dodge > 1 else np.array([0.0])

        key_to_row = {
            (r["lang"], r["variant"], r["model"]): r
            for r in summary_df.to_dict(orient="records")
        }

        for mi, model in enumerate(models_present):
            xs: list[float] = []
            ys: list[float] = []
            yerr_lo: list[float] = []
            yerr_hi: list[float] = []

            for xi, (lang, variant) in enumerate(x_pairs):
                r = key_to_row.get((lang, variant, model))
                if r is None:
                    continue

                mean = float(r["mean"])
                lo = float(r["lo"])
                hi = float(r["hi"])

                xs.append(float(x_base[xi] + offsets[mi]))
                ys.append(mean)
                yerr_lo.append(mean - lo)
                yerr_hi.append(hi - mean)

            if not xs:
                continue

            if COLOR_BY_VARIANT:
                # if variant-on-x, color_by_variant doesn't make sense per-series; keep by model
                color = MODEL_COLORS.get(model)
                linestyle = "-"
                label = model
            else:
                color = MODEL_COLORS.get(model)
                linestyle = "-"
                label = model

            ax.errorbar(
                xs,
                ys,
                yerr=[yerr_lo, yerr_hi],
                fmt="none",
                capsize=3,
                elinewidth=1.2,
                color=color,
                linestyle=linestyle,
                label=label,
                zorder=2,
            )

    else:
        # x is lang; we dodge by (variant, model) -> can get busy, but works
        variants_present = [v for v in INCLUDE_VARIANTS if v in set(summary_df["variant"].unique())]
        combos = [(v, m) for v in variants_present for m in models_present]
        n_dodge = max(1, len(combos))
        dodge = 0.35
        offsets = np.linspace(-dodge, dodge, n_dodge) if n_dodge > 1 else np.array([0.0])

        key_to_row = {
            (r["lang"], r["variant"], r["model"]): r
            for r in summary_df.to_dict(orient="records")
        }

        for ci, (variant, model) in enumerate(combos):
            xs: list[float] = []
            ys: list[float] = []
            yerr_lo: list[float] = []
            yerr_hi: list[float] = []

            for li, lang in enumerate(lang_order):
                r = key_to_row.get((lang, variant, model))
                if r is None:
                    continue

                mean = float(r["mean"])
                lo = float(r["lo"])
                hi = float(r["hi"])

                xs.append(float(li + offsets[ci]))
                ys.append(mean)
                yerr_lo.append(mean - lo)
                yerr_hi.append(hi - mean)

            if not xs:
                continue

            if COLOR_BY_VARIANT:
                color = VARIANT_COLORS.get(variant)
                linestyle = MODEL_LINESTYLES.get(model, "-")
                label = f"{variant}/{model}"
            else:
                color = MODEL_COLORS.get(model)
                linestyle = MODEL_LINESTYLES.get(variant, "-") if isinstance(MODEL_LINESTYLES.get(variant, None), str) else "-"
                label = f"{variant}/{model}"

            ax.errorbar(
                xs,
                ys,
                yerr=[yerr_lo, yerr_hi],
                fmt="none",
                capsize=3,
                elinewidth=1.2,
                color=color,
                linestyle=linestyle,
                label=label,
                zorder=2,
            )

    # Axes / labels
    ax.set_xticks(x_base if VARIANT_ON_X_AXIS else np.arange(len(lang_order), dtype=float))
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel("Language" + (" / Variant" if VARIANT_ON_X_AXIS and len(INCLUDE_VARIANTS) > 1 else ""))
    ax.set_ylabel("Mean fertility score")

    # Separators between categories
    if DRAW_LANG_SEPARATORS:
        add_language_separators(ax, len(x_pairs) if VARIANT_ON_X_AXIS else len(lang_order))

    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=False)

    # legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=max(1, min(6, len(ax.get_legend_handles_labels()[1]))),
        frameon=False,
    )

    plt.tight_layout()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved figure to {OUTPUT_PDF}")
    print(f"[INFO] Saved figure to {OUTPUT_PNG}")


def main() -> None:
    raw_df = build_raw_df()
    summary_df = summarize_ci(raw_df)
    plot_ci_bars_only(summary_df)


if __name__ == "__main__":
    main()
