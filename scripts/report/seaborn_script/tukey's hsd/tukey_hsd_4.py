#!/usr/bin/env python3
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from helpers.draw import (
    color_tukey_by_taxonomy,
    center_x_axis_at_zero,
    taxonomy_legend,
)

# =========================
# File Location
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit
from helpers.output_writer import write_df

# =========================
# Config
# =========================
TREC_DL_YEAR = "2022"
LABEL_ROOT = Path("outputs/llm_label") / f"trec_dl_{TREC_DL_YEAR}"
OUT_DIR = Path("figures") / TREC_DL_YEAR / "tukey_hsd" / "all_models_all_langs_mean"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TUKEY_CSV = OUT_DIR / "tukey_hsd_table_all_groups.csv"
OUT_TUKEY_TEX = OUT_DIR / "tukey_hsd_table_all_groups.tex"
OUT_SIMUL_SVG = OUT_DIR / "tukey_hsd_plot_simultaneous_all_groups_1.svg"
OUT_SAMPLES = OUT_DIR / "tukey_samples_long.csv"

GROUP_SEP = "|"
TAXONOMY_CSV = Path(__file__).resolve().parents[1] / "lang.csv"

# ========================
# Parameters
# ========================
ALPHA = 0.05
LABELS = [0, 1, 2, 3]
LANGS: List[str] = ["raw", "eng", "ru", "vi", "th", "sw"]  # strict include list
# LANGS: List[str] = ["raw", "eng", "fr", "ru", "ar", "vi", "th", "sw", "ga"]

# NEW: compute mean label (LLM or NIST) instead of mean difference
# Options: "mean_llm" or "mean_nist"
METRIC = "mean_llm"


def find_llm_files() -> Dict[str, List[Path]]:
    """
    Returns model -> list of label CSVs.
    Expected layout:
      outputs/llm_label/trec_dl_<YEAR>/<MODEL>/<MODEL>_trecdl_<YEAR>_<LANG>_labels.csv
    """
    if not LABEL_ROOT.exists():
        raise FileNotFoundError(f"LABEL_ROOT not found: {LABEL_ROOT}")

    model_files: Dict[str, List[Path]] = {}
    for model_dir in LABEL_ROOT.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        csv_files = list(model_dir.glob(f"{model_name}_trecdl_{TREC_DL_YEAR}_*_labels.csv"))
        if csv_files:
            model_files[model_name] = csv_files
        else:
            print(f"Warning: no label CSVs found for model {model_name} in {model_dir}")

    return model_files


def get_lang_from_filename(file_path: Path, model: str) -> Optional[str]:
    """
    Extract <LANG> from:
      <MODEL>_trecdl_<YEAR>_<LANG>_labels.csv
    """
    fname = file_path.name
    pattern = rf"^{re.escape(model)}_trecdl_\d{{4}}_(.+?)_labels\.csv$"
    match = re.search(pattern, fname)
    return match.group(1) if match else None


def load_labels(file_path: Path) -> pd.DataFrame:
    bump_field_limit()
    df = pd.read_csv(file_path)

    requisite = {"qid", "pid", "relevance", "llm_relevance"}
    for r in requisite:
        if r not in df.columns:
            raise ValueError(f"Missing required column '{r}' in {file_path}")

    df["NIST"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["LLM"] = pd.to_numeric(df["llm_relevance"], errors="coerce")

    df = df.dropna(subset=["NIST", "LLM"])
    df["NIST"] = df["NIST"].astype(int)
    df["LLM"] = df["LLM"].astype(int)

    valid = df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)
    return df[valid].copy()


def per_qid_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Turn a (qid,pid)-level label dataframe into (qid)-level samples:
      qid, value

    These 'value' samples are what Tukey HSD consumes.
    """
    # Ensure unique (qid,pid) pairs so duplicates don't distort per-qid averages
    base = df.drop_duplicates(subset=["qid", "pid"]).copy()

    if metric == "mean_llm":
        # mean LLM label per query
        return (
            base.groupby("qid", as_index=False)["LLM"]
            .mean()
            .rename(columns={"LLM": "value"})
        )

    if metric == "mean_nist":
        # mean NIST label per query (useful as a sanity baseline)
        return (
            base.groupby("qid", as_index=False)["NIST"]
            .mean()
            .rename(columns={"NIST": "value"})
        )

    # (kept for convenience; you can delete these if you want)
    if metric == "mae_4pt":
        base["abs_err"] = (base["LLM"] - base["NIST"]).abs()
        return (
            base.groupby("qid", as_index=False)["abs_err"]
            .mean()
            .rename(columns={"abs_err": "value"})
        )

    if metric == "disagree_rate":
        base["is_disagree"] = (base["LLM"] != base["NIST"]).astype(float)
        return (
            base.groupby("qid", as_index=False)["is_disagree"]
            .mean()
            .rename(columns={"is_disagree": "value"})
        )

    raise ValueError("Unknown METRIC. Use 'mean_llm', 'mean_nist', 'mae_4pt', or 'disagree_rate'.")


# =========================
# Tukey helpers (formatting/output)
# =========================
def safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def tukey_to_df(tukey) -> pd.DataFrame:
    table = tukey.summary().data
    header = table[0]
    body = table[1:]
    df = pd.DataFrame(body, columns=header)

    for c in ["meandiff", "p-adj", "lower", "upper"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "reject" in df.columns:
        df["reject"] = df["reject"].astype(str).str.lower().map({"true": True, "false": False})

    return df


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    fmt = df.copy()
    for c in ["meandiff", "p-adj", "lower", "upper"]:
        if c in fmt.columns:
            fmt[c] = fmt[c].map(lambda x: f"{x:.6g}" if pd.notnull(x) else "")

    return fmt.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="l l r r r r l",
    )


# =========================
# Main
# =========================
def main() -> None:
    model_files = find_llm_files()
    print(f"Found {len(model_files)} models under: {LABEL_ROOT}")

    rows: List[pd.DataFrame] = []
    skipped = 0

    for model, files in model_files.items():
        for f in files:
            lang = get_lang_from_filename(f, model)

            if lang not in LANGS:
                continue

            try:
                df = load_labels(f)
                perq = per_qid_metric(df, METRIC)
                if perq.empty:
                    continue

                perq["model"] = model
                perq["lang"] = lang
                perq["group"] = perq["model"] + GROUP_SEP + perq["lang"]

                rows.append(perq[["group", "model", "lang", "qid", "value"]])

            except Exception as e:
                skipped += 1
                print(f"[SKIP] {model} {lang} ({f.name}): {e}")

    if not rows:
        raise RuntimeError(
            f"No samples produced. Check LABEL_ROOT={LABEL_ROOT}, LANGS={LANGS}, and file schemas."
        )

    long_df = pd.concat(rows, ignore_index=True)

    # Require >=2 query samples per group
    counts = long_df["group"].value_counts()
    keep_groups = counts[counts >= 2].index
    dropped = [g for g in counts.index if g not in set(keep_groups)]
    long_df = long_df[long_df["group"].isin(keep_groups)].copy()

    if long_df["group"].nunique() < 2:
        raise RuntimeError("Not enough (model,lang) groups with >=2 samples to run Tukey.")

    # Save samples for reproducibility
    write_df(long_df, OUT_SAMPLES)
    if dropped:
        print(f"[INFO] Dropped {len(dropped)} groups with <2 qid samples.")

    # =========================
    # Run Tukey HSD
    # =========================
    tukey = pairwise_tukeyhsd(
        endog=long_df["value"].to_numpy(),
        groups=long_df["group"].to_numpy(),
        alpha=ALPHA,
    )

    tukey_df = tukey_to_df(tukey)

    # Save CSV
    write_df(tukey_df, OUT_TUKEY_CSV)

    # Save LaTeX table
    latex = to_latex_table(
        tukey_df,
        caption=f"Tukey HSD across all (model,lang) groups for {METRIC}, FWER={ALPHA}.",
        label=f"tab:tukey_all_models_all_langs_{safe_slug(METRIC)}",
    )
    OUT_TUKEY_TEX.write_text(latex, encoding="utf-8")

    # =========================
    # Plot simultaneous confidence intervals
    # =========================
    fig, ax = plt.subplots(figsize=(10, 8))
    tukey.plot_simultaneous(ax=ax)

    level_palette = color_tukey_by_taxonomy(
        fig,
        ax,
        taxonomy_csv=TAXONOMY_CSV,
        group_sep=GROUP_SEP,
        default_level=0,
        linewidth=2.5,
    )
    taxonomy_legend(ax, level_to_rgba=level_palette, title="Taxonomy level", loc="upper left")
    center_x_axis_at_zero(ax)

    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(None)

    plt.tight_layout()
    plt.savefig(OUT_SIMUL_SVG, format="svg")
    plt.close(fig)

    # =========================
    # Final logging
    # =========================
    print(f"\n[OK] Wrote samples: {OUT_SAMPLES}")
    print(f"[OK] Wrote Tukey CSV: {OUT_TUKEY_CSV}")
    print(f"[OK] Wrote Tukey TeX: {OUT_TUKEY_TEX}")
    print(f"[OK] Wrote plot: {OUT_SIMUL_SVG}")
    if skipped:
        print(f"[INFO] Skipped {skipped} files due to errors (see logs above).")


if __name__ == "__main__":
    main()
