#!/usr/bin/env python3
"""
1-way ANOVA: mean_diff (LLM - NIST) ~ language

Pipeline mirrors your tukey_hsd script:
  find files -> load labels -> per-row metric -> long_df -> ANOVA -> save tables
Optional:
  - exclude (qid,pid) in invalid_<YEAR>.csv (global drop)
  - aggregate across models (recommended) to avoid model confound
  - Tukey post-hoc on languages
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# =========================
# Repo root bootstrap (MUST be before repo imports)
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]  # adjust if needed
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEABORN_ROOT = THIS_FILE.parents[1]  # seaborn_script/
if str(SEABORN_ROOT) not in sys.path:
    sys.path.insert(0, str(SEABORN_ROOT))

# =========================
# Now safe to import repo modules
# =========================
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from helpers.lang_profiles import get_langs
from scripts.csv_helpers import bump_field_limit
from helpers.output_writer import write_df

# ========================
# Parameters
# ========================
ALPHA = 0.05
LABELS = [0, 1, 2, 3]

LANG_PROFILE = "script"  # change profiles in lang_profiles.py
LANGS: List[str] = get_langs(LANG_PROFILE)

METRIC = "mean_diff"  # for this script we expect mean_diff

# If you want to restrict to a subset of models, set list; else None means all.
MODEL_FILTER: Optional[List[str]] = None  # e.g. ["gpt-oss-20b"]

# IMPORTANT:
# If True, we average values across models per (lang,qid,pid) BEFORE ANOVA.
# This avoids mixing model effects into a "language" test.
AGGREGATE_ACROSS_MODELS = True

# If True, also run Tukey post-hoc on languages.
RUN_TUKEY_POSTHOC = True

# =========================
# Config
# =========================
TREC_DL_YEAR = "2022"
LABEL_ROOT = Path("outputs/llm_label") / f"trec_dl_{TREC_DL_YEAR}"

OUT_DIR = Path("figures") / TREC_DL_YEAR / "anova" / f"lang_only_{LANG_PROFILE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SAMPLES = OUT_DIR / "anova_samples_long.csv"
OUT_ANOVA_CSV = OUT_DIR / "anova_1way_lang_mean_diff.csv"
OUT_ANOVA_TEX = OUT_DIR / "anova_1way_lang_mean_diff.tex"

OUT_TUKEY_CSV = OUT_DIR / "tukey_posthoc_lang_mean_diff.csv"
OUT_TUKEY_TEX = OUT_DIR / "tukey_posthoc_lang_mean_diff.tex"

# optional invalid file (global drop)
INVALID_CSV = Path(__file__).resolve().parent / f"invalid_{TREC_DL_YEAR}.csv"
KEY_COLS = ["qid", "pid"]


# =========================
# Helpers (copied style from your tukey script)
# =========================
def find_llm_files() -> Dict[str, List[Path]]:
    """
    Returns model -> list of label CSVs.
    Expected layout:
      outputs/llm_label/trec_dl_2022/<MODEL>/<MODEL>_trecdl_2022_<LANG>_labels.csv
    """
    if not LABEL_ROOT.exists():
        raise FileNotFoundError(f"LABEL_ROOT not found: {LABEL_ROOT}")

    model_files: Dict[str, List[Path]] = {}
    for model_dir in LABEL_ROOT.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        if MODEL_FILTER is not None and model_name not in set(MODEL_FILTER):
            continue

        csv_files = list(model_dir.glob(f"{model_name}_trecdl_{TREC_DL_YEAR}_*_labels.csv"))
        if csv_files:
            model_files[model_name] = csv_files
        else:
            print(f"[WARN] No label CSVs found for model {model_name} in {model_dir}")
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


def load_invalid_keys(path: Path) -> Set[Tuple[int, str]]:
    """
    Loads invalid (qid,pid) pairs from invalid_YYYY.csv (columns include: model,lang,qid,pid,reason).
    We drop them globally across all models/langs.
    """
    if not path.exists():
        print(f"[INFO] No invalid file found: {path}")
        return set()

    inv = pd.read_csv(path)
    if not set(KEY_COLS).issubset(inv.columns):
        raise ValueError(f"{path} must contain columns {KEY_COLS}")

    inv = inv.dropna(subset=KEY_COLS).copy()
    inv["qid"] = pd.to_numeric(inv["qid"], errors="coerce")
    inv = inv.dropna(subset=["qid"]).copy()
    inv["qid"] = inv["qid"].astype(int)
    inv["pid"] = inv["pid"].astype(str)

    keys = set(inv[KEY_COLS].drop_duplicates().itertuples(index=False, name=None))
    print(f"[INFO] Loaded {len(keys)} invalid keys from {path}")
    return keys


def load_labels(file_path: Path, invalid_keys: Set[Tuple[int, str]]) -> pd.DataFrame:
    """
    Read labels, enforce schema + label validity, optionally drop invalid (qid,pid).
    Produces columns: qid, pid, NIST, LLM, ...
    """
    bump_field_limit()
    df = pd.read_csv(file_path)

    requisite = {"qid", "pid", "relevance", "llm_relevance"}
    missing = requisite - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {file_path}")

    df["NIST"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["LLM"] = pd.to_numeric(df["llm_relevance"], errors="coerce")

    # Keep only rows with valid NIST and LLM labels
    df = df.dropna(subset=["NIST", "LLM"]).copy()
    df["NIST"] = df["NIST"].astype(int)
    df["LLM"] = df["LLM"].astype(int)
    df = df[df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)].copy()

    # Normalize key types
    df = df.dropna(subset=["qid", "pid"]).copy()
    df["qid"] = pd.to_numeric(df["qid"], errors="coerce")
    df = df.dropna(subset=["qid"]).copy()
    df["qid"] = df["qid"].astype(int)
    df["pid"] = df["pid"].astype(str)

    # Drop invalid (qid,pid) keys globally
    if invalid_keys:
        keys = pd.Index(list(zip(df["qid"].to_numpy(), df["pid"].to_numpy())))
        df = df[~keys.isin(invalid_keys)].copy()

    return df


def per_row_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Turn a (qid,pid)-level label dataframe into per-row samples:
      qid, pid, value
    """
    base = df.drop_duplicates(subset=["qid", "pid"]).copy()

    if metric == "mean_diff":
        base["value"] = base["LLM"] - base["NIST"]
        return base[["qid", "pid", "value"]]

    if metric == "mae_4pt":
        base["value"] = (base["LLM"] - base["NIST"]).abs()
        return base[["qid", "pid", "value"]]

    if metric == "disagree_rate":
        base["value"] = (base["LLM"] != base["NIST"]).astype(float)
        return base[["qid", "pid", "value"]]

    raise ValueError("Unknown METRIC. Use 'mean_diff', 'mae_4pt', or 'disagree_rate'.")


def safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def df_to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Simple LaTeX render with controlled float formatting for common columns.
    """
    fmt = df.copy()
    for c in fmt.columns:
        if pd.api.types.is_numeric_dtype(fmt[c]):
            fmt[c] = fmt[c].map(lambda x: f"{x:.6g}" if pd.notnull(x) else "")
    return fmt.to_latex(index=False, escape=False, caption=caption, label=label)


# =========================
# Main
# =========================
def main() -> None:
    model_files = find_llm_files()
    print(f"Found {len(model_files)} models under: {LABEL_ROOT}")

    invalid_keys = load_invalid_keys(INVALID_CSV)

    rows: List[pd.DataFrame] = []
    skipped = 0

    for model, files in model_files.items():
        for f in files:
            lang = get_lang_from_filename(f, model)

            if lang is None:
                continue
            if lang not in LANGS:
                continue

            try:
                df = load_labels(f, invalid_keys)

                pair_count = (
                    df.dropna(subset=KEY_COLS)
                    .drop_duplicates(subset=KEY_COLS)
                    .shape[0]
                )
                print(f"[INFO] {model} {lang}: {pair_count} valid (qid,pid) pairs")

                perrow = per_row_metric(df, METRIC)
                if perrow.empty:
                    continue

                perrow["model"] = model
                perrow["lang"] = lang
                rows.append(perrow[["value", "model", "lang", "qid", "pid"]])

            except Exception as e:
                skipped += 1
                print(f"[SKIP] {model} {lang} ({f.name}): {e}")

    if not rows:
        raise RuntimeError(
            f"No samples produced. Check LABEL_ROOT={LABEL_ROOT}, LANGS={LANGS}, and file schemas."
        )

    long_df = pd.concat(rows, ignore_index=True)

    # Optional: avoid model confound by averaging across models at (lang,qid,pid)
    if AGGREGATE_ACROSS_MODELS:
        long_df = (
            long_df.groupby(["lang", "qid", "pid"], as_index=False)["value"]
            .mean()
        )
    # else: ANOVA sees each model’s row as a separate sample (language-only factor)

    # Require >=2 samples per language
    counts = long_df["lang"].value_counts()
    keep_langs = counts[counts >= 2].index
    dropped_langs = [l for l in counts.index if l not in set(keep_langs)]
    long_df = long_df[long_df["lang"].isin(keep_langs)].copy()

    if long_df["lang"].nunique() < 2:
        raise RuntimeError("Not enough languages with >=2 samples to run 1-way ANOVA.")

    if dropped_langs:
        print(f"[INFO] Dropped {len(dropped_langs)} langs with <2 samples: {dropped_langs}")

    # Save samples for reproducibility
    write_df(long_df, OUT_SAMPLES)

    # =========================
    # 1-way ANOVA: value ~ C(lang)
    # =========================
    # Note: 'C(lang)' treats language as categorical factor
    fit = ols("value ~ C(lang)", data=long_df).fit()
    anova_tbl = anova_lm(fit, typ=2).reset_index().rename(columns={"index": "term"})

    write_df(anova_tbl, OUT_ANOVA_CSV)
    OUT_ANOVA_TEX.write_text(
        df_to_latex_table(
            anova_tbl,
            caption=f"One-way ANOVA for {METRIC}: $value \\sim \\mathrm{{lang}}$ (alpha={ALPHA}).",
            label=f"tab:anova_1way_lang_{safe_slug(METRIC)}_{TREC_DL_YEAR}",
        ),
        encoding="utf-8",
    )

    print("\n[ANOVA] value ~ C(lang)")
    print(anova_tbl)

    # =========================
    # Optional: Tukey post-hoc on languages
    # =========================
    if RUN_TUKEY_POSTHOC:
        tukey = pairwise_tukeyhsd(
            endog=long_df["value"].to_numpy(),
            groups=long_df["lang"].to_numpy(),
            alpha=ALPHA,
        )

        # Convert to dataframe similarly to your tukey helper
        table = tukey.summary().data
        header = table[0]
        body = table[1:]
        tukey_df = pd.DataFrame(body, columns=header)

        for c in ["meandiff", "p-adj", "lower", "upper"]:
            if c in tukey_df.columns:
                tukey_df[c] = pd.to_numeric(tukey_df[c], errors="coerce")

        if "reject" in tukey_df.columns:
            tukey_df["reject"] = (
                tukey_df["reject"].astype(str).str.lower().map({"true": True, "false": False})
            )

        write_df(tukey_df, OUT_TUKEY_CSV)
        OUT_TUKEY_TEX.write_text(
            df_to_latex_table(
                tukey_df,
                caption=f"Tukey HSD post-hoc across languages for {METRIC} (alpha={ALPHA}).",
                label=f"tab:tukey_posthoc_lang_{safe_slug(METRIC)}_{TREC_DL_YEAR}",
            ),
            encoding="utf-8",
        )
        print(f"\n[OK] Wrote Tukey post-hoc: {OUT_TUKEY_CSV}")

    # =========================
    # Final logging
    # =========================
    print(f"\n[OK] Wrote samples: {OUT_SAMPLES}")
    print(f"[OK] Wrote ANOVA CSV: {OUT_ANOVA_CSV}")
    print(f"[OK] Wrote ANOVA TeX: {OUT_ANOVA_TEX}")
    if skipped:
        print(f"[INFO] Skipped {skipped} files due to errors (see logs above).")


if __name__ == "__main__":
    main()