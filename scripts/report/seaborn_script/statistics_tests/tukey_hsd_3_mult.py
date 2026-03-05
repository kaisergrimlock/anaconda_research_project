#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Dict, List, Optional

# =========================
# Bootstrap: allow importing sibling tukey_hsd_3.py
# =========================
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# =========================
# Repo root bootstrap (needed for helpers/, scripts/, etc.)
# Keep same style as your existing scripts.
# =========================
PROJECT_ROOT = THIS_FILE.parents[4]  # adjust if needed
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEABORN_ROOT = THIS_FILE.parents[1]  # seaborn_script/
if str(SEABORN_ROOT) not in sys.path:
    sys.path.insert(0, str(SEABORN_ROOT))

# =========================
# Imports
# =========================
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import tukey_hsd_3 as base  # <-- reuse functions from tukey_hsd_3.py (NO CHANGES to it)

from helpers.draw import (
    color_tukey_by_taxonomy,
    center_x_axis_at_zero,
    taxonomy_legend,
    # NOTE: we intentionally do NOT call add_model_separators here unless you want it.
)
from helpers.lang_profiles import get_langs
from helpers.output_writer import write_df

# ========================
# Parameters (lang-group specific)
# ========================
ALPHA = 0.05
LANG_PROFILE = "mult"
LANGS: List[str] = get_langs(LANG_PROFILE)
METRIC = "mean_diff"

# =========================
# Config
# =========================
TREC_DL_YEAR = "2022"
LABEL_ROOT = Path("outputs/llm_label") / f"trec_dl_{TREC_DL_YEAR}"

OUT_DIR = Path("figures") / TREC_DL_YEAR / "tukey_hsd" / f"lang_group_{LANG_PROFILE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TUKEY_CSV = OUT_DIR / "tukey_hsd_table_lang_group.csv"
OUT_TUKEY_TEX = OUT_DIR / "tukey_hsd_table_lang_group.tex"
OUT_SIMUL_SVG = OUT_DIR / f"tukey_hsd_plot_simultaneous_lang_group_{TREC_DL_YEAR}.svg"
OUT_SIMUL_PDF = OUT_DIR / f"tukey_hsd_plot_simultaneous_lang_group_{TREC_DL_YEAR}.pdf"
OUT_SAMPLES = OUT_DIR / "tukey_samples_long_lang_group.csv"

GROUP_SEP = "|"
TAXONOMY_CSV = Path(__file__).resolve().parents[1] / "lang.csv"

# invalid csv lives next to this script (same pattern as tukey_hsd_3)
INVALID_CSV = Path(__file__).resolve().parent / f"invalid_{TREC_DL_YEAR}.csv"

# Match base expectations
KEY_COLS = ["qid", "pid"]
LABELS = [0, 1, 2, 3]

# =========================
# Language -> Script-group mapping
# =========================
LANG_TO_GROUP = {
    "eng": "eng_1",
    "eng_mult": "eng_2",
    "eng_mult_2": "eng_3",
    "eng_mult_3": "eng_4"
}

def get_lang_group(lang: str) -> Optional[str]:
    return LANG_TO_GROUP.get(lang)

# =========================
# Main
# =========================
def main() -> None:
    """
    Lang-group Tukey:
      - Load all (model, lang) label files like tukey_hsd_3
      - Map lang -> lang_group (Latin/Cyrillic/Hanzi)
      - Compute per-row metric (e.g., mean_diff) per (qid,pid)
      - Pool languages within same (model, lang_group, qid, pid) by mean()
      - Run Tukey over groups = model|lang_group
      - Reuse tukey_hsd_3 table + plot helpers
    """

    # ------------------------------------------------------------------
    # Make tukey_hsd_3's functions operate with THIS script's config
    # WITHOUT modifying tukey_hsd_3.py:
    # ------------------------------------------------------------------
    base.TREC_DL_YEAR = TREC_DL_YEAR
    base.LABEL_ROOT = LABEL_ROOT
    base.LABELS = LABELS
    base.KEY_COLS = KEY_COLS

    model_files = base.find_llm_files()
    print(f"Found {len(model_files)} models under: {LABEL_ROOT}")

    # invalid filtering (if your tukey_hsd_3.py contains load_invalid_keys + load_labels(invalid_keys))
    invalid_keys = set()
    if hasattr(base, "load_invalid_keys"):
        invalid_keys = base.load_invalid_keys(INVALID_CSV)
    else:
        print("[WARN] tukey_hsd_3 has no load_invalid_keys(); skipping invalid filtering.")

    rows: List[pd.DataFrame] = []
    skipped = 0

    for model, files in model_files.items():
        for f in files:
            lang = base.get_lang_from_filename(f, model)

            if lang is None:
                continue

            # strict language include list
            if lang not in LANGS:
                continue

            # map language into script-groups
            lang_group = get_lang_group(lang)
            if lang_group is None:
                continue  # skip langs not in mapping

            try:
                # reuse base loader (with invalid filtering if available)
                if "invalid_keys" in base.load_labels.__code__.co_varnames:
                    df = base.load_labels(f, invalid_keys)
                else:
                    # fallback (older base.load_labels signature)
                    df = base.load_labels(f)

                pair_count = (
                    df.dropna(subset=KEY_COLS)
                    .drop_duplicates(subset=KEY_COLS)
                    .shape[0]
                )
                print(f"[INFO] {model} {lang} -> {lang_group}: {pair_count} valid (qid,pid) pairs")

                perrow = base.per_row_metric(df, METRIC)
                if perrow.empty:
                    continue

                perrow["model"] = model
                perrow["lang"] = lang
                perrow["lang_group"] = lang_group

                rows.append(perrow[["model", "lang", "lang_group", "qid", "pid", "value"]])

            except Exception as e:
                skipped += 1
                print(f"[SKIP] {model} {lang} ({f.name}): {e}")

    if not rows:
        raise RuntimeError(
            f"No samples produced. Check LABEL_ROOT={LABEL_ROOT}, LANGS={LANGS}, and LANG_TO_GROUP mapping."
        )

    long_df = pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Pool languages within the same script-group by averaging per (model,qid,pid).
    # Example: (gpt-oss-20b, Latin) averages eng + fr if both exist for same (qid,pid).
    # ------------------------------------------------------------------
    long_df = (
        long_df.groupby(["model", "lang_group", "qid", "pid"], as_index=False)["value"]
        .mean()
    )

    long_df["group"] = long_df["model"] + GROUP_SEP + long_df["lang_group"]
    long_df = long_df[["group", "model", "lang_group", "qid", "pid", "value"]].copy()

    # Require >=2 row samples per group
    counts = long_df["group"].value_counts()
    keep_groups = counts[counts >= 2].index
    dropped = [g for g in counts.index if g not in set(keep_groups)]
    long_df = long_df[long_df["group"].isin(keep_groups)].copy()

    if long_df["group"].nunique() < 2:
        raise RuntimeError("Not enough (model,lang_group) groups with >=2 samples to run Tukey.")

    if dropped:
        print(f"[INFO] Dropped {len(dropped)} groups with <2 samples.")

    # Save pooled samples
    write_df(long_df, OUT_SAMPLES)

    # =========================
    # Tukey HSD
    # =========================
    tukey = pairwise_tukeyhsd(
        endog=long_df["value"].to_numpy(),
        groups=long_df["group"].to_numpy(),
        alpha=ALPHA,
    )

    tukey_df = base.tukey_to_df(tukey)

    write_df(tukey_df, OUT_TUKEY_CSV)

    latex = base.to_latex_table(
        tukey_df,
        caption=f"Tukey HSD across (model,lang_group) for {METRIC}, FWER={ALPHA}.",
        label=f"tab:tukey_langgroups_{base.safe_slug(METRIC)}",
    )
    OUT_TUKEY_TEX.write_text(latex, encoding="utf-8")

    # =========================
    # Plot simultaneous CIs (reuse base styling)
    # =========================
    fig, ax = plt.subplots(figsize=(8, 7))

    # base wrapper does: plot + ytick restyle + model block labels
    if hasattr(base, "plot_simultaneous_with_model_blocks"):
        base.plot_simultaneous_with_model_blocks(tukey, ax, group_sep=GROUP_SEP)
        plt.subplots_adjust(left=0.28)
    else:
        # fallback: plain plot
        tukey.plot_simultaneous(ax=ax)

    # Taxonomy coloring note:
    # color_tukey_by_taxonomy expects lang codes in lang.csv.
    # Here ticks are Latin/Cyrillic/Hanzi, so they likely won't match and will fall back to default_level.
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
    ax.set_xlim(-0.1, 1.75)

    plt.tight_layout()
    plt.savefig(OUT_SIMUL_SVG, format="svg")
    plt.savefig(OUT_SIMUL_PDF, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"\n[OK] Wrote samples: {OUT_SAMPLES}")
    print(f"[OK] Wrote Tukey CSV: {OUT_TUKEY_CSV}")
    print(f"[OK] Wrote Tukey TeX: {OUT_TUKEY_TEX}")
    print(f"[OK] Wrote plot: {OUT_SIMUL_PDF}")
    if skipped:
        print(f"[INFO] Skipped {skipped} files due to errors (see logs above).")


if __name__ == "__main__":
    main()