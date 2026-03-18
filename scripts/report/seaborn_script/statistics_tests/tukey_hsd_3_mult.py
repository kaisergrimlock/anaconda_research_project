#!/usr/bin/env python3
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple

# =========================
# Bootstrap: allow importing sibling tukey_hsd_3.py
# =========================
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# =========================
# Repo root bootstrap
# =========================
PROJECT_ROOT = THIS_FILE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEABORN_ROOT = THIS_FILE.parents[1]
if str(SEABORN_ROOT) not in sys.path:
    sys.path.insert(0, str(SEABORN_ROOT))

# =========================
# Imports
# =========================
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import tukey_hsd_3 as base

from settings import apply_paper_fmt
from helpers.draw import center_x_axis_at_zero
from helpers.lang_profiles import get_langs
from helpers.output_writer import write_df

# ========================
# Parameters
# ========================
ALPHA = 0.05
LANG_PROFILE = "mult"
LANGS: List[str] = get_langs(LANG_PROFILE)
METRIC = "mean_diff"

VARIANT_TO_MARKER = {
    "base": "^",
    "mult_2": "s",
    "mult_3": "D",
}

# =========================
# Config
# =========================
TREC_DL_YEAR = "2021"
LABEL_ROOT = Path("outputs/llm_label") / f"trec_dl_{TREC_DL_YEAR}"

OUT_DIR = Path("figures") / TREC_DL_YEAR / "tukey_hsd" / f"lang_group_{LANG_PROFILE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TUKEY_CSV = OUT_DIR / "tukey_hsd_table_lang_group.csv"
OUT_TUKEY_TEX = OUT_DIR / "tukey_hsd_table_lang_group.tex"
OUT_SIMUL_SVG = OUT_DIR / f"tukey_hsd_plot_simultaneous_all_groups_{TREC_DL_YEAR}.svg"
OUT_SIMUL_PDF = OUT_DIR / f"tukey_hsd_plot_simultaneous_all_groups_{TREC_DL_YEAR}.pdf"
OUT_SAMPLES = OUT_DIR / "tukey_samples_long_lang_group.csv"

GROUP_SEP = "|"
INVALID_CSV = Path(__file__).resolve().parent / f"invalid_{TREC_DL_YEAR}.csv"

KEY_COLS = ["qid", "pid"]
LABELS = [0, 1, 2, 3]

# =========================
# Language mapping
# =========================
LANG_TO_GROUP = {
    "raw": "eng_raw",
    "eng": "eng",
    "eng_mult_2": "eng_mult_2",
    "eng_mult_3": "eng_mult_3",
}


def get_lang_group(lang: str) -> Optional[str]:
    return LANG_TO_GROUP.get(lang)


# =========================
# Helpers
# =========================
def safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def split_group(group: str, group_sep: str = GROUP_SEP) -> Tuple[str, str]:
    if group_sep in group:
        return tuple(group.split(group_sep, 1))  # type: ignore[return-value]
    return "", group


def get_variant(lang_group: str) -> str:
    s = str(lang_group).strip().lower()
    if s.endswith("_raw") or s == "raw":
        return "raw"
    if s.endswith("_mult_3"):
        return "mult_3"
    if s.endswith("_mult_2"):
        return "mult_2"
    if s.endswith("_mult"):
        return "mult"
    return "base"

def build_group_metadata(groups: List[str], *, group_sep: str) -> pd.DataFrame:
    rows = []
    for group in groups:
        model, lang_group = split_group(group, group_sep=group_sep)
        variant = get_variant(lang_group)

        rows.append(
            {
                "group": group,
                "model": model,
                "lang_group": lang_group,
                "variant": variant,
                "is_raw": variant == "raw",
                "marker": VARIANT_TO_MARKER.get(variant, "o"),
            }
        )

    meta = pd.DataFrame(rows).drop_duplicates()

    variant_order = {
        "raw": 0,
        "base": 1,
        "mult": 2,
        "mult_2": 3,
        "mult_3": 4,
    }
    meta["variant_order"] = meta["variant"].map(variant_order).fillna(999).astype(int)

    meta = meta.sort_values(
        by=["model", "variant_order", "lang_group"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    # visible rows exclude RAW
    visible = meta[~meta["is_raw"]].copy()

    model_gap = 2.0
    row_step = 1.0

    ys = []
    y = 0.0
    prev_model = None

    for _, row in visible.iterrows():
        model = row["model"]
        if prev_model is not None and model != prev_model:
            y += model_gap
        ys.append(y)
        y += row_step
        prev_model = model

    visible["y"] = ys

    meta = meta.merge(
        visible[["group", "y"]],
        on="group",
        how="left",
    )

    return meta

def add_combined_variant_legend(ax) -> None:
    handles = [
        Line2D(
            [0], [0],
            marker="^",
            linestyle="None",
            color="black",
            markersize=6,
            markeredgewidth=1.0,
            label="Rand QP",
        ),
        
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            color="black",
            markersize=7,
            markeredgewidth=1.0,
            label="2 x Rand QP",
        ),
        
        Line2D(
            [0], [0],
            marker="D",
            linestyle="None",
            color="black",
            markersize=7,
            markeredgewidth=1.0,
            label="3 x Rand QP",
        ),

        Patch(
            facecolor="#7ec8e3",
            edgecolor="none",
            alpha=0.22,
            label="Baseline",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        fontsize="small",
        handletextpad=0.6,
        columnspacing=1.6,
        borderaxespad=0.0,
    )


def plot_simultaneous_lang_group(
    tukey,
    ax,
    *,
    group_sep: str = GROUP_SEP,
    model_x: float = -0.08,
    raw_band_alpha: float = 0.22,
    raw_band_color: str = "#7ec8e3",
) -> None:
    """
    Plot Tukey simultaneous CIs so that:
      - eng / eng_mult / eng_mult_2 / eng_mult_3 each keep separate rows
      - raw is not shown as a row or datapoint
      - raw is shown as a vertical shaded band spanning the full model block
      - y-axis labels are hidden
    """
    groups_unique = list(tukey.groupsunique)
    meta = build_group_metadata(groups_unique, group_sep=group_sep)

    means = tukey._multicomp.groupstats.groupmean
    tukey._simultaneous_ci()
    halfwidths = tukey.halfwidths

    group_to_mean = dict(zip(groups_unique, means))
    group_to_halfwidth = dict(zip(groups_unique, halfwidths))

    visible_meta = meta[~meta["is_raw"]].copy()

    group_to_y = dict(zip(visible_meta["group"], visible_meta["y"]))
    group_to_marker = dict(zip(visible_meta["group"], visible_meta["marker"]))

    row_df = visible_meta[["model", "group", "y"]].drop_duplicates().sort_values("y")

    model_blocks = []
    for model, sub in visible_meta.groupby("model", sort=False):
        ys = sub["y"].dropna().tolist()
        if ys:
            model_blocks.append((model, min(ys), max(ys)))

    model_to_block = {model: (y0, y1) for model, y0, y1 in model_blocks}

    # Draw RAW band first
    for group in groups_unique:
        model, lang_group = split_group(group, group_sep=group_sep)

        if get_variant(lang_group) != "raw":
            continue
        if model not in model_to_block:
            continue

        mean = group_to_mean[group]
        halfwidth = group_to_halfwidth[group]
        left = mean - halfwidth
        right = mean + halfwidth

        y0, y1 = model_to_block[model]
        ymin = y0 - 1.0
        ymax = y1 + 1.0

        ax.fill_betweenx(
            [ymin, ymax],
            left,
            right,
            color=raw_band_color,
            alpha=raw_band_alpha,
            linewidth=0,
            zorder=0,
        )

    # Draw visible non-raw rows
    for group in visible_meta["group"].tolist():
        y = group_to_y[group]
        mean = group_to_mean[group]
        halfwidth = group_to_halfwidth[group]
        marker = group_to_marker[group]

        left = mean - halfwidth
        right = mean + halfwidth

        ax.hlines(y, left, right, color="black", linewidth=1.4, zorder=2)
        ax.plot(
            mean,
            y,
            marker=marker,
            linestyle="None",
            color="black",
            markersize=9.5,
            markeredgewidth=2.0 if marker in {"x", "+", "X"} else 1.2,
            zorder=3,
        )

    # keep row positions, but hide y labels
    ax.set_yticks(row_df["y"].tolist())
    ax.set_yticklabels([""] * len(row_df))

    for i in range(len(model_blocks) - 1):
        _, _, y1 = model_blocks[i]
        _, y2, _ = model_blocks[i + 1]
        sep_y = (y1 + y2) / 2.0
        ax.axhline(y=sep_y, linewidth=1.2, alpha=0.8, color="black")

    trans = ax.get_yaxis_transform()
    for model, y0, y1 in model_blocks:
        ymid = (y0 + y1) / 2.0
        ax.text(
            model_x,
            ymid,
            model,
            transform=trans,
            rotation=90,
            va="center",
            ha="right",
            fontweight="bold",
        )

    ax.invert_yaxis()

# =========================
# Main
# =========================
def main() -> None:
    base.TREC_DL_YEAR = TREC_DL_YEAR
    base.LABEL_ROOT = LABEL_ROOT
    base.LABELS = LABELS
    base.KEY_COLS = KEY_COLS

    model_files = base.find_llm_files()
    print(f"Found {len(model_files)} models under: {LABEL_ROOT}")

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
            if lang not in LANGS:
                continue

            lang_group = get_lang_group(lang)
            if lang_group is None:
                continue

            try:
                if "invalid_keys" in base.load_labels.__code__.co_varnames:
                    df = base.load_labels(f, invalid_keys)
                else:
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

    long_df = (
        long_df.groupby(["model", "lang_group", "qid", "pid"], as_index=False)["value"]
        .mean()
    )

    long_df["group"] = long_df["model"] + GROUP_SEP + long_df["lang_group"]
    long_df = long_df[["group", "model", "lang_group", "qid", "pid", "value"]].copy()

    counts = long_df["group"].value_counts()
    keep_groups = counts[counts >= 2].index
    dropped = [g for g in counts.index if g not in set(keep_groups)]
    long_df = long_df[long_df["group"].isin(keep_groups)].copy()

    if long_df["group"].nunique() < 2:
        raise RuntimeError("Not enough (model,lang_group) groups with >=2 samples to run Tukey.")

    if dropped:
        print(f"[INFO] Dropped {len(dropped)} groups with <2 samples.")

    write_df(long_df, OUT_SAMPLES)

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
        label=f"tab:tukey_langgroups_{safe_slug(METRIC)}",
    )
    OUT_TUKEY_TEX.write_text(latex, encoding="utf-8")

    apply_paper_fmt()
    fig, ax = plt.subplots(figsize=(8, 8))

    plot_simultaneous_lang_group(
        tukey,
        ax,
        group_sep=GROUP_SEP,
        model_x=-0.08,
    )

    add_combined_variant_legend(ax)

    center_x_axis_at_zero(ax)
    ax.set_xlim(-0.1, 1.5)
    ax.tick_params(axis="y", length=0, pad=10)

    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
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