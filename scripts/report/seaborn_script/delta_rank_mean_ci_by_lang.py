#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D


# ===============================================================
# Repo root discovery (robust)
# ===============================================================
THIS_FILE = Path(__file__).resolve()

def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "scripts").is_dir() and ((p / "outputs").is_dir() or (p / "retrieved").is_dir()):
            return p
    return start.parents[4]

PROJECT_ROOT = find_repo_root(THIS_FILE.parent)

IN_DIR = PROJECT_ROOT / "outputs" / "anserini_rank_deltas"
FIG_DIR = PROJECT_ROOT / "figures"

# Adjust if your lang.csv lives elsewhere
LANG_CSV = PROJECT_ROOT / "scripts" / "report" / "seaborn_script" / "lang.csv"


def detect_year_from_name(name: str) -> Optional[str]:
    m = re.search(r"(20\d{2})", name)
    return m.group(1) if m else None


# ===============================================================
# lang.csv loading: language -> level, and level -> color
# ===============================================================
def _pick_first_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for want in candidates:
        if want.lower() in cols:
            return cols[want.lower()]
    return None


def load_lang_levels(
    lang_csv: Path,
    *,
    default_level: Optional[int] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Returns:
      - lang_to_level: {"eng": 5, ...}
      - level_to_color: {2:"#...", 3:"#...", ...} using tab10 (includes default_level if provided)
    """
    if not lang_csv.exists():
        raise FileNotFoundError(f"lang.csv not found at: {lang_csv}")

    df = pd.read_csv(lang_csv)

    lang_col = _pick_first_col(
        df, ("lang", "language", "code", "lang_code", "iso", "iso639", "iso_639_1", "iso_639_3")
    )
    level_col = _pick_first_col(
        df, ("taxonomy", "class", "resource_class", "joshi_class", "tax_class", "level", "resource_level")
    )

    if lang_col is None or level_col is None:
        raise ValueError(
            f"Could not infer language+level columns from {lang_csv.name}. Found columns: {list(df.columns)}"
        )

    df[lang_col] = df[lang_col].astype(str).str.strip().str.lower().replace({"en": "eng"})
    df[level_col] = pd.to_numeric(df[level_col], errors="coerce")

    levels = sorted({int(v) for v in df[level_col].dropna().astype(int)})
    if default_level is not None and default_level not in levels:
        levels = sorted([default_level, *levels])
    cmap = plt.get_cmap("tab10")
    level_to_color = {lvl: to_hex(cmap(i % cmap.N)) for i, lvl in enumerate(levels)}

    lang_to_level: Dict[str, int] = {}
    for _, row in df.iterrows():
        lang = str(row[lang_col]).strip().lower()
        if not lang or pd.isna(row[level_col]):
            continue
        lang_to_level[lang] = int(row[level_col])

    return lang_to_level, level_to_color


# ===============================================================
# Plot: RAW boxplot per language, colored by resource level
# ===============================================================
def plot_raw_box_by_lang(
    df_raw: pd.DataFrame,
    title: str,
    out_pdf: Path,
    lang_to_level: Dict[str, int],
    level_to_color: Dict[int, str],
    *,
    default_level: Optional[int] = None,
) -> None:
    sns.set_theme(style="whitegrid")

    dfp = df_raw.copy()
    dfp["language"] = dfp["language"].astype(str).str.strip().str.lower().replace({"en": "eng"})
    dfp["delta_rank"] = pd.to_numeric(dfp["delta_rank"], errors="coerce")
    dfp = dfp.dropna(subset=["language", "delta_rank"])

    # attach resource level for coloring
    dfp["resource_level"] = dfp["language"].map(lang_to_level)

    # If a language is missing from lang.csv, fall back to default_level (if provided).
    if default_level is not None:
        dfp["resource_level"] = dfp["resource_level"].fillna(default_level)
    dfp["resource_level"] = dfp["resource_level"].fillna(-1).astype(int)

    if dfp.empty:
        raise ValueError("No rows left after cleaning delta_rank/language.")

    # order languages by median delta
    order = (
        dfp.groupby("language")["delta_rank"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    # palette keyed by resource level (hue)
    uniq_levels = sorted(dfp["resource_level"].unique())
    palette = {}
    for lvl in uniq_levels:
        if lvl == -1:
            palette[lvl] = "#9e9e9e"  # unknown
        else:
            palette[lvl] = level_to_color.get(lvl, "#9e9e9e")

    fig_h = max(4, 0.45 * len(order))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    meanprops = {
        "marker": "D",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": 5,
    }

    sns.boxplot(
        data=dfp,
        y="language",
        x="delta_rank",
        order=order,
        hue="resource_level",
        palette=palette,
        dodge=False,        # IMPORTANT: one box per language
        whis=1.5,           # Tukey whiskers
        showfliers=False,   # hide outlier dots
        showmeans=True,
        meanprops=meanprops,
        linewidth=1,
        ax=ax,
    )

    ax.axvline(0, linewidth=1)
    ax.set_ylabel("Language")
    ax.set_xlabel("")  # remove axis label

    # Legend: convert "-1" to "Unknown", 0 to "Baseline", others to "Level X", plus Mean
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lab in labels:
        try:
            lvl = int(lab)
        except ValueError:
            new_labels.append(lab)
            continue
        if lvl == -1:
            new_labels.append("Unknown")
        elif lvl == 0:
            new_labels.append("Baseline")
        else:
            new_labels.append(f"Level {lvl}")

    mean_handle = Line2D(
        [0], [0],
        marker=meanprops["marker"],
        color="black",
        markerfacecolor=meanprops["markerfacecolor"],
        markeredgecolor=meanprops["markeredgecolor"],
        markersize=meanprops["markersize"],
        linestyle="None",
        label="Mean",
    )

    ax.legend(
        handles=handles + [mean_handle],
        labels=new_labels + ["Mean"],
        title="Resource Level",
        loc="upper right",
        framealpha=0.9,
        edgecolor="black",
        fontsize="small",
        title_fontsize="medium",
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


# ===============================================================
# Main
# ===============================================================
def main() -> None:
    print(f"[INFO] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[INFO] IN_DIR:       {IN_DIR}")
    print(f"[INFO] LANG_CSV:     {LANG_CSV}")

    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {IN_DIR}")

    # RAW deltas only
    in_files = sorted(IN_DIR.glob("*.delta_rank.csv"))
    if not in_files:
        raise FileNotFoundError(
            f"No *.delta_rank.csv files found in {IN_DIR}.\n"
            f"(You are probably only generating *.delta_rank_mean.csv right now.)"
        )

    lang_to_level, level_to_color = load_lang_levels(LANG_CSV, default_level=0)

    print(f"[INFO] Found {len(in_files)} raw delta file(s).")
    print(f"[INFO] Loaded {len(lang_to_level)} language level(s) from lang.csv.")

    for in_csv in in_files:
        df = pd.read_csv(in_csv)

        required = {"language", "delta_rank"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{in_csv.name} missing columns: {sorted(missing)}. Found: {list(df.columns)}")

        year = detect_year_from_name(in_csv.name) or "misc"
        out_dir = FIG_DIR / year
        out_pdf = out_dir / f"{in_csv.stem}.raw_box_by_lang_whis1p5.langcsv.pdf"

        title = f"ΔRank by Language (raw; boxplot, whis=1.5) — {in_csv.name}"
        plot_raw_box_by_lang(
            df_raw=df,
            title=title,
            out_pdf=out_pdf,
            lang_to_level=lang_to_level,
            level_to_color=level_to_color,
            default_level=0,
        )

        print(f"[DONE] {in_csv.name}")
        print(f"       plot: {out_pdf.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
