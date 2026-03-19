import sys
import re
from pathlib import Path
from typing import Dict, List, Optional

# =========================
# Repo root bootstrap (MUST be before repo imports)
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]  # adjust if needed
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

THIS_FILE = Path(__file__).resolve()
SEABORN_ROOT = THIS_FILE.parents[1]  # seaborn_script/

if str(SEABORN_ROOT) not in sys.path:
    sys.path.insert(0, str(SEABORN_ROOT))

# =========================
# Now safe to import repo modules
# =========================
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from helpers.draw import (
    color_tukey_by_taxonomy,
    center_x_axis_at_zero,
    taxonomy_legend,
    add_model_separators,
)
from helpers.lang_profiles import get_langs
from scripts.csv_helpers import bump_field_limit
from helpers.output_writer import write_df
# ========================
# Parameters
# ========================
ALPHA = 0.05
LABELS = [0, 1, 2, 3]
LANG_PROFILE = "instruct_defended"  # change profiles in lang_profiles.py
LANGS: List[str] = get_langs(LANG_PROFILE)
METRIC = "mean_diff"

# =========================
# Config
# =========================
TREC_DL_YEAR = "2021"
LABEL_ROOT = Path("outputs/llm_label") / f"trec_dl_{TREC_DL_YEAR}"
OUT_DIR = Path("figures") / TREC_DL_YEAR / "tukey_hsd" / f"all_models_all_{LANG_PROFILE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TUKEY_CSV = OUT_DIR / "tukey_hsd_table_all_groups.csv"
OUT_TUKEY_TEX = OUT_DIR / "tukey_hsd_table_all_groups.tex"
OUT_SIMUL_SVG = OUT_DIR / f"tukey_hsd_plot_simultaneous_all_groups_{TREC_DL_YEAR}.svg"
OUT_SIMUL_PDF = OUT_DIR / f"tukey_hsd_plot_simultaneous_all_groups_{TREC_DL_YEAR}.pdf"
OUT_SAMPLES   = OUT_DIR / "tukey_samples_long.csv"
INVALID_CSV = Path(__file__).resolve().parent / f"invalid_{TREC_DL_YEAR}.csv"
KEY_COLS = ["qid", "pid"]

GROUP_SEP = "|"
TAXONOMY_CSV = Path(__file__).resolve().parents[1] / "lang.csv"

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
        csv_files = list(model_dir.glob(f"{model_name}_trecdl_{TREC_DL_YEAR}_*_labels.csv"))
        if csv_files:
            model_files[model_name] = csv_files
        if not csv_files:
            print(f"Warning: no label CSVs found for model {model_name} in {model_dir}")
    return model_files

def get_lang_from_filename(file_path: Path, model: str) -> Optional[str]:
    """
    Extract <LANG> from:
      <MODEL>_trecdl_<YEAR>_<LANG>_labels.csv
    """
    fname = file_path.name  # <-- Path -> filename string
    pattern = rf"^{re.escape(model)}_trecdl_\d{{4}}_(.+?)_labels\.csv$"
    match = re.search(pattern, fname)
    return match.group(1) if match else None

def load_labels(file_path: Path, invalid_keys: set[tuple[int, str]]) -> pd.DataFrame:
    bump_field_limit()
    df = pd.read_csv(file_path)

    requisite = {"qid", "pid", "relevance", "llm_relevance"}
    missing = requisite - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {file_path}")

    # Parse labels
    df["NIST"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["LLM"] = pd.to_numeric(df["llm_relevance"], errors="coerce")

    # Keep only rows with valid labels
    df = df.dropna(subset=["NIST", "LLM"]).copy()
    df["NIST"] = df["NIST"].astype(int)
    df["LLM"] = df["LLM"].astype(int)
    df = df[df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)].copy()

    # Normalize keys
    df = df.dropna(subset=["qid", "pid"]).copy()
    df["qid"] = pd.to_numeric(df["qid"], errors="coerce")
    df = df.dropna(subset=["qid"]).copy()
    df["qid"] = df["qid"].astype(int)
    df["pid"] = df["pid"].astype(str)

    # Drop invalid keys
    if invalid_keys:
        keys = pd.Index(list(zip(df["qid"].to_numpy(), df["pid"].to_numpy())))
        df = df[~keys.isin(invalid_keys)].copy()

    return df
def per_row_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Turn a (qid,pid)-level label dataframe into per-row samples:
      qid, pid, value

    These 'value' samples are what Tukey HSD consumes.
    """
    # Keep only valid, unique (qid,pid) pairs so duplicates or missing IDs
    # don't distort per-row samples.
    base = df.dropna(subset=["qid", "pid"]).drop_duplicates(subset=["qid", "pid"]).copy()

    if metric == "mean_diff":
        # signed difference: positive means LLM > NIST
        base["value"] = base["LLM"] - base["NIST"]
        return base[["qid", "pid", "value"]]

    if metric == "mae_4pt":
        # absolute error per row
        base["value"] = (base["LLM"] - base["NIST"]).abs()
        return base[["qid", "pid", "value"]]

    if metric == "disagree_rate":
        # 1 if LLM != NIST for the row, else 0
        base["value"] = (base["LLM"] != base["NIST"]).astype(float)
        return base[["qid", "pid", "value"]]

    raise ValueError("Unknown METRIC. Use 'mean_diff', 'mae_4pt', or 'disagree_rate'.")

def load_invalid_keys(path: Path) -> set[tuple[int, str]]:
    """
    Loads invalid (qid,pid) pairs from invalid_YYYY.csv (columns: model,lang,qid,pid,reason).
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
    inv = inv.dropna(subset=["qid"])
    inv["qid"] = inv["qid"].astype(int)
    inv["pid"] = inv["pid"].astype(str)

    keys = set(inv[KEY_COLS].drop_duplicates().itertuples(index=False, name=None))
    print(f"[INFO] Loaded {len(keys)} invalid keys from {path}")
    return keys

# =========================
# Step 6: Tukey helpers (formatting/output)
# =========================
def safe_slug(s: str) -> str:
    """
    Convert a string to a filesystem/latex-friendly slug.
    Example: 'MAE 4pt' -> 'mae_4pt'
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def tukey_to_df(tukey) -> pd.DataFrame:
    """
    Convert statsmodels Tukey result into a DataFrame.

    Tukey summary columns typically include:
      group1, group2, meandiff, p-adj, lower, upper, reject
    """
    table = tukey.summary().data   # list-of-lists (header + rows)
    header = table[0]
    body = table[1:]
    df = pd.DataFrame(body, columns=header)

    # Convert numeric columns from strings -> floats
    for c in ["meandiff", "p-adj", "lower", "upper"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert reject column into real booleans if present
    if "reject" in df.columns:
        df["reject"] = (
            df["reject"].astype(str).str.lower().map({"true": True, "false": False})
        )

    return df


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Render a DataFrame as a LaTeX table with controlled numeric formatting.
    """
    fmt = df.copy()

    # short numeric formatting (you can change precision here)
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
    
def plot_simultaneous_with_model_blocks(
    tukey,
    ax,
    group_sep: str = "|",
    model_fontsize: int = 14,
    model_x: float = -0.14,
    sep_kwargs: dict | None = None,
):
    tukey.plot_simultaneous(ax=ax)

    # ---- draw separators ONLY between model blocks (before we change tick labels) ----
    if sep_kwargs is None:
        sep_kwargs = {"linewidth": 1.5, "alpha": 0.6}

    labels = [t.get_text() for t in ax.get_yticklabels()]
    y_ticks = ax.get_yticks()

    models = []
    for s in labels:
        models.append(s.split(group_sep, 1)[0] if group_sep in s else "")

    for i in range(len(models) - 1):
        if models[i] != models[i + 1]:
            y = (y_ticks[i] + y_ticks[i + 1]) / 2.0
            ax.axhline(y=y, **sep_kwargs)

    # ---- now proceed with your existing restyling ----
    parsed = []
    for s in labels:
        if group_sep in s:
            model, lang = s.split(group_sep, 1)
        else:
            model, lang = "", s
        parsed.append((model, lang))

    ax.set_yticklabels([lang for _, lang in parsed])

    # model block labels (unchanged)
    blocks = []
    cur_model = None
    start = 0
    for i, (m, _) in enumerate(parsed):
        if m != cur_model:
            if cur_model is not None:
                blocks.append((cur_model, start, i - 1))
            cur_model = m
            start = i
    if cur_model is not None:
        blocks.append((cur_model, start, len(parsed) - 1))

    trans = ax.get_yaxis_transform()
    for model, i0, i1 in blocks:
        if not model:
            continue
        ymid = (y_ticks[i0] + y_ticks[i1]) / 2.0
        ax.text(
            model_x, ymid, model,
            transform=trans,
            rotation=90,
            va="center",
            ha="right",
            fontsize=model_fontsize,
            fontweight="bold",
        )
# =========================
# Step 7: Main (data assembly)
# =========================
def main() -> None:
    model_files = find_llm_files()
    invalid_keys = load_invalid_keys(INVALID_CSV)
    print(f"Found {len(model_files)} models under: {LABEL_ROOT}")

    rows: List[pd.DataFrame] = []
    skipped = 0

    for model, files in model_files.items():
        for f in files:
            lang = get_lang_from_filename(f, model)

            # strict language include list
            if lang not in LANGS:
                continue
            try:
                df = load_labels(f, invalid_keys)
                pair_count = (
                    df.dropna(subset=["qid", "pid"])
                    .drop_duplicates(subset=["qid", "pid"])
                    .shape[0]
                )
                print(f"[INFO] {model} {lang}: {pair_count} valid (qid,pid) pairs")
                perrow = per_row_metric(df, METRIC)
                if perrow.empty:
                    continue

                perrow["model"] = model
                perrow["lang"] = lang
                perrow["group"] = perrow["model"] + GROUP_SEP + perrow["lang"]

                rows.append(perrow[["group", "model", "lang", "qid", "pid", "value"]])

            except Exception as e:
                skipped += 1
                print(f"[SKIP] {model} {lang} ({f.name}): {e}")

    if not rows:
        raise RuntimeError(
            f"No samples produced. Check LABEL_ROOT={LABEL_ROOT}, LANGS={LANGS}, and file schemas."
        )

    long_df = pd.concat(rows, ignore_index=True)

    # Require >=2 row samples per group
    counts = long_df["group"].value_counts()
    keep_groups = counts[counts >= 2].index
    dropped = [g for g in counts.index if g not in set(keep_groups)]
    long_df = long_df[long_df["group"].isin(keep_groups)].copy()

    group_count = long_df["group"].nunique()
    if group_count < 2:
        raise RuntimeError("Not enough (model,lang) groups with >=2 samples to run Tukey.")

    # Save samples for reproducibility
    write_df(long_df, OUT_SAMPLES)
    if dropped:
        print(f"[INFO] Dropped {len(dropped)} groups with <2 qid samples.")

    # =========================
    # Step 8: Run Tukey HSD
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
    # Step 8b: Plot simultaneous confidence intervals
    # =========================
    fig_height = 100.0
    fig, ax = plt.subplots(figsize=(10, fig_height))
    plot_simultaneous_with_model_blocks(tukey, ax, group_sep=GROUP_SEP)
    fig.set_size_inches(9, 8, forward=True)
    # Customize plot
    #add_model_separators(fig, ax, group_sep=GROUP_SEP, linewidth=1.0, alpha=0.5)
    level_palette = color_tukey_by_taxonomy(
        fig,
        ax,
        taxonomy_csv=TAXONOMY_CSV,
        group_sep=GROUP_SEP,
        default_level=0,   # e.g. color "raw" or unknown langs consistently
        linewidth=2.5,
    )
    taxonomy_legend(ax, level_to_rgba=level_palette, title="Taxonomy level", loc="upper left")
    center_x_axis_at_zero(ax)
    ax.set_xlim(-0.1, 1.75)
    ax.tick_params(axis="y", pad=8, labelsize=12)

    # Axis labels and title
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(None)


    plt.tight_layout()
    plt.savefig(OUT_SIMUL_SVG, format="svg")
    plt.savefig(OUT_SIMUL_PDF, format="pdf", bbox_inches="tight", pad_inches=0.02)
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
