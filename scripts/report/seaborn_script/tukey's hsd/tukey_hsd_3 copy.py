
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from helpers.draw import color_tukey_by_taxonomy, center_x_axis_at_zero, taxonomy_legend, add_model_separators

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
OUT_DIR = Path("figures") / TREC_DL_YEAR / "tukey_hsd" / "all_models_all_last"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TUKEY_CSV = OUT_DIR / "tukey_hsd_table_all_groups.csv"
OUT_TUKEY_TEX = OUT_DIR / "tukey_hsd_table_all_groups.tex"
OUT_SIMUL_SVG = OUT_DIR / f"tukey_hsd_plot_simultaneous_all_groups_{TREC_DL_YEAR}.svg"
OUT_SIMUL_PDF = OUT_DIR / f"tukey_hsd_plot_simultaneous_all_groups_{TREC_DL_YEAR}.pdf"
OUT_SAMPLES   = OUT_DIR / "tukey_samples_long.csv"
OUT_EXCLUDED  = Path(__file__).resolve().parent / f"invalid_{TREC_DL_YEAR}.csv"
GROUP_SEP = "|"
TAXONOMY_CSV = Path(__file__).resolve().parents[1] / "lang.csv"
# ========================
# Parameters
# ========================
ALPHA = 0.05
LABELS = [0, 1, 2, 3]
#LANGS: List[str] = ["vi", "vi_corrected", "th", "th_corrected", "ko", "ko_corrected"]  # if empty, allow all langs found
#LANGS: List[str] = ["raw", "eng", "ru", "vi", "th", "sw", "ga", "eng_brackets", "ru_brackets", "vi_brackets", "sw_brackets", "ga_brackets", "th_brackets"]  # if empty, allow all langs found
#LANGS: List[str] = ["raw", "eng", "eng_vi", "eng", "vi_th", "vi"]  # if empty, allow all langs found
#LANGS: List[str] = ["raw", "eng", "eng_mult_2", "eng_mult_3"]
LANGS: List[str] = ["raw", "eng_last", "fr_last", "ru_last", "ar_last", "he_last", "vi_last", "th_last", "sw_last", "ga_last", "zh_last", "hi_last"]
#LANGS: List[str] = ["raw", "eng_first", "fr_first", "ru_first", "ar_first", "he_first", "vi_first", "th_first", "sw_first", "ga_first", "zh_first", "hi_first"]
#LANGS: List[str] = ["raw", "eng", "fr", "ru", "vi", "he", "ar", "th", "sw", "ga", "eng_word", "fr_word", "ru_word", "vi_word", "he_word", "ar_word", "sw_word", "ga_word", "th_word"] 

#LANGS: List[str] = ["raw", "eng", "fr", "ru", "vi", "he", "ar", "th", "sw", "ga", "zh", "hi"] # if empty, allow all langs found
#LANGS: List[str] = ["vi", "vi_corrected", "th", "th_corrected", "ko", "ko_corrected"]  # if empty, allow all langs found
METRIC = "mean_diff"
KEY_COLS = ["qid", "pid"]
REASON_COL = "reason"


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

def load_labels(file_path: Path) -> pd.DataFrame:
    bump_field_limit()
    df = pd.read_csv(file_path)
    requisite = {"qid", "pid", "relevance", "llm_relevance"}
    for r in requisite:
        if r not in df.columns:
            raise ValueError(f"Missing required column '{r}' in {file_path}")
    df["NIST"] = pd.to_numeric(df["relevance"], errors="coerce") # If conversion fails, NaN
    df["LLM"] = pd.to_numeric(df["llm_relevance"], errors="coerce")

    # Keep only rows with valid NIST and LLM labels
    df = df.dropna(subset=["NIST", "LLM"])
    df["NIST"] = df["NIST"].astype(int)
    df["LLM"] = df["LLM"].astype(int)
    valid = df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)
    return df[valid].copy()

def per_row_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Turn a (qid,pid)-level label dataframe into per-row samples:
      qid, pid, value

    These 'value' samples are what Tukey HSD consumes.
    """
    # Keep only valid, unique (qid,pid) pairs so duplicates or missing IDs
    # don't distort per-row samples.
    base = df.dropna(subset=KEY_COLS).drop_duplicates(subset=KEY_COLS).copy()

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

# =========================
# Step 7: Main (data assembly)
# =========================
def main() -> None:
    model_files = find_llm_files()
    print(f"Found {len(model_files)} models under: {LABEL_ROOT}")

    rows: List[pd.DataFrame] = []
    excluded_rows: List[pd.DataFrame] = []
    skipped = 0
    loaded: List[tuple] = []
    common_keys: Optional[set] = None
    lang_model_keys: Dict[str, Dict[str, set]] = {}
    invalid_df: Optional[pd.DataFrame] = None
    invalid_keys: Optional[set] = None
    if OUT_EXCLUDED.exists():
        try:
            invalid_df = pd.read_csv(OUT_EXCLUDED)
        except pd.errors.EmptyDataError:
            invalid_df = pd.DataFrame()
        if invalid_df is not None and not invalid_df.empty and REASON_COL not in invalid_df.columns:
            invalid_df[REASON_COL] = "preexisting_invalid"
        if invalid_df is not None and not invalid_df.empty and set(KEY_COLS).issubset(invalid_df.columns):
            invalid_keys = set(invalid_df[KEY_COLS].drop_duplicates().itertuples(index=False, name=None))

    for model, files in model_files.items():
        for f in files:
            lang = get_lang_from_filename(f, model)

            # strict language include list
            if lang not in LANGS:
                continue
            try:
                df = load_labels(f)
                if invalid_df is not None and not invalid_df.empty:
                    if set(KEY_COLS).issubset(invalid_df.columns):
                        df = df.merge(
                            invalid_df[KEY_COLS].drop_duplicates(),
                            on=KEY_COLS,
                            how="left",
                            indicator=True,
                        )
                        df = df[df["_merge"] == "left_only"].drop(columns=["_merge"])

                key_df = (
                    df.dropna(subset=KEY_COLS)
                    .drop_duplicates(subset=KEY_COLS)
                    .loc[:, KEY_COLS]
                )
                key_set = set(key_df.itertuples(index=False, name=None))
                if common_keys is None:
                    common_keys = key_set
                else:
                    common_keys &= key_set
                lang_model_keys.setdefault(lang, {})[model] = key_set
                loaded.append((model, lang, df))

            except Exception as e:
                skipped += 1
                print(f"[SKIP] {model} {lang} ({f.name}): {e}")

    if not loaded:
        raise RuntimeError(
            f"No samples produced. Check LABEL_ROOT={LABEL_ROOT}, LANGS={LANGS}, and file schemas."
        )

    if not common_keys:
        raise RuntimeError(
            "No common (qid,pid) rows across all valid files."
        )

    common_df = pd.DataFrame(list(common_keys), columns=KEY_COLS)
    missing_map_per_lang: Dict[str, Dict[tuple, List[str]]] = {}
    for lang, model_keys in lang_model_keys.items():
        models = sorted(model_keys.keys())
        all_keys: set = set()
        for ks in model_keys.values():
            all_keys |= ks
        missing_for_lang: Dict[tuple, List[str]] = {}
        for key in all_keys:
            missing = [m for m in models if key not in model_keys[m]]
            if missing:
                missing_for_lang[key] = missing
        missing_map_per_lang[lang] = missing_for_lang

    for model, lang, df in loaded:
        key_df = (
            df.dropna(subset=KEY_COLS)
            .drop_duplicates(subset=KEY_COLS)
            .loc[:, KEY_COLS]
        )
        excluded = key_df.merge(common_df, on=KEY_COLS, how="left", indicator=True)
        excluded = excluded[excluded["_merge"] == "left_only"].drop(columns=["_merge"])
        if not excluded.empty:
            excluded = excluded.copy()
            excluded["model"] = model
            excluded["lang"] = lang
            reasons = []
            for row in excluded[KEY_COLS].itertuples(index=False, name=None):
                missing_models = missing_map_per_lang.get(lang, {}).get(row, [])
                missing_models = [m for m in missing_models if m != model]
                if missing_models:
                    reasons.append(f"missing_in_models={','.join(missing_models)}")
                else:
                    reasons.append("not_in_common_keys")
            excluded[REASON_COL] = reasons
            excluded_rows.append(excluded[["model", "lang"] + KEY_COLS + [REASON_COL]])

        df = df.merge(common_df, on=KEY_COLS, how="inner")
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
        perrow["group"] = perrow["model"] + GROUP_SEP + perrow["lang"]

        rows.append(perrow[["group", "model", "lang", "qid", "pid", "value"]])

    if not rows:
        raise RuntimeError(
            "No samples produced after intersecting on (qid,pid)."
        )

    long_df = pd.concat(rows, ignore_index=True)

    excluded_df: Optional[pd.DataFrame] = None
    if excluded_rows:
        excluded_df = pd.concat(excluded_rows, ignore_index=True)
        if invalid_keys is not None:
            mask = ~excluded_df[KEY_COLS].apply(tuple, axis=1).isin(invalid_keys)
            excluded_df = excluded_df[mask]
        if invalid_df is not None and not invalid_df.empty:
            excluded_df = pd.concat([invalid_df, excluded_df], ignore_index=True)
        if REASON_COL not in excluded_df.columns:
            excluded_df[REASON_COL] = "not_in_common_keys"
        excluded_df = excluded_df.drop_duplicates(subset=KEY_COLS)

    # Require >=2 row samples per group
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
    fig, ax = plt.subplots(figsize=(10, 8))
    tukey.plot_simultaneous(ax=ax)
    # Customize plot
    add_model_separators(fig, ax, group_sep=GROUP_SEP, linewidth=1.0, alpha=0.5)
    level_palette = color_tukey_by_taxonomy(
        fig,
        ax,
        taxonomy_csv=TAXONOMY_CSV,
        group_sep=GROUP_SEP,
        default_level=0,   # e.g. color "raw" or unknown langs consistently
        linewidth=2.5,
    )
    taxonomy_legend(ax, level_to_rgba=level_palette, title="Resource Class", loc="upper left")
    center_x_axis_at_zero(ax)

    # Axis labels and title
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(None)
    ax.set_xlim(-0.1, 1.75)


    plt.tight_layout()
    plt.savefig(OUT_SIMUL_SVG, format="svg")
    plt.savefig(OUT_SIMUL_PDF, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


    # =========================
    # Final logging
    # =========================
    print(f"\n[OK] Wrote samples: {OUT_SAMPLES}")
    if OUT_EXCLUDED.exists():
        print(f"[OK] Wrote excluded rows: {OUT_EXCLUDED}")
    print(f"[OK] Wrote Tukey CSV: {OUT_TUKEY_CSV}")
    print(f"[OK] Wrote Tukey TeX: {OUT_TUKEY_TEX}")
    print(f"[OK] Wrote plot: {OUT_SIMUL_SVG}")
    if excluded_df is not None:
        write_df(excluded_df, OUT_EXCLUDED)
        print(f"[OK] Wrote excluded rows: {OUT_EXCLUDED}")
    if skipped:
        print(f"[INFO] Skipped {skipped} files due to errors (see logs above).")

if __name__ == "__main__":
    main()
