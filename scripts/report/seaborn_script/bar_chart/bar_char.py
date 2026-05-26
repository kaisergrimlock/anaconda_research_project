import sys
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# =========================
# Repo root bootstrap
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEABORN_ROOT = THIS_FILE.parents[1]

if str(SEABORN_ROOT) not in sys.path:
    sys.path.insert(0, str(SEABORN_ROOT))

# =========================
# Repo imports
# =========================
from helpers.lang_profiles import get_langs
from scripts.csv_helpers import bump_field_limit
from helpers.output_writer import write_df

# =========================
# Parameters
# =========================
ALPHA = 0.05
LABELS = [0, 1, 2, 3]

LANG_PROFILE = "crit_instruct_test"
LANGS: List[str] = get_langs(LANG_PROFILE)

METRIC = "mean_diff"
TREC_DL_YEARS = ["2021", "2022"]

# =========================
# Config
# =========================
OUT_DIR = (
    Path("figures")
    / "2021_2022"
    / "bar_chart"
    / f"all_models_all_{LANG_PROFILE}"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TUKEY_CSV = OUT_DIR / "tukey_hsd_table_all_groups.csv"
OUT_TUKEY_TEX = OUT_DIR / "tukey_hsd_table_all_groups.tex"
OUT_SAMPLES = OUT_DIR / "samples_long.csv"

OUT_BAR_SVG = OUT_DIR / f"bar_chart_{METRIC}_2021_2022.svg"
OUT_BAR_PDF = OUT_DIR / f"bar_chart_{METRIC}_2021_2022.pdf"
OUT_SUMMARY_CSV = OUT_DIR / f"bar_chart_summary_{METRIC}_2021_2022.csv"

TAXONOMY_CSV = Path(__file__).resolve().parents[1] / "lang.csv"

KEY_COLS = ["qid", "pid"]
GROUP_SEP = "|"


# =========================
# File discovery
# =========================
def find_llm_files(year: str) -> Dict[str, List[Path]]:
    label_root = Path("outputs/llm_label") / f"trec_dl_{year}"

    if not label_root.exists():
        raise FileNotFoundError(f"LABEL_ROOT not found: {label_root}")

    model_files: Dict[str, List[Path]] = {}

    for model_dir in label_root.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        csv_files = list(
            model_dir.glob(f"{model_name}_trecdl_{year}_*_labels.csv")
        )

        if csv_files:
            model_files[model_name] = csv_files
        else:
            print(f"[WARN] No label CSVs found for model {model_name} in {year}")

    return model_files


def get_lang_from_filename(file_path: Path, model: str) -> Optional[str]:
    fname = file_path.name
    pattern = rf"^{re.escape(model)}_trecdl_\d{{4}}_(.+?)_labels\.csv$"

    match = re.search(pattern, fname)
    return match.group(1) if match else None


# =========================
# Loading helpers
# =========================
def load_invalid_keys(year: str) -> set[tuple[int, str]]:
    path = Path(__file__).resolve().parent / f"invalid_{year}.csv"

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


def load_labels(file_path: Path, invalid_keys: set[tuple[int, str]]) -> pd.DataFrame:
    bump_field_limit()

    df = pd.read_csv(file_path)

    required = {"qid", "pid", "relevance", "llm_relevance"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {file_path}")

    df["NIST"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["LLM"] = pd.to_numeric(df["llm_relevance"], errors="coerce")

    df = df.dropna(subset=["NIST", "LLM", "qid", "pid"]).copy()

    df["NIST"] = df["NIST"].astype(int)
    df["LLM"] = df["LLM"].astype(int)

    df = df[df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)].copy()

    df["qid"] = pd.to_numeric(df["qid"], errors="coerce")
    df = df.dropna(subset=["qid"]).copy()

    df["qid"] = df["qid"].astype(int)
    df["pid"] = df["pid"].astype(str)

    if invalid_keys:
        keys = pd.Index(list(zip(df["qid"].to_numpy(), df["pid"].to_numpy())))
        df = df[~keys.isin(invalid_keys)].copy()

    return df


# =========================
# Metric helpers
# =========================
def per_row_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    base = (
        df.dropna(subset=["qid", "pid"])
        .drop_duplicates(subset=["qid", "pid"])
        .copy()
    )

    if metric == "mean_diff":
        base["value"] = base["LLM"] - base["NIST"]

    elif metric == "mae_4pt":
        base["value"] = (base["LLM"] - base["NIST"]).abs()

    elif metric == "disagree_rate":
        base["value"] = (base["LLM"] != base["NIST"]).astype(float)

    else:
        raise ValueError(
            "Unknown METRIC. Use 'mean_diff', 'mae_4pt', or 'disagree_rate'."
        )

    return base[["qid", "pid", "value"]]


# =========================
# Tukey helpers
# =========================
def tukey_to_df(tukey) -> pd.DataFrame:
    table = tukey.summary().data
    header = table[0]
    body = table[1:]

    df = pd.DataFrame(body, columns=header)

    for col in ["meandiff", "p-adj", "lower", "upper"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "reject" in df.columns:
        df["reject"] = (
            df["reject"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
        )

    return df


def safe_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)

    return text.strip("_")


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    fmt = df.copy()

    for col in ["meandiff", "p-adj", "lower", "upper"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:.6g}" if pd.notnull(x) else "")

    return fmt.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
    )


# =========================
# Summary helpers
# =========================
def make_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_df.groupby(["model", "lang"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["sem"] = summary["std"] / summary["count"].pow(0.5)

    return summary


def get_base_lang(lang: str) -> str:
    """
    Extract the base language before the first underscore.

    Examples:
      eng -> eng
      eng_instruct -> eng
      ru_word -> ru
      vi_qp -> vi
    """
    match = re.match(r"^([^_]+)", str(lang))
    return match.group(1) if match else str(lang)


def get_language_order_by_taxonomy(summary: pd.DataFrame) -> List[str]:
    if not TAXONOMY_CSV.exists():
        raise FileNotFoundError(f"Cannot find taxonomy file: {TAXONOMY_CSV}")

    taxonomy = pd.read_csv(TAXONOMY_CSV)

    taxonomy.columns = (
        taxonomy.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    if not {"lang", "taxonomy"}.issubset(taxonomy.columns):
        raise ValueError(
            f"lang.csv must contain columns: lang, taxonomy. "
            f"Found columns: {list(taxonomy.columns)}"
        )

    taxonomy["taxonomy"] = pd.to_numeric(
        taxonomy["taxonomy"],
        errors="coerce",
    )

    taxonomy = taxonomy.dropna(subset=["taxonomy"])

    level_map = dict(zip(taxonomy["lang"], taxonomy["taxonomy"]))

    langs = list(summary["lang"].unique())

    ordered_langs = sorted(
        langs,
        key=lambda lang: (
            level_map.get(get_base_lang(lang), 999),
            get_base_lang(lang),
            lang,
        ),
    )

    return ordered_langs


# =========================
# Plotting
# =========================
def plot_grouped_bar_chart(summary: pd.DataFrame) -> None:
    lang_order = get_language_order_by_taxonomy(summary)

    summary = summary.copy()

    summary["lang"] = pd.Categorical(
        summary["lang"],
        categories=lang_order,
        ordered=True,
    )

    summary = summary.sort_values(["lang", "model"])

    pivot_mean = summary.pivot_table(
        index="lang",
        columns="model",
        values="mean",
        aggfunc="mean",
        observed=False,
    )

    pivot_sem = summary.pivot_table(
        index="lang",
        columns="model",
        values="sem",
        aggfunc="mean",
        observed=False,
    )

    pivot_mean = pivot_mean.reindex(lang_order)
    pivot_sem = pivot_sem.reindex(lang_order)

    fig, ax = plt.subplots(figsize=(13, 6))

    pivot_mean.plot(
        kind="bar",
        ax=ax,
        yerr=pivot_sem,
        capsize=3,
        edgecolor="black",
        width=0.8,
    )

    ax.set_ylim(0, None)
    ax.set_xlabel("Language")

    if METRIC == "mean_diff":
        ax.set_ylabel("Mean LLM - NIST")
        ax.set_title("Mean relevance inflation grouped by language, 2021–2022")

    elif METRIC == "mae_4pt":
        ax.set_ylabel("Mean absolute error")
        ax.set_title("Mean absolute error grouped by language, 2021–2022")

    elif METRIC == "disagree_rate":
        ax.set_ylabel("Disagreement rate")
        ax.set_title("Disagreement rate grouped by language, 2021–2022")

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    ax.legend(title="Model", ncols=1)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(OUT_BAR_SVG, format="svg")
    plt.savefig(OUT_BAR_PDF, format="pdf", bbox_inches="tight", pad_inches=0.02)

    plt.close(fig)


# =========================
# Main
# =========================
def main() -> None:
    rows: List[pd.DataFrame] = []
    skipped = 0

    for year in TREC_DL_YEARS:
        print("\n==============================")
        print(f"[INFO] Processing TREC DL {year}")
        print("==============================")

        model_files = find_llm_files(year)
        invalid_keys = load_invalid_keys(year)

        print(f"[INFO] Found {len(model_files)} models for {year}")

        for model, files in model_files.items():
            for file_path in files:
                lang = get_lang_from_filename(file_path, model)

                if lang not in LANGS:
                    continue

                try:
                    df = load_labels(file_path, invalid_keys)

                    pair_count = (
                        df.dropna(subset=["qid", "pid"])
                        .drop_duplicates(subset=["qid", "pid"])
                        .shape[0]
                    )

                    print(f"[INFO] {year} {model} {lang}: {pair_count} valid pairs")

                    perrow = per_row_metric(df, METRIC)

                    if perrow.empty:
                        continue

                    perrow["year"] = year
                    perrow["model"] = model
                    perrow["lang"] = lang

                    perrow["group"] = perrow["model"] + GROUP_SEP + perrow["lang"]

                    rows.append(
                        perrow[
                            ["year", "group", "model", "lang", "qid", "pid", "value"]
                        ]
                    )

                except Exception as e:
                    skipped += 1
                    print(f"[SKIP] {year} {model} {lang} ({file_path.name}): {e}")

    if not rows:
        raise RuntimeError(
            f"No samples produced. Check TREC_DL_YEARS={TREC_DL_YEARS}, "
            f"LANGS={LANGS}, and file schemas."
        )

    long_df = pd.concat(rows, ignore_index=True)

    counts = long_df["group"].value_counts()
    keep_groups = counts[counts >= 2].index

    long_df = long_df[long_df["group"].isin(keep_groups)].copy()

    if long_df["group"].nunique() < 2:
        raise RuntimeError("Not enough groups with >=2 samples.")

    write_df(long_df, OUT_SAMPLES)

    # =========================
    # Tukey HSD table
    # =========================
    tukey = pairwise_tukeyhsd(
        endog=long_df["value"].to_numpy(),
        groups=long_df["group"].to_numpy(),
        alpha=ALPHA,
    )

    tukey_df = tukey_to_df(tukey)
    write_df(tukey_df, OUT_TUKEY_CSV)

    latex = to_latex_table(
        tukey_df,
        caption=(
            f"Tukey HSD across all model-language groups for {METRIC}, "
            f"TREC DL 2021--2022, FWER={ALPHA}."
        ),
        label=f"tab:tukey_all_models_all_langs_2021_2022_{safe_slug(METRIC)}",
    )

    OUT_TUKEY_TEX.write_text(latex, encoding="utf-8")

    # =========================
    # Bar chart
    # =========================
    summary = make_summary(long_df)
    write_df(summary, OUT_SUMMARY_CSV)

    plot_grouped_bar_chart(summary)

    print(f"\n[OK] Wrote samples: {OUT_SAMPLES}")
    print(f"[OK] Wrote summary CSV: {OUT_SUMMARY_CSV}")
    print(f"[OK] Wrote Tukey CSV: {OUT_TUKEY_CSV}")
    print(f"[OK] Wrote Tukey TeX: {OUT_TUKEY_TEX}")
    print(f"[OK] Wrote bar chart SVG: {OUT_BAR_SVG}")
    print(f"[OK] Wrote bar chart PDF: {OUT_BAR_PDF}")

    if skipped:
        print(f"[INFO] Skipped {skipped} files due to errors.")


if __name__ == "__main__":
    main()