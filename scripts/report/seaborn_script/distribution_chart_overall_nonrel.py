#!/usr/bin/env python3
from pathlib import Path
from typing import List, Dict

import csv
import pandas as pd
from pandas.errors import ParserError
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Config
# =========================
TREC_DL_YEAR = "2022"
MODEL = "gpt-oss-20b"    # e.g. "gpt-oss-20b", "qwen3-32b-v1", ...

# Where the baseline figs live (just to get project root)
BASELINE_DIR = Path("outputs") / "baseline" / TREC_DL_YEAR / MODEL
PROJECT_ROOT = BASELINE_DIR.parents[3]           # .../<project_root>/outputs/...

# Output figure
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = FIG_DIR / "nonrel_nist" / "bar_chart" / f"nonreloverall_{MODEL}_{TREC_DL_YEAR}_lang.png"
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Where the LLM label CSVs live
LABEL_DIR = Path("outputs") / "llm_label" / f"trec_dl_{TREC_DL_YEAR}" / MODEL

# Only keep these “language” variants.
# Set to [] or None to include all non-raw files.
TARGET_LANGS: List[str] = ["eng_word", "eng_word_crit"]

# Relevance scores used by the models
SCORES: List[int] = [0, 1, 2, 3]


# =========================
# Helpers
# =========================
def read_csv_safe(path: Path) -> pd.DataFrame:
    """
    Read a CSV robustly:
    - First try the default fast pandas parser.
    - If we hit a ParserError (e.g. 'Expected 8 fields, saw 10'),
      re-parse the file with Python's csv module to:
        * print malformed lines (wrong number of columns)
        * drop them and return a cleaned DataFrame.
    """
    try:
        return pd.read_csv(path)
    except ParserError as e:
        print(f"[WARN] ParserError in {path.name}: {e}")
        print("[INFO] Scanning for malformed lines with csv.reader ...")

        bad_count = 0
        good_rows = []
        header = None

        # Re-parse manually with csv module
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                print(f"[WARN] {path.name} appears to be empty.")
                return pd.DataFrame()

            expected_cols = len(header)
            for lineno, row in enumerate(reader, start=2):
                if len(row) != expected_cols:
                    bad_count += 1
                    print(
                        f"[BAD LINE] {path.name}:{lineno} "
                        f"(expected {expected_cols} fields, got {len(row)})"
                    )
                    joined = ",".join(row)
                    if len(joined) > 200:
                        joined = joined[:200] + "..."
                    print("          " + joined)
                else:
                    good_rows.append(row)

        print(
            f"[INFO] Found {bad_count} malformed line(s) in {path.name}. "
            "Building DataFrame from remaining rows."
        )

        if not good_rows:
            # only header and bad rows -> empty df with correct columns
            return pd.DataFrame(columns=header)

        return pd.DataFrame(good_rows, columns=header)


def parse_lang_from_filename(path: Path) -> str:
    """
    Given a filename like:
        gpt-oss-20b_trecdl_2022_eng_mult_labels.csv
        gpt-oss-20b_trecdl_2022_eng_vi_between_labels.csv
    return:
        "eng_mult"
        "eng_vi_between"

    Pattern is assumed:
        {MODEL}_trecdl_{YEAR}_{lang}_labels.csv
    where {lang} may itself contain underscores.
    """
    parts = path.stem.split("_")
    # Expect at least: [MODEL, 'trecdl', YEAR, <lang...>, 'labels']
    if len(parts) >= 5 and parts[1] == "trecdl":
        # join everything between YEAR and 'labels'
        lang = "_".join(parts[3:-1])
    else:
        # fallback: second last token
        lang = parts[-2] if len(parts) >= 2 else "unknown"
    return lang


def filter_nist_nonrel(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """
    Keep only rows where the NIST / gold relevance == 0.
    Tries 'relevance' first, then 'NIST_relevance'.
    """
    if "relevance" in df.columns:
        return df[df["relevance"] == 0]
    elif "NIST_relevance" in df.columns:
        return df[df["NIST_relevance"] == 0]
    else:
        print(
            f"[WARN] {context}: no NIST column found ('relevance' or 'NIST_relevance'); "
            f"using ALL rows instead of NIST==0."
        )
        return df


# =========================
# Data loading
# =========================
def load_label_distributions(label_dir: Path) -> pd.DataFrame:
    """
    Build a long-form DataFrame (NIST non-relevant only):

        variant : 'raw' for raw file, otherwise the parsed lang (e.g. 'eng', 'vi_trans_q')
        lang    : same as variant
        score   : llm_relevance (0..3)
        prop    : proportion in [0, 1] among rows where NIST relevance == 0
    """
    records: List[Dict] = []

    # ---------- RAW ----------
    raw_file = label_dir / f"{MODEL}_trecdl_{TREC_DL_YEAR}_raw_labels.csv"
    if raw_file.exists():
        df_raw = read_csv_safe(raw_file)
        df_raw["llm_relevance"] = pd.to_numeric(df_raw["llm_relevance"], errors="coerce")

        # filter to NIST non-relevant only
        df_raw = filter_nist_nonrel(df_raw, context=f"RAW ({raw_file.name})")

        # use only rows with a valid numeric label
        df_valid = df_raw.dropna(subset=["llm_relevance"])
        if df_valid.empty:
            print("[WARN] No valid llm_relevance labels in RAW after NIST==0 filtering.")
        else:
            counts_raw = df_valid["llm_relevance"].value_counts().to_dict()
            total_raw = len(df_valid)

            for s in SCORES:
                prop = counts_raw.get(s, 0) / total_raw if total_raw > 0 else 0.0
                records.append(
                    {
                        "lang": "raw",
                        "variant": "raw",
                        "score": s,
                        "prop": prop,
                    }
                )
    else:
        print(f"[WARN] Missing raw labels file: {raw_file}")

    # ---------- OTHER VARIANTS ----------
    seen_langs: set[str] = set()

    pattern = f"{MODEL}_trecdl_{TREC_DL_YEAR}_*_labels.csv"
    for file_path in label_dir.glob(pattern):
        # skip raw (already handled)
        if file_path.name.endswith("_raw_labels.csv"):
            continue

        lang = parse_lang_from_filename(file_path)

        # optionally restrict to selected variants
        if TARGET_LANGS and (lang not in TARGET_LANGS):
            continue

        # avoid accidental duplicates for the same lang
        if lang in seen_langs:
            print(f"[SKIP] Duplicate file for lang '{lang}': {file_path.name}")
            continue
        seen_langs.add(lang)

        df_other = read_csv_safe(file_path)
        df_other["llm_relevance"] = pd.to_numeric(
            df_other["llm_relevance"], errors="coerce"
        )

        # filter to NIST non-relevant only
        df_other = filter_nist_nonrel(df_other, context=file_path.name)

        df_valid = df_other.dropna(subset=["llm_relevance"])
        if df_valid.empty:
            print(
                f"[INFO] No valid llm_relevance labels in {file_path.name} "
                "after NIST==0 filtering."
            )
            continue

        counts = df_valid["llm_relevance"].value_counts().to_dict()
        total = len(df_valid)
        variant_name = lang

        for s in SCORES:
            prop = counts.get(s, 0) / total if total > 0 else 0.0
            records.append(
                {
                    "lang": lang,
                    "variant": variant_name,
                    "score": s,
                    "prop": prop,
                }
            )

    df = pd.DataFrame.from_records(records)

    # Debug: ensure each variant's props sum to ~1
    if not df.empty:
        print("\n[DEBUG] Sum of props per variant (should be ~1.0 each, NIST==0 only):")
        print(df.groupby("variant")["prop"].sum())

    return df


# =========================
# Plotting
# =========================
def plot_nonrel_distribution(df: pd.DataFrame, title: str, out_path: Path) -> None:
    sns.set_theme(style="darkgrid")
    plt.style.use("dark_background")

    # Order: raw first, then others alphabetically
    variants = sorted(
        df["variant"].unique(),
        key=lambda v: (v != "raw", v),
    )
    scores = sorted(df["score"].unique())

    # Aggregate just in case there are duplicates
    grouped = df.groupby(["score", "variant"], as_index=False)["prop"].sum()

    # X-axis = scores; for each score, we show one bar per variant
    n_scores = len(scores)
    n_variants = len(variants)
    x_base = list(range(n_scores))

    # Total width of each group (per score)
    group_width = 0.8
    bar_width = group_width / max(1, n_variants)

    fig, ax = plt.subplots(figsize=(6, 6))

    # One color per variant
    colors = sns.color_palette(n_colors=n_variants)
    variant_colors: Dict[str, tuple] = {v: colors[i] for i, v in enumerate(variants)}

    for j, variant in enumerate(variants):
        var_data = grouped[grouped["variant"] == variant].set_index("score")["prop"]
        heights = [float(var_data.get(s, 0.0)) for s in scores]

        # Shift bars within each group
        offsets = [
            x + (j - (n_variants - 0.5) / 2) * bar_width for x in x_base
        ]

        ax.bar(
            offsets,
            heights,
            width=bar_width,
            label=variant,
            color=variant_colors[variant],
            edgecolor="black",
            linewidth=0.5,
        )

        ax.grid(False)

        # Optional: label bars if tall enough
        for x, h in zip(offsets, heights):
            if h >= 0.04:
                ax.text(
                    x,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    color="white",
                )

    ax.set_xticks(x_base)
    ax.set_xticklabels([str(s) for s in scores], fontsize=9)
    ax.set_xlabel("LLM Relevance Score")
    ax.set_ylabel("Proportion (NIST non-relevant only)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=12)

    ax.legend(
        title="Variant",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=1.0,
        edgecolor="white",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
if __name__ == "__main__":
    df_labels = load_label_distributions(LABEL_DIR)

    if df_labels.empty:
        print("[ERROR] No data found after NIST==0 filtering; check LABEL_DIR and columns.")
    else:
        plot_nonrel_distribution(
            df_labels,
            title="NIST non-relevant pairs – LLM label distribution across variants",
            out_path=FIG_PATH,
        )
        print(f"[OK] Saved figure to {FIG_PATH}")
