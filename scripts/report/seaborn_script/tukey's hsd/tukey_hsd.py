#!/usr/bin/env python3
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

# Root that contains:
# outputs/llm_label/trec_dl_2022/<MODEL>/<MODEL>_trecdl_2022_<LANG>_labels.csv
LABEL_ROOT = Path("outputs/llm_label") / f"trec_dl_{TREC_DL_YEAR}"

# Output: single table + single plot for all (model,lang)
OUT_DIR = Path("figures") / TREC_DL_YEAR / "tukey_hsd" / "all_models_all_langs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TUKEY_CSV = OUT_DIR / "tukey_hsd_table_all_groups.csv"
OUT_TUKEY_TEX = OUT_DIR / "tukey_hsd_table_all_groups.tex"
OUT_SIMUL_SVG = OUT_DIR / "tukey_hsd_plot_simultaneous_all_groups.svg"
OUT_SAMPLES   = OUT_DIR / "tukey_samples_long.csv"

ALPHA = 0.05
LABELS = [0, 1, 2, 3]

# Choose which languages to include
# - list: only those langs
# - "all": include all discovered langs
LANGS: Union[str, List[str]] = ["raw", "eng", "vi", "ru", "sw", "ga"]  # or "all"

# Metric to build per-qid samples for Tukey:
# - "pos_rate": fraction of passages where LLM predicts "relevant" (LLM>0) per qid
# - "mae_4pt": mean absolute error per qid
METRIC = "pos_rate"

# qid column candidates
QID_CANDIDATES = ["qid", "query_id", "topic_id"]

# Separator used in group names (avoid confusion with model names)
GROUP_SEP = "|"

# Define what counts as "positive/relevant" in LLM labels:
# - If you want {1,2,3} as positive (anything > 0), keep as-is.
# - If you want {2,3} only, change to: lambda s: s >= 2
LLM_POSITIVE_FN = lambda s: s > 0


# =========================
# Helpers
# =========================
def safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def want_lang(lang: str, langs) -> bool:
    if langs == "all":
        return True
    return lang in set(map(str, langs))


def detect_qid_column(df: pd.DataFrame) -> str:
    for c in QID_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find qid column. Tried: {QID_CANDIDATES}. Found: {list(df.columns)}"
    )


def parse_lang_from_filename(path: Path, model: str) -> str:
    """
    Expected:
      <MODEL>_trecdl_<YEAR>_<LANG>_labels.csv
    """
    name = path.name
    prefix = f"{model}_trecdl_{TREC_DL_YEAR}_"
    suffix = "_labels.csv"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    m = re.search(rf"_trecdl_{re.escape(TREC_DL_YEAR)}_(.+?)_labels\.csv$", name)
    if m:
        return m.group(1)
    return "unknown"


def discover_model_files() -> Dict[str, List[Path]]:
    """
    Returns model -> list of label CSVs.
    """
    if not LABEL_ROOT.exists():
        raise FileNotFoundError(f"LABEL_ROOT not found: {LABEL_ROOT}")

    out: Dict[str, List[Path]] = {}
    for model_dir in sorted([p for p in LABEL_ROOT.iterdir() if p.is_dir()]):
        model = model_dir.name
        files = sorted(model_dir.glob(f"{model}_trecdl_{TREC_DL_YEAR}_*_labels.csv"))
        if files:
            out[model] = files

    if not out:
        raise RuntimeError(
            f"No model label files found under {LABEL_ROOT}. "
            f"Expected pattern: <MODEL>_trecdl_{TREC_DL_YEAR}_<LANG>_labels.csv"
        )
    return out


def load_labels_csv(path: Path) -> pd.DataFrame:
    """
    Load a labels CSV, coerce relevance columns, and keep only valid label pairs.
    NOTE: We still validate BOTH columns exist and are within LABELS, even if
    the chosen metric only uses LLM. This keeps your dataset consistent.
    """
    bump_field_limit()
    df = pd.read_csv(path)

    if "relevance" not in df.columns or "llm_relevance" not in df.columns:
        raise ValueError(
            f"Expected 'relevance' and 'llm_relevance' in {path}, got: {list(df.columns)}"
        )

    df["NIST"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["LLM"]  = pd.to_numeric(df["llm_relevance"], errors="coerce")

    valid = df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)
    return df[valid].copy()


def per_qid_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Returns df with columns: qid, value
    one row per query => one Tukey sample.
    """
    qid_col = detect_qid_column(df)

    if metric == "mae_4pt":
        out = (
            df.assign(abs_err=(df["NIST"] - df["LLM"]).abs())
              .groupby(qid_col, as_index=False)["abs_err"]
              .mean()
              .rename(columns={qid_col: "qid", "abs_err": "value"})
        )
        return out

    if metric == "pos_rate":
        # FIX: keep the original rows (no filtering to NIST==0).
        # Compute "positive/relevant rate" per qid over ALL passages.
        base = df.copy()
        base["is_pos"] = LLM_POSITIVE_FN(base["LLM"]).astype(float)
        out = (
            base.groupby(qid_col, as_index=False)["is_pos"]
                .mean()
                .rename(columns={qid_col: "qid", "is_pos": "value"})
        )
        return out

    raise ValueError("Unknown METRIC. Use 'pos_rate' or 'mae_4pt'.")


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
    model_files = discover_model_files()
    print(f"Found {len(model_files)} models under: {LABEL_ROOT}")

    # Build one long df across *all* (model, lang)
    rows = []
    skipped = 0

    for model, files in model_files.items():
        for f in files:
            lang = parse_lang_from_filename(f, model)
            if not want_lang(lang, LANGS):
                continue

            try:
                df = load_labels_csv(f)
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
        raise RuntimeError(f"No samples produced. Check LANGS={LANGS} and LABEL_ROOT={LABEL_ROOT}")

    long_df = pd.concat(rows, ignore_index=True)

    # Require >=2 samples per group for Tukey stability
    counts = long_df["group"].value_counts()
    keep_groups = counts[counts >= 2].index
    long_df = long_df[long_df["group"].isin(keep_groups)].copy()

    if long_df["group"].nunique() < 2:
        raise RuntimeError("Not enough (model,lang) groups with >=2 samples to run Tukey.")

    # Save the long samples used
    write_df(long_df, OUT_SAMPLES)

    # One Tukey across all groups
    tukey = pairwise_tukeyhsd(
        endog=long_df["value"].to_numpy(),
        groups=long_df["group"].to_numpy(),
        alpha=ALPHA,
    )
    tukey_df = tukey_to_df(tukey)

    # Outputs: one table + one plot
    write_df(tukey_df, OUT_TUKEY_CSV)

    latex = to_latex_table(
        tukey_df,
        caption=f"Tukey HSD across all (model,lang) groups for {METRIC}, FWER={ALPHA}.",
        label=f"tab:tukey_all_models_all_langs_{safe_slug(METRIC)}",
    )
    OUT_TUKEY_TEX.write_text(latex, encoding="utf-8")

    # Plot (single graph). Color all "...|raw" tick labels red.
    fig, ax = plt.subplots(figsize=(10, 8))
    tukey.plot_simultaneous(ax=ax)

    for tick in ax.get_yticklabels():
        label = tick.get_text()
        if label.endswith(f"{GROUP_SEP}raw") or label == "raw":
            tick.set_color("red")

    plt.tight_layout()
    plt.savefig(OUT_SIMUL_SVG, format="svg")
    plt.close(fig)

    print(f"\n[OK] Wrote samples: {OUT_SAMPLES}")
    print(f"[OK] Wrote Tukey CSV: {OUT_TUKEY_CSV}")
    print(f"[OK] Wrote Tukey TeX: {OUT_TUKEY_TEX}")
    print(f"[OK] Wrote plot: {OUT_SIMUL_SVG}")
    if skipped:
        print(f"[INFO] Skipped {skipped} files due to errors (see logs above).")


if __name__ == "__main__":
    main()
