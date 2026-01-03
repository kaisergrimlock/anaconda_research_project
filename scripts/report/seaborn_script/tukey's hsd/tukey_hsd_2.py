#!/usr/bin/env python3
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Dict, List, Union, Optional

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
LABEL_ROOT = Path("outputs/llm_label") / f"trec_dl_{TREC_DL_YEAR}"

OUT_DIR = Path("figures") / TREC_DL_YEAR / "tukey_hsd" / "all_models_all_langs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TUKEY_CSV = OUT_DIR / "tukey_hsd_table_all_groups.csv"
OUT_TUKEY_TEX = OUT_DIR / "tukey_hsd_table_all_groups.tex"
OUT_SIMUL_SVG = OUT_DIR / "tukey_hsd_plot_simultaneous_all_groups.svg"
OUT_SAMPLES   = OUT_DIR / "tukey_samples_long.csv"

ALPHA = 0.05
LABELS = [0, 1, 2, 3]

LANGS: Union[str, List[str]] = ["raw", "eng", "vi", "ru", "sw", "ga"]  # or "all"

# Row-level metric (computed per qid-pid row)
METRIC = "mean_diff"  # "mean_diff" | "mae_4pt" | "disagree_rate"

QID_CANDIDATES = ["qid", "query_id", "topic_id"]
PID_CANDIDATES = ["pid", "passage_id", "docid", "document_id"]

GROUP_SEP = "|"


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
    raise ValueError(f"Could not find qid column. Tried: {QID_CANDIDATES}. Found: {list(df.columns)}")


def detect_pid_column(df: pd.DataFrame) -> Optional[str]:
    for c in PID_CANDIDATES:
        if c in df.columns:
            return c
    return None


def parse_lang_from_filename(path: Path, model: str) -> str:
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
    bump_field_limit()
    df = pd.read_csv(path)

    if "relevance" not in df.columns or "llm_relevance" not in df.columns:
        raise ValueError(f"Expected 'relevance' and 'llm_relevance' in {path}, got: {list(df.columns)}")

    df["NIST"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["LLM"]  = pd.to_numeric(df["llm_relevance"], errors="coerce")

    valid = df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)
    return df[valid].copy()


def per_row_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    One replicate per row (qid-pid pair if pid exists).
    Returns columns:
      - qid
      - pid (if available)
      - value
    """
    qid_col = detect_qid_column(df)
    pid_col = detect_pid_column(df)  # may be None

    base = df.copy()

    if metric == "mean_diff":
        base["value"] = base["LLM"] - base["NIST"]
    elif metric == "mae_4pt":
        base["value"] = (base["LLM"] - base["NIST"]).abs()
    elif metric == "disagree_rate":
        base["value"] = (base["LLM"] != base["NIST"]).astype(float)
    else:
        raise ValueError("Unknown METRIC. Use 'mean_diff', 'mae_4pt', or 'disagree_rate'.")

    cols = [qid_col]
    if pid_col is not None:
        cols.append(pid_col)
    cols.append("value")

    out = base[cols].rename(columns={qid_col: "qid"})
    if pid_col is not None:
        out = out.rename(columns={pid_col: "pid"})
        # Ensure unique qid-pid pairs
        out = out.drop_duplicates(subset=["qid", "pid"])

    return out


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

    rows: List[pd.DataFrame] = []
    skipped = 0

    for model, files in model_files.items():
        for f in files:
            lang = parse_lang_from_filename(f, model)
            if not want_lang(lang, LANGS):
                continue

            try:
                df = load_labels_csv(f)
                perr = per_row_metric(df, METRIC)
                if perr.empty:
                    continue

                perr["model"] = model
                perr["lang"] = lang
                perr["group"] = perr["model"] + GROUP_SEP + perr["lang"]

                keep_cols = ["group", "model", "lang", "qid", "value"]
                if "pid" in perr.columns:
                    keep_cols.insert(3, "pid")

                rows.append(perr[keep_cols])

            except Exception as e:
                skipped += 1
                print(f"[SKIP] {model} {lang} ({f.name}): {e}")

    if not rows:
        raise RuntimeError(f"No samples produced. Check LANGS={LANGS} and LABEL_ROOT={LABEL_ROOT}")

    long_df = pd.concat(rows, ignore_index=True)

    # Need >=2 samples per group
    counts = long_df["group"].value_counts()
    keep_groups = counts[counts >= 2].index
    long_df = long_df[long_df["group"].isin(keep_groups)].copy()

    if long_df["group"].nunique() < 2:
        raise RuntimeError("Not enough (model,lang) groups with >=2 samples to run Tukey.")

    write_df(long_df, OUT_SAMPLES)

    tukey = pairwise_tukeyhsd(
        endog=long_df["value"].to_numpy(),
        groups=long_df["group"].to_numpy(),
        alpha=ALPHA,
    )
    tukey_df = tukey_to_df(tukey)

    write_df(tukey_df, OUT_TUKEY_CSV)

    latex = to_latex_table(
        tukey_df,
        caption=f"Tukey HSD across all (model,lang) groups for {METRIC}, FWER={ALPHA}.",
        label=f"tab:tukey_all_models_all_langs_{safe_slug(METRIC)}",
    )
    OUT_TUKEY_TEX.write_text(latex, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 8))
    tukey.plot_simultaneous(ax=ax)

    # --- Color RAW labels ---
    for tick in ax.get_yticklabels():
        if tick.get_text().endswith(f"{GROUP_SEP}raw"):
            tick.set_color("red")

    # --- Color RAW CI bars ---
    yticks = ax.get_yticks()
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    raw_y_positions = {y for y, lab in zip(yticks, ylabels) if lab.endswith(f"{GROUP_SEP}raw")}

    for line in ax.lines:
        ydata = line.get_ydata()
        if len(ydata) > 0:
            y = float(ydata[0])
            if any(abs(y - ry) < 1e-6 for ry in raw_y_positions):
                line.set_color("red")
                line.set_linewidth(2.5)

    # --- Center axis at 0 (nice for mean_diff) ---
    xmin, xmax = ax.get_xlim()
    m = max(abs(xmin), abs(xmax))
    ax.set_xlim(-m, m)
    ax.axvline(0, linewidth=1)

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
