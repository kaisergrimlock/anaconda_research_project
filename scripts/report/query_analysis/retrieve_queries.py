#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG (edit these)
# ============================================================
THIS_FILE = Path(__file__).resolve()

# If this file is at: scripts/report/query_analysis/<this_script>.py
# Then repo root is: THIS_FILE.parents[3]
PROJECT_ROOT = THIS_FILE.parents[3]

YEAR = "2021"

# Language folder names under: retrieved/trec_dl_<YEAR>/
LANGS = [
    "eng",
    "vi",
    "th",
    "ar",
    "he",
    "ru",
    "fr",
    "sw",
    "hi",
    "zh",
    "ga",
]

INPUT_ROOT = PROJECT_ROOT / "retrieved" / f"trec_dl_{YEAR}"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "queries" / f"queries_trecdl_{YEAR}_all_lang.csv"


# ============================================================
# Helpers
# ============================================================
def infer_query_col(df: pd.DataFrame, lang: str) -> str:
    """
    Prefer query_<lang> if present, else fall back to query.
    """
    preferred = f"query_{lang}"
    if preferred in df.columns:
        return preferred
    if "query" in df.columns:
        return "query"
    raise KeyError(
        f"Missing query column for lang='{lang}'. "
        f"Expected '{preferred}' or 'query'. Columns={list(df.columns)}"
    )


def load_lang_queries(lang: str) -> pd.DataFrame:
    """
    Reads all all_topics_trecdl_<YEAR>_part*.csv from one language folder
    and returns a df with columns: qid, <lang>
    """
    lang_dir = INPUT_ROOT / lang
    if not lang_dir.exists():
        raise FileNotFoundError(f"Missing language folder: {lang_dir}")

    part_files = sorted(lang_dir.glob(f"all_topics_trecdl_{YEAR}_part*.csv"))
    if not part_files:
        raise FileNotFoundError(f"No part CSVs found in: {lang_dir}")

    frames: list[pd.DataFrame] = []

    for fp in part_files:
        print(f"[READ] {fp}")
        df = pd.read_csv(fp)

        if "qid" not in df.columns:
            raise KeyError(f"{fp} missing 'qid' column. Columns={list(df.columns)}")

        qcol = infer_query_col(df, lang)

        tmp = df[["qid", qcol]].rename(columns={qcol: lang})
        frames.append(tmp)

    merged = pd.concat(frames, ignore_index=True)

    merged = merged.dropna(subset=["qid"])
    merged["qid"] = merged["qid"].astype(int)

    # Warn if duplicates have conflicting strings
    dups = merged[merged.duplicated("qid", keep=False)]
    if not dups.empty:
        distinct = dups.groupby("qid")[lang].nunique(dropna=True)
        conflicts = int((distinct > 1).sum())
        print(
            f"[WARN] {lang}: {dups['qid'].nunique()} duplicated qids; "
            f"{conflicts} with conflicting strings. Keeping first."
        )

    merged = merged.drop_duplicates(subset=["qid"], keep="first").sort_values("qid")
    return merged


# ============================================================
# Main
# ============================================================
def main() -> None:
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] INPUT_ROOT   = {INPUT_ROOT}")

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input root does not exist: {INPUT_ROOT}")

    out: pd.DataFrame | None = None

    for lang in LANGS:
        lang_df = load_lang_queries(lang)
        out = lang_df if out is None else out.merge(lang_df, on="qid", how="outer")

    assert out is not None

    # Stable column order
    cols = ["qid"] + [lang for lang in LANGS if lang in out.columns]
    out = out[cols].sort_values("qid")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"[DONE] wrote {len(out)} rows → {OUTPUT_CSV}")

    # Quick missingness report
    missing = out.isna().sum()
    print("[INFO] missing values per column:")
    for col in cols[1:]:
        print(f"  {col}: {int(missing[col])}")


if __name__ == "__main__":
    main()
