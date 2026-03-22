#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# =========================
# Path setup
# =========================
THIS_FILE = Path(__file__).resolve()
SEABORN_SCRIPT_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[3]
print("ROOT:", PROJECT_ROOT)

if str(SEABORN_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SEABORN_SCRIPT_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit

# -------- Config --------
TREC_DL_YEAR = "2022"
MODEL = "qwen3-32b-v1"
#MODEL = "gpt-oss-20b"

LANGUAGES = [
    "ru_instruct",
    "zh_instruct",
    "ga_instruct",
    "ar_instruct",
    "fr_instruct",
    "vi_instruct",
    "sw_instruct",
    "ga_instruct",
    "eng_instruct",
    "th_instruct"
]


def get_llm_file(lang: str) -> Path:
    return (
        Path("outputs/llm_label")
        / f"trec_dl_{TREC_DL_YEAR}"
        / MODEL
        / f"{MODEL}_trecdl_{TREC_DL_YEAR}_{lang}_labels.csv"
    )


def get_out_missing_file(lang: str) -> Path:
    if lang == "raw":
        return (
            Path("retrieved")
            / f"trec_dl_{TREC_DL_YEAR}"
            / "judged"
            / f"all_topics_trecdl_{TREC_DL_YEAR}_part0.csv"
        )

    return (
        Path("retrieved")
        / f"trec_dl_{TREC_DL_YEAR}"
        / lang
        / f"all_topics_trecdl_{TREC_DL_YEAR}_part0.csv"
    )


def check_missing_llm_labels(lang: str) -> pd.DataFrame:
    """
    Load the combined label CSV for one language, coerce relevance columns to numeric,
    and isolate rows where llm_relevance is missing/invalid after coercion.
    """
    bump_field_limit()

    llm_file = get_llm_file(lang)
    out_missing_file = get_out_missing_file(lang)
    out_missing_file.parent.mkdir(parents=True, exist_ok=True)

    if not llm_file.exists():
        print(f"[WARN] Missing input file for {lang}: {llm_file}")
        return pd.DataFrame()

    df = pd.read_csv(llm_file)

    if "relevance" not in df.columns or "llm_relevance" not in df.columns:
        raise ValueError(
            f"Expected columns 'relevance' and 'llm_relevance' in {llm_file}, "
            f"but got: {list(df.columns)}"
        )

    # Coerce to numeric; invalid values become NaN
    df["NIST"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["LLM"] = pd.to_numeric(df["llm_relevance"], errors="coerce")

    # Keep only rows where LLM label is missing after coercion
    missing_llm_df = df[df["LLM"].isna()].copy()

    # Drop helper columns before saving
    missing_llm_df.drop(columns=["NIST", "LLM"], errors="ignore").to_csv(
        out_missing_file,
        index=False,
        encoding="utf-8",
    )

    print(f"[{lang}] Total rows: {len(df)}")
    print(f"[{lang}] Rows with missing/invalid llm_relevance: {len(missing_llm_df)}")
    print(f"[{lang}] Saved to: {out_missing_file}")

    return missing_llm_df


def main() -> None:
    summary = []

    for lang in LANGUAGES:
        try:
            missing_df = check_missing_llm_labels(lang)
            summary.append((lang, len(missing_df)))
        except Exception as e:
            print(f"[ERROR] Failed for {lang}: {e}")
            summary.append((lang, "ERROR"))

    print("\n=== Summary ===")
    for lang, missing_count in summary:
        print(f"{lang}: {missing_count}")


if __name__ == "__main__":
    main()