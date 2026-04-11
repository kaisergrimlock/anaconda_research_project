#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# =========================
# Path setup
# =========================
THIS_FILE = Path(__file__).resolve()
LABEL_SCRIPT_DIR = THIS_FILE.parent
PROJECT_ROOT = THIS_FILE.parents[2]
print("ROOT:", PROJECT_ROOT)

if str(LABEL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(LABEL_SCRIPT_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit

# -------- Config --------
TREC_DL_YEAR = "2021"
MODEL = "gpt-oss-20b"

LANGUAGES = [
    "eng",
    "eng_instruct"
]

def get_alignment_file(lang: str) -> Path:
    return (
        Path("outputs/alignment_checker")
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

def check_missing_alignment_scores(lang: str) -> pd.DataFrame:
    """
    Load the combined label CSV for one language, coerce alignment_score to numeric,
    and isolate rows where alignment_score is missing/invalid after coercion.
    """
    bump_field_limit()

    align_file = get_alignment_file(lang)
    out_missing_file = get_out_missing_file(lang)
    out_missing_file.parent.mkdir(parents=True, exist_ok=True)

    if not align_file.exists():
        print(f"[WARN] Missing input file for {lang}: {align_file}")
        return pd.DataFrame()

    df = pd.read_csv(align_file)

    if "alignment_score" not in df.columns:
        raise ValueError(
            f"Expected column 'alignment_score' in {align_file}, "
            f"but got: {list(df.columns)}"
        )

    # Coerce to numeric; invalid values become NaN
    df["ALIGNMENT"] = pd.to_numeric(df["alignment_score"], errors="coerce")

    # Keep only rows where alignment_score is missing after coercion
    missing_df = df[df["ALIGNMENT"].isna()].copy()

    # Drop helper columns before saving
    missing_df.drop(columns=["ALIGNMENT"], errors="ignore").to_csv(
        out_missing_file,
        index=False,
        encoding="utf-8",
    )

    print(f"[{lang}] Total rows: {len(df)}")
    print(f"[{lang}] Rows with missing/invalid alignment_score: {len(missing_df)}")
    print(f"[{lang}] Saved to: {out_missing_file}")

    return missing_df

def main() -> None:
    summary = []

    for lang in LANGUAGES:
        try:
            missing_df = check_missing_alignment_scores(lang)
            if not missing_df.empty or get_alignment_file(lang).exists():
                summary.append((lang, len(missing_df)))
            else:
                summary.append((lang, "FILE NOT FOUND"))
        except Exception as e:
            print(f"[ERROR] Failed for {lang}: {e}")
            summary.append((lang, "ERROR"))

    print("\n=== Summary ===")
    for lang, missing_count in summary:
        print(f"{lang}: {missing_count}")

if __name__ == "__main__":
    main()
