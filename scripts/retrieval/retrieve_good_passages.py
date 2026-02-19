#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd

# =========================
# CONFIG
# =========================
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent

YEAR = "2021"  # <-- set year here

# If your script sits at repo root, keep this:
PROJECT_ROOT = THIS_FILE.parents[2]
# If your script is inside a subfolder (e.g. scripts/report/...), change to e.g.:
# PROJECT_ROOT = SCRIPT_DIR.parents[2]

GOOD_PASSAGE_CSV = SCRIPT_DIR / "good_passage.csv"
INPUT_ROOT = PROJECT_ROOT / "retrieved" / f"trec_dl_{YEAR}"

OUT_DIR = SCRIPT_DIR / "outputs" / f"trec_dl_{YEAR}"
OUT_CSV = OUT_DIR / f"qid_slice_injected_only_{YEAR}.csv"

# Only these languages will be read (must match folder names under retrieved/trec_dl_{YEAR}/)
# Example: ["eng", "ar", "ar_word", "vi", "vi_word", ...]
LANGS: List[str] = [
    "eng", "vi", "th" , "ar", "ru", "fr", "he", "zh", "hi", "sw"
]

# =========================
# Helpers
# =========================
def bump_field_limit() -> None:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            return
        except OverflowError:
            max_int = int(max_int / 10)


def load_qids(path: Path) -> Set[str]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "qid" not in df.columns:
        raise ValueError(f"{path} must contain a 'qid' column. Found: {list(df.columns)}")
    qids = set(df["qid"].astype(str).str.strip())
    qids.discard("")
    return qids


def list_csv_files(lang_dir: Path) -> List[Path]:
    return sorted([p for p in lang_dir.rglob("*.csv") if p.is_file()])


# =========================
# Main
# =========================
def main() -> None:
    bump_field_limit()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qids = load_qids(GOOD_PASSAGE_CSV)
    print(f"[info] Loaded {len(qids)} qids from: {GOOD_PASSAGE_CSV}")

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT}")

    # Validate language folders exist (warn but continue)
    missing_langs = [l for l in LANGS if not (INPUT_ROOT / l).is_dir()]
    if missing_langs:
        print(f"[warn] These language folders were not found under {INPUT_ROOT}: {missing_langs}")

    all_frames: List[pd.DataFrame] = []
    used_files = 0

    # We only require these columns now (since we only want injected passages)
    required_cols = ["qid", "query", "pid", "passage_injected", "relevance"]

    for lang in LANGS:
        lang_dir = INPUT_ROOT / lang
        if not lang_dir.is_dir():
            continue

        files = list_csv_files(lang_dir)
        if not files:
            continue

        for fp in files:
            try:
                df = pd.read_csv(fp, dtype=str, keep_default_na=False)
            except Exception as e:
                print(f"[warn] Cannot read {fp}: {e}")
                continue

            # Ensure required columns exist
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"[warn] Skip {fp} (lang={lang}) missing columns: {missing}")
                continue

            # Filter qids
            df["qid"] = df["qid"].astype(str).str.strip()
            df = df[df["qid"].isin(qids)]
            if df.empty:
                continue

            # Output only injected passage (rename it to "passage" if you prefer)
            out_cols = ["qid", "pid", "query", "query_eng", "passage_injected", "relevance"]
            for c in out_cols:
                if c not in df.columns:
                    df[c] = ""

            df = df[out_cols].copy()
            df.insert(1, "lang", lang)
            df.insert(0, "year", YEAR)

            # Optional: keep provenance
            df["source_file"] = str(fp.relative_to(PROJECT_ROOT)).replace("\\", "/")

            # Optional rename so downstream scripts can always use "passage"
            df = df.rename(columns={"passage_injected": "passage"})

            all_frames.append(df)
            used_files += 1

    if not all_frames:
        raise RuntimeError(
            "No matching rows found.\n"
            "Check:\n"
            f"  - YEAR={YEAR}\n"
            f"  - INPUT_ROOT={INPUT_ROOT}\n"
            f"  - LANGS={LANGS}\n"
            "  - qids in good_passage.csv actually exist in per-language CSVs\n"
            "  - per-language CSVs contain 'passage_injected'\n"
        )

    out_df = pd.concat(all_frames, ignore_index=True)
    out_df = out_df.sort_values(["qid", "lang", "pid", "source_file"], kind="stable")

    out_df.to_csv(OUT_CSV, index=False)
    print(f"[done] Used {used_files} files containing at least one matching qid")
    print(f"[done] Wrote {len(out_df)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
