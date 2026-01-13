#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
from transformers import AutoTokenizer

# ============================================================
# Config
# ============================================================
YEAR = "2022"

# Run one or many langs at once:
LANGS: List[str] = [
    "he_first", "ar_first", "ru_first", "eng_first", "vi_first", "th_first", "fr_first", "sw_first", "ga_first"
]

PART_MIN = 1
PART_MAX = 6

OUTPUT_DIR = Path("outputs") / "token" / "qwen"
WRITE_PER_LANG = True
WRITE_COMBINED = True  # writes passage_tokens_<YEAR>_ALL.csv with a 'lang' column

TOKENIZER_NAME = "Qwen/Qwen3-32B"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)


# ============================================================
# Token helpers
# ============================================================
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def debug_tokens(text: str) -> None:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print("token_ids:", token_ids)
    print("tokens:", tokenizer.convert_ids_to_tokens(token_ids))


# ============================================================
# Data loading
# ============================================================
def load_passages_df(year: str, lang: str, base_dir: str | Path = "retrieved") -> pd.DataFrame:
    data_dir = Path(base_dir) / f"trec_dl_{year}" / lang
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    csv_files: list[Path] = []
    for part in range(PART_MIN, PART_MAX + 1):
        csv_files.extend(data_dir.glob(f"*part{part}.csv"))
    csv_files = sorted(csv_files)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        missing = [col for col in ("passage", "passage_injected") if col not in df.columns]
        if missing:
            raise ValueError(f"Expected columns {missing} in {csv_path}, found: {list(df.columns)}")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ============================================================
# Compute + write
# ============================================================
def compute_token_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure pid/qid exist (some files might not have them)
    for c in ("pid", "qid"):
        if c not in df.columns:
            df[c] = ""

    df["orig_token"] = df["passage"].apply(count_tokens)
    df["inj_token"] = df["passage_injected"].apply(count_tokens)
    df["delta_token"] = df["inj_token"] - df["orig_token"]
    df["fertility_score"] = df["inj_token"] / df["orig_token"].replace(0, pd.NA)

    return df[
        [
            "pid",
            "qid",
            "passage",
            "passage_injected",
            "orig_token",
            "inj_token",
            "delta_token",
            "fertility_score",
        ]
    ]


def run_for_langs(year: str, langs: Iterable[str]) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # quick tokenizer sanity
    debug_tokens("hello world")

    all_frames: list[pd.DataFrame] = []

    for lang in langs:
        df = load_passages_df(year, lang)
        out_df = compute_token_df(df)
        out_df.insert(0, "lang", lang)

        if WRITE_PER_LANG:
            out_csv = OUTPUT_DIR / f"passage_tokens_{year}_{lang}.csv"
            out_df.to_csv(out_csv, index=False)
            print(f"[{lang}] passage_count: {len(out_df)} -> {out_csv}")

        if len(df) > 0:
            print(f"[{lang}] passage_sample:", df["passage"].iloc[0][:200])

        all_frames.append(out_df)

    if not all_frames:
        raise ValueError("No languages were processed (LANGS is empty or all missing).")

    combined = pd.concat(all_frames, ignore_index=True)

    if WRITE_COMBINED:
        combined_csv = OUTPUT_DIR / f"passage_tokens_{year}_ALL.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"[ALL] total_rows: {len(combined)} -> {combined_csv}")

    return combined


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    run_for_langs(YEAR, LANGS)
