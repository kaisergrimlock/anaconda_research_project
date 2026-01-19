#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import tiktoken
from transformers import AutoTokenizer

# ============================================================
# Config
# ============================================================
YEAR = "2021"

# Run one or many langs at once:
LANGS: List[str] = ["eng", "ar", "fr", "zh", "vi", "he", "hi", "th", "sw", "ga"]

# If you still want the old behavior, set LANGS = ["ga_first"]
PART_MIN = 1
PART_MAX = 6

# Choose tokenizer backend
TOKENIZER_NAME = "gpt-oss-120b"  # options: "gpt-oss-120b", "meta-llama/Meta-Llama-3-8B"

# Output layout:
# - per-lang CSVs: outputs/token/gpt/passage_tokens_<YEAR>_<LANG>.csv
# - combined CSV:  outputs/token/gpt/passage_tokens_<YEAR>_ALL.csv
OUTPUT_DIR = Path("outputs") / "token" / "gpt"
WRITE_PER_LANG = True
WRITE_COMBINED = True

# ============================================================
# Tokenizer setup
# ============================================================
def make_token_counter(tokenizer_name: str):
    if tokenizer_name == "gpt-oss-120b":
        enc = tiktoken.encoding_for_model("gpt-oss-120b")

        def count_tokens(text: str) -> int:
            return len(enc.encode(text))

        def debug_tokens(text: str) -> None:
            token_ids = enc.encode(text)
            print("token_ids:", token_ids)
            print("tokens:", [enc.decode([t]) for t in token_ids])

        return count_tokens, debug_tokens

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def debug_tokens(text: str) -> None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print("token_ids:", token_ids)
        print("tokens:", tokenizer.convert_ids_to_tokens(token_ids))

    return count_tokens, debug_tokens


count_tokens, debug_tokens = make_token_counter(TOKENIZER_NAME)

# Quick sanity check
text = "hello world"
debug_tokens(text)


# ============================================================
# Data loading
# ============================================================
def load_passages_df(
    year: str,
    lang: str,
    *,
    base_dir: str | Path = "retrieved",
    part_min: int = 1,
    part_max: int = 6,
) -> pd.DataFrame:
    data_dir = Path(base_dir) / f"trec_dl_{year}" / lang
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    csv_files: List[Path] = []
    for part in range(part_min, part_max + 1):
        csv_files.extend(data_dir.glob(f"*part{part}.csv"))
    csv_files = sorted(csv_files)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        missing = [c for c in ("passage", "passage_injected") if c not in df.columns]
        if missing:
            raise ValueError(
                f"Expected columns {missing} in {csv_path}, found: {list(df.columns)}"
            )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


# ============================================================
# Processing
# ============================================================
def compute_token_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["orig_token"] = df["passage"].apply(count_tokens)
    df["inj_token"] = df["passage_injected"].apply(count_tokens)
    df["delta_token"] = df["inj_token"] - df["orig_token"]

    denom = df["orig_token"].replace(0, pd.NA)
    df["fertility_score"] = df["inj_token"] / denom

    cols = [
        "pid",
        "qid",
        "passage",
        "passage_injected",
        "orig_token",
        "inj_token",
        "delta_token",
        "fertility_score",
    ]
    # Add any missing pid/qid columns gracefully (in case some files lack them)
    for c in ("pid", "qid"):
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def run_for_langs(year: str, langs: Iterable[str]) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []

    for lang in langs:
        df = load_passages_df(
            year,
            lang,
            base_dir="retrieved",
            part_min=PART_MIN,
            part_max=PART_MAX,
        )

        out_df = compute_token_metrics(df)
        out_df.insert(0, "lang", lang)  # keep language in output

        if WRITE_PER_LANG:
            out_csv = OUTPUT_DIR / f"passage_tokens_{year}_{lang}.csv"
            out_df.to_csv(out_csv, index=False)
            print(f"[{lang}] passage_count: {len(out_df)} -> {out_csv}")

        all_frames.append(out_df)

        # Small sample to confirm things look sane
        if len(df) > 0:
            print(f"[{lang}] passage_sample:", df["passage"].iloc[0][:200])

    if not all_frames:
        raise ValueError("No languages were processed (LANGS was empty?)")

    combined = pd.concat(all_frames, ignore_index=True)

    if WRITE_COMBINED:
        combined_csv = OUTPUT_DIR / f"passage_tokens_{year}_ALL.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"[ALL] total_rows: {len(combined)} -> {combined_csv}")

    return combined


# ============================================================
# Main
# ============================================================
combined_df = run_for_langs(YEAR, LANGS)
print("done. langs:", list(dict.fromkeys(combined_df["lang"].tolist())))
