#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple, Any
import sys

# ========= Config you can edit =========
TREC_DL_YEAR = "2022"
MODEL        = "llama3-8b-instruct" # e.g. "qwen3-32b-v1", "gpt-oss-20b", etc.
LANGS        = ["raw", "eng", "ru", "fr", "ar", "zh", "vi", "hi", "he", "th", "sw", "ga"]  # e.g. ["raw", "eng", "vi"], etc.

CRITERIA = ["contextuality", "coverage", "exactness", "topicality"]

# Relevance column name in the *criterion label files*
RELEVANCE_COL = "relevance"
# ======================================

# If this script is in scripts/ or similar, adjust depth accordingly
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional: if you have bump_field_limit helper, you can use it
# from scripts.csv_helpers import bump_field_limit
# bump_field_limit()

CRITERION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "llm_label"
    / f"trec_dl_{TREC_DL_YEAR}"
    / MODEL
    / "criterion"
)

def cache_dir_for_lang(lang: str) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "llm_label"
        / f"trec_dl_{TREC_DL_YEAR}"
        / MODEL
        / "criteria_composed"
        / lang
    )


def cache_prefix_for_lang(lang: str) -> str:
    return f"{MODEL}_trecdl_{TREC_DL_YEAR}_{lang}_criterion_cache"

# Type alias: (qid, pid) -> combined info
RowKey = Tuple[str, str]
RowDict = Dict[RowKey, Dict[str, Any]]


def find_file_for_criterion(criterion: str, lang: str) -> Path:
    """
    Find the CSV file for a given criterion.
    Expected pattern like:
      gpt-oss-20b_trecdl_2022_vi_contextuality_labels.csv
    """
    pattern = f"*_{lang}_{criterion}_labels.csv"
    matches = list(CRITERION_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching {pattern} in {CRITERION_DIR}"
        )
    if len(matches) > 1:
        # If there are multiple, you can change this to handle differently
        raise RuntimeError(
            f"Multiple files found for criterion '{criterion}': {matches}"
        )
    return matches[0]


def load_criterion_into_dict(
    data: RowDict, csv_path: Path, criterion_name: str, lang: str, passage_col: str
) -> None:
    """
    Read a single criterion CSV and merge into `data`.

    `data` maps (qid, pid) -> {
        'qid', 'pid', 'query', passage_col,
        'contextuality', 'coverage', 'exactness', 'topicality',
        RELEVANCE_COL
    }

    Relevance is taken *directly* from the criterion files (RELEVANCE_COL).
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return

        # Require that criterion files contain the relevance column
        if RELEVANCE_COL not in fieldnames:
            raise KeyError(
                f"Expected relevance column '{RELEVANCE_COL}' not found in {csv_path.name}. "
                f"Available columns: {fieldnames}"
            )

        # Assume the last column is the score for this specific criterion
        last_col_name = fieldnames[-1]

        for row in reader:
            qid = row.get("qid", "")
            pid = row.get("pid", "")  # adjust if your column is named differently
            query = row.get("query", "")

            # Decide which passage column to READ from this file:
            #   - For raw: prefer 'passage', fallback to 'passage_injected'
            #   - For others: prefer 'passage_injected', fallback to 'passage'
            if lang == "raw":
                passage_val = row.get("passage", "")
                if not passage_val:
                    passage_val = row.get("passage_injected", "")
            else:
                passage_val = row.get("passage_injected", "")
                if not passage_val:
                    passage_val = row.get("passage", "")

            criterion_score = row.get(last_col_name, "")
            relevance_val   = row.get(RELEVANCE_COL, "")

            key: RowKey = (qid, pid)

            if key not in data:
                data[key] = {
                    "qid": qid,
                    "pid": pid,
                    "query": query,
                    passage_col: passage_val,
                }
            else:
                # If passage isn't set yet for this key, fill it
                if not data[key].get(passage_col):
                    data[key][passage_col] = passage_val

            # Store the score under the logical criterion name
            data[key][criterion_name] = criterion_score

            # Always take relevance directly from the criterion file;
            # if it already exists and differs, last one wins (they should match anyway).
            data[key][RELEVANCE_COL] = relevance_val


def build_combined_dict(lang: str, passage_col: str) -> RowDict:
    """
    Load all criterion files for the given LANG and return the big dictionary.
    """
    if not CRITERION_DIR.exists():
        raise FileNotFoundError(f"Criterion directory not found: {CRITERION_DIR}")

    combined: RowDict = {}

    for criterion in CRITERIA:
        csv_path = find_file_for_criterion(criterion, lang)
        print(f"Loading {criterion} from {csv_path.name}")
        load_criterion_into_dict(combined, csv_path, criterion, lang, passage_col)

    return combined


def save_cache(data: RowDict, lang: str, passage_col: str) -> None:
    """
    Save the combined dict to chunked CSV cache files.

    We write rows as flat dicts with:
      ["qid", "pid", "query", passage_col] + CRITERIA + [RELEVANCE_COL]
    """
    chunk_size = 500

    # Keep header order similar to your raw_crit files:
    # qid,pid,query,passage,contextuality,coverage,exactness,topicality,llm_relevance
    fieldnames = ["qid", "pid", "query", passage_col] + CRITERIA + [RELEVANCE_COL]

    rows = list(data.values())
    total = len(rows)
    if total == 0:
        cache_dir = cache_dir_for_lang(lang)
        print(f"[WARN] No data to save to cache at: {cache_dir}")
        return

    cache_dir = cache_dir_for_lang(lang)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prefix = cache_prefix_for_lang(lang)
    num_parts = (total + chunk_size - 1) // chunk_size

    for part_idx in range(num_parts):
        start = part_idx * chunk_size
        end = min(start + chunk_size, total)
        part_rows = rows[start:end]

        part_path = cache_dir / f"{cache_prefix}_part{part_idx + 1:03d}.csv"
        with part_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in part_rows:
                out_row = {fn: row.get(fn, "") for fn in fieldnames}
                for criterion in CRITERIA:
                    if not out_row.get(criterion):
                        out_row[criterion] = "NaN"
                writer.writerow(out_row)

        print(f"  Wrote rows {start}–{end-1} to {part_path.name}")


def main() -> None:
    if not LANGS:
        raise ValueError("LANGS must contain at least one language.")

    for lang in LANGS:
        # Decide which passage column we will OUTPUT
        #   - "raw" => "passage"
        #   - others => "passage_injected"
        passage_col = "passage" if lang == "raw" else "passage_injected"

        print(f"\n=== Processing language: {lang} ===")
        combined_dict = build_combined_dict(lang, passage_col)
        print(f"\nTotal (qid, pid) pairs: {len(combined_dict)}")

        # Show one example row
        if combined_dict:
            example_key = next(iter(combined_dict))
            print("\nExample entry:")
            for k, v in combined_dict[example_key].items():
                print(f"  {k}: {v}")

        # Save to cache
        save_cache(combined_dict, lang, passage_col)


if __name__ == "__main__":
    main()
