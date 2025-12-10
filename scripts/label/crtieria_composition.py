#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple, Any
import sys

# ========= Config you can edit =========
TREC_DL_YEAR = "2022"
MODEL        = "gpt-oss-20b"
LANG         = "eng_word"  # e.g. "raw", "eng", "vi", etc.

CRITERIA = ["contextuality", "coverage", "exactness", "topicality"]

# Relevance column name in the *criterion label files*
RELEVANCE_COL = "relevance"
# ======================================

# Decide which passage column we will OUTPUT
#   - "raw" => "passage"
#   - others => "passage_injected"
PASSAGE_COL = "passage" if LANG == "raw" else "passage_injected"

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

# Where to store the cache
CACHE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "llm_label"
    / f"trec_dl_{TREC_DL_YEAR}"
    / MODEL
    / "criteria_composed"
    / LANG
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PREFIX = f"{MODEL}_trecdl_{TREC_DL_YEAR}_{LANG}_criterion_cache"

# Type alias: (qid, pid) -> combined info
RowKey = Tuple[str, str]
RowDict = Dict[RowKey, Dict[str, Any]]


def find_file_for_criterion(criterion: str) -> Path:
    """
    Find the CSV file for a given criterion.
    Expected pattern like:
      gpt-oss-20b_trecdl_2022_vi_contextuality_labels.csv
    """
    pattern = f"*_{LANG}_{criterion}_labels.csv"
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
    data: RowDict, csv_path: Path, criterion_name: str
) -> None:
    """
    Read a single criterion CSV and merge into `data`.

    `data` maps (qid, pid) -> {
        'qid', 'pid', 'query', PASSAGE_COL,
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
            if LANG == "raw":
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
                    PASSAGE_COL: passage_val,
                }
            else:
                # If passage isn't set yet for this key, fill it
                if not data[key].get(PASSAGE_COL):
                    data[key][PASSAGE_COL] = passage_val

            # Store the score under the logical criterion name
            data[key][criterion_name] = criterion_score

            # Always take relevance directly from the criterion file;
            # if it already exists and differs, last one wins (they should match anyway).
            data[key][RELEVANCE_COL] = relevance_val


def build_combined_dict() -> RowDict:
    """
    Load all criterion files for the given LANG and return the big dictionary.
    """
    if not CRITERION_DIR.exists():
        raise FileNotFoundError(f"Criterion directory not found: {CRITERION_DIR}")

    combined: RowDict = {}

    for criterion in CRITERIA:
        csv_path = find_file_for_criterion(criterion)
        print(f"Loading {criterion} from {csv_path.name}")
        load_criterion_into_dict(combined, csv_path, criterion)

    return combined


def save_cache(data: RowDict) -> None:
    """
    Save the combined dict to chunked CSV cache files.

    We write rows as flat dicts with:
      ["qid", "pid", "query", PASSAGE_COL] + CRITERIA + [RELEVANCE_COL]
    """
    chunk_size = 500

    # Keep header order similar to your raw_crit files:
    # qid,pid,query,passage,contextuality,coverage,exactness,topicality,llm_relevance
    fieldnames = ["qid", "pid", "query", PASSAGE_COL] + CRITERIA + [RELEVANCE_COL]

    rows = list(data.values())
    total = len(rows)
    if total == 0:
        print(f"[WARN] No data to save to cache at: {CACHE_DIR}")
        return

    num_parts = (total + chunk_size - 1) // chunk_size

    for part_idx in range(num_parts):
        start = part_idx * chunk_size
        end = min(start + chunk_size, total)
        part_rows = rows[start:end]

        part_path = CACHE_DIR / f"{CACHE_PREFIX}_part{part_idx + 1:03d}.csv"
        with part_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in part_rows:
                out_row = {fn: row.get(fn, "") for fn in fieldnames}
                writer.writerow(out_row)

        print(f"  Wrote rows {start}–{end-1} to {part_path.name}")


def main() -> None:
    combined_dict = build_combined_dict()
    print(f"\nTotal (qid, pid) pairs: {len(combined_dict)}")

    # Show one example row
    if combined_dict:
        example_key = next(iter(combined_dict))
        print("\nExample entry:")
        for k, v in combined_dict[example_key].items():
            print(f"  {k}: {v}")

    # Save to cache
    save_cache(combined_dict)


if __name__ == "__main__":
    main()
