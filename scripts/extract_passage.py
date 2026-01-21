#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

# ===== repo imports =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit, pick_passage_for_lang  # type: ignore

# ===============================================================
# Config (match your inference script structure)
# ===============================================================
TREC_DL_YEAR = "2022"

# Example: "raw", "vi", "hi_first", "ga_last", ...
LANG = "raw"

START_PART = 1
END_PART = 6

# Input dir logic (same as your script)
if LANG == "raw":
    PART_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged/")
else:
    PART_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/{LANG}/")

PART_PATTERN = f"all_topics_trecdl_{TREC_DL_YEAR}_part{{n}}.csv"

# Output JSONL path (one JSON object per line)
OUT_DIR = Path("outputs/passages_jsonl") / f"trec_dl_{TREC_DL_YEAR}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / f"passages_{LANG}_part{START_PART}-{END_PART}.jsonl"

bump_field_limit()  # allow large CSV fields


# ===============================================================
# Helpers
# ===============================================================
def iter_part_files(start: int, end: int) -> Iterator[Path]:
    for n in range(start, end + 1):
        p = PART_DIR / PART_PATTERN.format(n=n)
        if p.exists():
            yield p
        else:
            print(f"[WARN] Missing file: {p}")


def read_rows_stream(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            # normalize None -> ""
            yield {k: (v or "") for k, v in row.items()}


def _first_nonempty(row: Dict[str, str], keys: Tuple[str, ...]) -> str:
    for k in keys:
        v = (row.get(k, "") or "").strip()
        if v:
            return v
    return ""


def resolve_pid(row: Dict[str, str]) -> str:
    """
    Try hard to resolve the passage id to something like:
      msmarco_passage_00_172
    """
    return _first_nonempty(
        row,
        (
            "pid_resolved",
            "pid_qrels",
            "pid",
            "passage_id",
            "passageid",
            "docno",   # sometimes used in IR exports
            "docid",   # fallback (not ideal, but better than empty)
        ),
    )


def resolve_docid(row: Dict[str, str]) -> str:
    """
    Resolve the document id (MS MARCO doc id) if present.
    """
    return _first_nonempty(
        row,
        (
            "docid",
            "msmarco_docid",
            "document_id",
            "did",
            "doc_no",
            "docno",
        ),
    )


def resolve_spans(row: Dict[str, str]) -> str:
    """
    Spans formatting in your example is a STRING like:
      "(0,75)" or "(3301,3430),(3431,3598)"
    We keep it as-is if the column exists, else "".
    """
    return _first_nonempty(
        row,
        (
            "spans",
            "span",
            "passage_spans",
            "highlight_spans",
        ),
    )


# ===============================================================
# Main
# ===============================================================
def main() -> None:
    part_files = list(iter_part_files(START_PART, END_PART))
    if not part_files:
        print("[INFO] No part files found in range.")
        return

    wrote = 0
    missing_pid = 0
    missing_passage = 0

    with OUT_JSONL.open("w", encoding="utf-8", newline="\n") as out_f:
        for part_csv in part_files:
            print(f"[READ] {part_csv}")
            for row in read_rows_stream(part_csv):
                pid = resolve_pid(row)
                docid = resolve_docid(row)
                spans = resolve_spans(row)

                # Use the same passage selection logic as your labeling script.
                # For LANG != raw, this typically returns passage_injected.
                passage = (pick_passage_for_lang(row, LANG) or "").strip()

                if not pid:
                    missing_pid += 1
                    # still write? typically no—skip to avoid garbage keys
                    continue

                if not passage:
                    missing_passage += 1
                    continue

                obj = {
                    "pid": pid,
                    "passage": passage,
                    "spans": spans,
                    "docid": docid,
                }
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                wrote += 1

    print(f"[DONE] Wrote {wrote:,} JSONL lines -> {OUT_JSONL}")
    if missing_pid or missing_passage:
        print(
            f"[WARN] skipped rows: missing_pid={missing_pid:,}, missing_passage={missing_passage:,}"
        )


if __name__ == "__main__":
    main()
