#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

# ===== repo imports (same pattern as your scripts) =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit  # type: ignore


# ===============================================================
# Config
# ===============================================================
TREC_DL_YEAR = "2022"
START_PART = 1
END_PART = 6

SCRIPT_DIR = Path(__file__).resolve().parent
SUFFIX_CSV = SCRIPT_DIR / "suffix.csv"

PART_PATTERN = f"all_topics_trecdl_{TREC_DL_YEAR}_part{{n}}.csv"

OUT_DIR = Path("outputs/passages_jsonl") / f"trec_dl_{TREC_DL_YEAR}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Combined output (all folders merged)
COMBINED_JSONL = OUT_DIR / f"passages_combined_part{START_PART}-{END_PART}.jsonl"

bump_field_limit()  # allow large CSV fields


# ===============================================================
# Helpers
# ===============================================================
def read_suffix_map(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing suffix file: {path}")
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for i, row in enumerate(reader, start=1):
            folder = (row.get("folder", "") or "").strip()
            suffix = (row.get("suffix", "") or "").strip()
            if not folder or not suffix:
                raise ValueError(f"Bad row {i} in {path}: expected 'folder' and 'suffix' values.")
            pairs.append((folder, suffix))
    return pairs


def iter_part_files(part_dir: Path, start: int, end: int) -> Iterator[Path]:
    for n in range(start, end + 1):
        p = part_dir / PART_PATTERN.format(n=n)
        if p.exists():
            yield p
        else:
            print(f"[WARN] Missing file: {p}")


def read_rows_stream(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            yield {k: (v or "") for k, v in row.items()}


def _first_nonempty(row: Dict[str, str], keys: Tuple[str, ...]) -> str:
    for k in keys:
        v = (row.get(k, "") or "").strip()
        if v:
            return v
    return ""


def resolve_pid(row: Dict[str, str]) -> str:
    return _first_nonempty(
        row,
        (
            "pid_resolved",
            "pid_qrels",
            "pid",
            "passage_id",
            "passageid",
            "docno",
        ),
    )


def resolve_docid(row: Dict[str, str]) -> str:
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
    return _first_nonempty(
        row,
        (
            "spans",
            "span",
            "passage_spans",
            "highlight_spans",
        ),
    )


def pick_passage(row: Dict[str, str]) -> str:
    """
    Prefer injected passage if present, otherwise fall back to raw passage.
    """
    p = (row.get("passage_injected", "") or "").strip()
    if p:
        return p
    return (row.get("passage", "") or "").strip()


def safe_suffix_for_filename(s: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    return "".join(ch if ch in allowed else "_" for ch in s)


# ===============================================================
# Main
# ===============================================================
def main() -> None:
    pairs = read_suffix_map(SUFFIX_CSV)
    if not pairs:
        print("[INFO] No suffix entries found.")
        return

    # Open combined output once; append all objects as JSONL
    combined_wrote = 0
    with COMBINED_JSONL.open("w", encoding="utf-8", newline="\n") as combined_f:
        for folder, suffix in pairs:
            part_dir = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/{folder}/")

            suffix_safe = safe_suffix_for_filename(suffix)
            out_jsonl = OUT_DIR / f"passages_{folder}{suffix_safe}_part{START_PART}-{END_PART}.jsonl"

            part_files = list(iter_part_files(part_dir, START_PART, END_PART))
            if not part_files:
                print(f"[INFO] No part files for folder='{folder}' in {part_dir} (range {START_PART}-{END_PART}).")
                continue

            wrote = 0
            skipped_missing_pid = 0
            skipped_missing_passage = 0

            with out_jsonl.open("w", encoding="utf-8", newline="\n") as out_f:
                for part_csv in part_files:
                    print(f"[READ] folder='{folder}' suffix='{suffix}' -> {part_csv}")
                    for row in read_rows_stream(part_csv):
                        pid = resolve_pid(row)
                        if not pid:
                            skipped_missing_pid += 1
                            continue

                        passage = pick_passage(row)
                        if not passage:
                            skipped_missing_passage += 1
                            continue

                        docid = resolve_docid(row)
                        spans = resolve_spans(row)

                        # Append pid with suffix in output
                        pid_out = f"{pid}{suffix}"

                        obj = {
                            "pid": pid_out,
                            "passage": passage,
                            "spans": spans,
                            "docid": docid,
                        }

                        line = json.dumps(obj, ensure_ascii=False) + "\n"
                        out_f.write(line)

                        # Also write to combined file
                        combined_f.write(line)
                        combined_wrote += 1
                        wrote += 1

            print(f"[DONE] folder='{folder}' | wrote {wrote:,} -> {out_jsonl}")
            if skipped_missing_pid or skipped_missing_passage:
                print(
                    f"[WARN] skipped rows: missing_pid={skipped_missing_pid:,}, "
                    f"missing_passage={skipped_missing_passage:,}"
                )

    print(f"[COMBINED] wrote {combined_wrote:,} -> {COMBINED_JSONL}")


if __name__ == "__main__":
    main()
