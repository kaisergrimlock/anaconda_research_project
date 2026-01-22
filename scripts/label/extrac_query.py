#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Iterator, Tuple

# ===== repo imports (same pattern as your scripts) =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit  # type: ignore

# ===============================================================
# Config
# ===============================================================
TREC_DL_YEAR = "2021"
START_PART = 1
END_PART = 6

# Where the *judged* part CSVs live
JUDGED_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged")
PART_PATTERN = f"all_topics_trecdl_{TREC_DL_YEAR}_part{{n}}.csv"

OUT_DIR = Path("outputs/queries") / f"trec_dl_{TREC_DL_YEAR}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TSV = OUT_DIR / f"queries_part{START_PART}-{END_PART}.tsv"

bump_field_limit()  # allow large CSV fields

# ===============================================================
# Helpers
# ===============================================================
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


def resolve_qid(row: Dict[str, str]) -> str:
    # common variants seen across TREC/MS MARCO pipelines
    return _first_nonempty(
        row,
        (
            "qid",
            "topic_id",
            "topicid",
            "query_id",
            "q_id",
        ),
    )


def resolve_query(row: Dict[str, str]) -> str:
    return _first_nonempty(
        row,
        (
            "query",
            "topic",
            "topic_query",
            "query_text",
            "title",
            "question",
            "raw_query",
            "original_query",
        ),
    )


# ===============================================================
# Main
# ===============================================================
def main() -> None:
    if not JUDGED_DIR.exists():
        raise FileNotFoundError(f"Judged folder not found: {JUDGED_DIR}")

    part_files = list(iter_part_files(JUDGED_DIR, START_PART, END_PART))
    if not part_files:
        print(f"[INFO] No judged part files found in {JUDGED_DIR} (range {START_PART}-{END_PART}).")
        return

    seen: set[str] = set()
    wrote = 0
    skipped_missing_qid = 0
    skipped_missing_query = 0
    skipped_dupe = 0

    with OUT_TSV.open("w", encoding="utf-8", newline="") as out_f:
        w = csv.writer(out_f, delimiter="\t", lineterminator="\n")
        w.writerow(["qid", "query"])

        for part_csv in part_files:
            print(f"[READ] {part_csv}")
            for row in read_rows_stream(part_csv):
                qid = resolve_qid(row)
                if not qid:
                    skipped_missing_qid += 1
                    continue

                query = resolve_query(row)
                if not query:
                    skipped_missing_query += 1
                    continue

                if qid in seen:
                    # keep first occurrence (stable, preserves earliest-seen text)
                    skipped_dupe += 1
                    continue

                seen.add(qid)
                w.writerow([qid, query])
                wrote += 1

    print(f"[DONE] wrote {wrote:,} unique queries -> {OUT_TSV}")
    if skipped_missing_qid or skipped_missing_query or skipped_dupe:
        print(
            "[WARN] skipped rows: "
            f"missing_qid={skipped_missing_qid:,}, "
            f"missing_query={skipped_missing_query:,}, "
            f"duplicate_qid={skipped_dupe:,}"
        )


if __name__ == "__main__":
    main()
