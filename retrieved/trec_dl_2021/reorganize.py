#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Dict, Tuple

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent          # .../retrieved/trec_dl_2021
IN_DIR   = BASE_DIR / "judged_new"
OUT_DIR  = BASE_DIR / "judged_new"                 # overwrite in-place
LINES_PER_FILE = 500                               # data rows per file (excluding header)

# Naming:
# If input files look like all_topics_trecdl_2021_part5.csv ... part7.csv
# this script will rewrite them as contiguous parts starting at START_PART.
PREFIX = "all_topics_trecdl_2021_part"
START_PART = 5                                     # change if you want to start from 0/1/etc


def read_all_rows(csv_paths: List[Path]) -> Tuple[List[str], List[List[str]]]:
    header: List[str] | None = None
    rows: List[List[str]] = []

    for fp in csv_paths:
        with fp.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            h = next(r, None)
            if not h:
                continue
            if header is None:
                header = h
            else:
                # Basic sanity: same columns count
                if len(h) != len(header):
                    raise ValueError(f"Header mismatch in {fp.name}.\nExpected: {header}\nGot: {h}")

            for row in r:
                if row:
                    rows.append(row)

    if header is None:
        raise FileNotFoundError(f"No readable CSVs in {IN_DIR}")

    return header, rows


def write_chunked(header: List[str], rows: List[List[str]]) -> List[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old CSVs in judged_new (only those matching our prefix OR all csvs, your call)
    # Here: remove ALL csvs in judged_new to avoid mixing old/new parts.
    for old in OUT_DIR.glob("*.csv"):
        old.unlink()

    total = len(rows)
    if total == 0:
        return []

    num_files = math.ceil(total / LINES_PER_FILE)
    written: List[Path] = []

    for i in range(num_files):
        part_no = START_PART + i
        out_fp = OUT_DIR / f"{PREFIX}{part_no}.csv"
        start = i * LINES_PER_FILE
        end = min(start + LINES_PER_FILE, total)

        with out_fp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows[start:end])

        written.append(out_fp)

    return written


def main() -> None:
    csv_paths = sorted(IN_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in: {IN_DIR}")
        return

    header, rows = read_all_rows(csv_paths)
    out_files = write_chunked(header, rows)

    print(f"Read {len(rows):,} rows from {len(csv_paths)} file(s) in {IN_DIR}")
    if not out_files:
        print("No output written (0 rows).")
        return

    print(f"Wrote {len(out_files)} file(s) with up to {LINES_PER_FILE} rows each:")
    for fp in out_files:
        print(f"  - {fp.name}")


if __name__ == "__main__":
    main()
