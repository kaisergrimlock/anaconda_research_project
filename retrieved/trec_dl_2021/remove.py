#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path
from typing import Set, Tuple


BASE_DIR = Path(__file__).resolve().parent  # .../trec_dl_2021
JUDGED_DIR = BASE_DIR / "judged"
JUDGED_NEW_DIR = BASE_DIR / "judged_new"


def _find_col_idx(header: list[str], want: str) -> int:
    """
    Find a column index by name (case-insensitive, strip spaces).
    Raises ValueError if not found.
    """
    want_norm = want.strip().lower()
    for i, col in enumerate(header):
        if col.strip().lower() == want_norm:
            return i
    raise ValueError(f"Missing required column '{want}' in header: {header}")


def load_existing_pairs(judged_dir: Path) -> Set[Tuple[str, str]]:
    existing: Set[Tuple[str, str]] = set()

    files = sorted(judged_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {judged_dir}")

    for fp in files:
        with fp.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                continue

            qid_i = _find_col_idx(header, "qid")
            pid_i = _find_col_idx(header, "pid")

            for row in reader:
                if not row:
                    continue
                # guard against short/bad rows
                if len(row) <= max(qid_i, pid_i):
                    continue
                qid = row[qid_i]
                pid = row[pid_i]
                existing.add((qid, pid))

    return existing


def filter_file_inplace(csv_path: Path, existing: Set[Tuple[str, str]]) -> tuple[int, int]:
    """
    Overwrite csv_path, removing rows whose (qid,pid) exists in `existing`.
    Returns (kept, removed).
    """
    kept = 0
    removed = 0

    with csv_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.reader(src)
        header = next(reader, None)
        if not header:
            return (0, 0)

        qid_i = _find_col_idx(header, "qid")
        pid_i = _find_col_idx(header, "pid")

        # write to temp in same dir so replace is safe on Windows
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=str(csv_path.parent),
            prefix=csv_path.stem + "_tmp_",
            suffix=".csv",
        ) as tmp:
            tmp_path = Path(tmp.name)
            writer = csv.writer(tmp)

            writer.writerow(header)

            for row in reader:
                if not row:
                    continue
                if len(row) <= max(qid_i, pid_i):
                    # keep malformed rows (or skip; choose what you prefer)
                    writer.writerow(row)
                    kept += 1
                    continue

                key = (row[qid_i], row[pid_i])
                if key in existing:
                    removed += 1
                    continue

                writer.writerow(row)
                kept += 1

    # replace original
    tmp_path.replace(csv_path)
    return kept, removed


def main() -> None:
    if not JUDGED_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {JUDGED_DIR}")
    if not JUDGED_NEW_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {JUDGED_NEW_DIR}")

    existing = load_existing_pairs(JUDGED_DIR)
    print(f"Loaded {len(existing):,} existing (qid,pid) pairs from: {JUDGED_DIR}")

    new_files = sorted(JUDGED_NEW_DIR.glob("*.csv"))
    if not new_files:
        print(f"No CSV files found in: {JUDGED_NEW_DIR}")
        return

    total_kept = 0
    total_removed = 0

    for fp in new_files:
        kept, removed = filter_file_inplace(fp, existing)
        total_kept += kept
        total_removed += removed
        print(f"{fp.name}: kept {kept:,}, removed {removed:,}")

    print(f"\nDone. Total kept: {total_kept:,} | Total removed: {total_removed:,}")


if __name__ == "__main__":
    main()
