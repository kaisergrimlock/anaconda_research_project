#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Iterable, Set
import sys

sys.path.append(str(Path(__file__).parent.parent))
from helper import allow_huge_csv_fields

# ==============================
# Config (edit as needed)
# ==============================
TRECDL_YEAR = "2022"        # for folder naming only

INJECTION_YEAR = "2024"

# Text to inject at the end of each passage
INJECT_TEXT = f"Published in June {INJECTION_YEAR}."

INPUT_DIR  = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/judged")             # read these CSVs
OUTPUT_DIR = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/date_{INJECTION_YEAR}")   # write these CSVs

# Filenames pattern to process
GLOB_PATTERN = "*.csv"

# ==============================
allow_huge_csv_fields()  # Raise CSV field size limit for giant cells


# ---------- Optional: collect queries (debug/stats only) ----------
def collect_unique_queries(files: Iterable[Path]) -> Set[str]:
    unique: Set[str] = set()
    for f in files:
        with f.open("r", newline="", encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            for row in r:
                q = (row.get("query") or "").strip()
                if q:
                    unique.add(q)
    return unique


# ---------- File processing ----------
def process_file(in_path: Path, out_path: Path) -> None:
    """
    For each input row:
      - Append INJECT_TEXT to the end of 'passage'
      - Write new column 'passage_injected'
    """
    col_injected = "passage_injected"

    with in_path.open("r", newline="", encoding="utf-8") as fin, \
         out_path.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        base_fieldnames = list(reader.fieldnames or [])

        # Add passage_injected column if not present
        if col_injected not in base_fieldnames:
            base_fieldnames.append(col_injected)

        writer = csv.DictWriter(fout, fieldnames=base_fieldnames)
        writer.writeheader()

        for row in reader:
            p = (row.get("passage", "") or "")

            # Append the injected text once at the end
            if INJECT_TEXT:
                if p.strip():
                    p_inj = p.rstrip() + " " + INJECT_TEXT
                else:
                    p_inj = INJECT_TEXT
            else:
                p_inj = p

            row[col_injected] = p_inj

            # Ensure all keys in row are valid and not None
            valid_row = {
                k: ("" if v is None else v)
                for k, v in row.items()
                if k in base_fieldnames
            }
            writer.writerow(valid_row)


def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
    if not files:
        raise SystemExit(f"No CSV files found in: {INPUT_DIR} (pattern: {GLOB_PATTERN})")

    print(f"Scanning {len(files)} file(s) for unique queries (optional)...")
    unique_queries = collect_unique_queries(files)
    print(f"Unique queries found: {len(unique_queries)}")

    print(f"\nProcessing {len(files)} file(s) from {INPUT_DIR}")
    print(f"Writing outputs to {OUTPUT_DIR}\n")

    for i, in_path in enumerate(files, 1):
        out_path = OUTPUT_DIR / in_path.name
        print(f"[{i}/{len(files)}] {in_path.name} -> {out_path.name}")
        process_file(in_path, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
