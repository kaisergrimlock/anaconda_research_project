#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

# =======================================================
# CONFIG — EDIT HERE IF NEEDED
# =======================================================
# Input CSV
INPUT_CSV = Path(__file__).parent / "gpt-4o_prompt_thomas_dl22-seeds.csv"

# Output CSV
OUTPUT_CSV = Path(__file__).parent / "judged" / "all_topics_trecdl_2022_part1.csv"

# Columns to keep (renaming nist_judgment → relevance)
INPUT_COLS = {
    "id": "id",
    "query": "query",
    "passage_id": "passage_id",
    "passage": "passage",
    "nist_judgment": "relevance"
}
# =======================================================


# Handle huge fields (passages)
try:
    csv.field_size_limit(10_000_000)
except OverflowError:
    csv.field_size_limit(2_000_000)


def remap_csv(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as f_in, \
         output_path.open("w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in)

        if reader.fieldnames is None:
            raise RuntimeError("ERROR: CSV has no header row.")

        # Check all required columns exist in input
        missing = [src for src in INPUT_COLS.keys() if src not in reader.fieldnames]
        if missing:
            raise RuntimeError(
                f"ERROR: Missing columns in input CSV: {missing}\n"
                f"Found columns: {reader.fieldnames}"
            )

        # Output header uses the renamed column names
        output_header = list(INPUT_COLS.values())
        writer = csv.DictWriter(f_out, fieldnames=output_header)
        writer.writeheader()

        for row in reader:
            new_row = {
                OUTPUT: row.get(INPUT, "")
                for INPUT, OUTPUT in INPUT_COLS.items()
            }
            writer.writerow(new_row)


if __name__ == "__main__":
    if not INPUT_CSV.exists():
        raise SystemExit(f"ERROR: Input CSV does not exist: {INPUT_CSV}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    remap_csv(INPUT_CSV, OUTPUT_CSV)
    print(f"Remapped CSV written to:\n{OUTPUT_CSV}")
