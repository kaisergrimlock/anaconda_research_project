#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

# =======================================================
# CONFIG — EDIT HERE IF NEEDED
# =======================================================
# Input CSV
INPUT_CSV = Path(__file__).parent / "gpt-4o_prompt_thomas_dl21-seeds.csv"

# Output directory + base name (parts will be suffixed with 1, 2, 3, ...)
OUTPUT_DIR      = Path(__file__).parent / "trec_dl_2021" / "judged"
OUTPUT_BASENAME = "all_topics_trecdl_2021_part"

# How many rows (after header) per output CSV
ROWS_PER_FILE = 500

# Columns to keep (renaming nist_judgment → relevance)
INPUT_COLS = {
    "id": "qid",
    "query": "query",
    "passage_id": "pid",
    "passage": "passage",
    "nist_judgment": "relevance",
}
# =======================================================

# Handle huge fields (passages)
try:
    csv.field_size_limit(10_000_000)
except OverflowError:
    csv.field_size_limit(2_000_000)


def remap_and_split_csv(
    input_path: Path,
    output_dir: Path,
    basename: str,
    rows_per_file: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as f_in:
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

        output_header = list(INPUT_COLS.values())

        part_idx = 1
        rows_in_current_part = 0
        writer = None
        current_output_path: Path | None = None
        opened_file = None  # track the file handle so we can close it

        def open_new_part(nonlocal_part_idx: int):
            """Helper to open a new part file and return (file_handle, writer, path)."""
            out_path = output_dir / f"{basename}{nonlocal_part_idx}.csv"
            f_out = out_path.open("w", encoding="utf-8", newline="")
            w = csv.DictWriter(f_out, fieldnames=output_header)
            w.writeheader()
            return f_out, w, out_path

        # Open first part
        opened_file, writer, current_output_path = open_new_part(part_idx)

        try:
            for row in reader:
                # If we hit the limit, rotate to a new file
                if rows_in_current_part >= rows_per_file:
                    opened_file.close()
                    part_idx += 1
                    rows_in_current_part = 0
                    opened_file, writer, current_output_path = open_new_part(part_idx)

                new_row = {
                    OUTPUT: row.get(INPUT, "")
                    for INPUT, OUTPUT in INPUT_COLS.items()
                }
                writer.writerow(new_row)
                rows_in_current_part += 1
        finally:
            if opened_file is not None and not opened_file.closed:
                opened_file.close()

        print(f"Finished writing {part_idx} part file(s) to {output_dir}")


if __name__ == "__main__":
    if not INPUT_CSV.exists():
        raise SystemExit(f"ERROR: Input CSV does not exist: {INPUT_CSV}")

    remap_and_split_csv(
        INPUT_CSV,
        OUTPUT_DIR,
        OUTPUT_BASENAME,
        ROWS_PER_FILE,
    )
