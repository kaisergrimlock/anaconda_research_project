#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable

# ===== repo imports =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_helpers import bump_field_limit


def _detect_dialect(sample: str, delimiter: str | None, quotechar: str | None) -> csv.Dialect:
    if delimiter or quotechar:
        class _D(csv.excel):
            pass

        if delimiter:
            _D.delimiter = delimiter
        if quotechar:
            _D.quotechar = quotechar
        return _D

    sniffer = csv.Sniffer()
    try:
        return sniffer.sniff(sample)
    except csv.Error:
        return csv.excel


def _sanitize_cell(value: str, replacement: str) -> str:
    if not value:
        return value
    # Normalize CRLF/CR to LF, then replace LF.
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", replacement)


def _iter_rows(
    path: Path,
    encoding: str,
    dialect: csv.Dialect,
) -> Iterable[list[str]]:
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, dialect)
        for row in reader:
            yield row


def fix_csv(
    in_path: Path,
    out_path: Path,
    *,
    input_encoding: str,
    output_encoding: str,
    dialect: csv.Dialect,
    newline_replacement: str,
) -> tuple[int, int]:
    rows = 0
    cells_changed = 0
    with out_path.open("w", encoding=output_encoding, newline="") as f_out:
        writer = csv.writer(f_out, dialect, lineterminator="\n")
        for row in _iter_rows(in_path, input_encoding, dialect):
            rows += 1
            fixed = []
            for cell in row:
                new_cell = _sanitize_cell(cell, newline_replacement)
                if new_cell != cell:
                    cells_changed += 1
                fixed.append(new_cell)
            writer.writerow(fixed)
    return rows, cells_changed


def _resolve_output_path(in_path: Path, out_path: Path | None, inplace: bool) -> Path:
    if inplace:
        return in_path.with_suffix(in_path.suffix + ".tmp")
    if out_path:
        return out_path
    return in_path.with_suffix(in_path.suffix + ".fixed.csv")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix CSV so each row is on a single line by removing newlines inside fields."
    )
    parser.add_argument("input_csv", type=Path, nargs="?")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file.")
    parser.add_argument("--delimiter", default=None, help="Override CSV delimiter.")
    parser.add_argument("--quotechar", default=None, help="Override CSV quote character.")
    parser.add_argument("--input-encoding", default="utf-8-sig")
    parser.add_argument("--output-encoding", default="utf-8")
    parser.add_argument(
        "--newline-replacement",
        default=" ",
        help="Replacement text for embedded newlines (default: single space).",
    )

    args = parser.parse_args()

    env_input = os.getenv("CSV_FIX_INPUT")
    env_input_dir = os.getenv("CSV_FIX_INPUT_DIR")
    env_output = os.getenv("CSV_FIX_OUTPUT")
    env_output_dir = os.getenv("CSV_FIX_OUTPUT_DIR")

    in_path = args.input_csv or (Path(env_input) if env_input else None)
    if in_path is None:
        print(
            "[ERROR] Missing input CSV. Provide it as an argument or set CSV_FIX_INPUT / CSV_FIX_INPUT_DIR."
        )
        return 2

    if env_input_dir and not in_path.is_absolute():
        in_path = Path(env_input_dir) / in_path

    out_arg = args.output or (Path(env_output) if env_output else None)
    if out_arg is None and env_output_dir:
        out_arg = Path(env_output_dir) / in_path.name

    if "\n" in args.newline_replacement or "\r" in args.newline_replacement:
        print("[ERROR] --newline-replacement must not contain newline characters.")
        return 2

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}")
        return 2

    bump_field_limit()

    sample = in_path.read_text(encoding=args.input_encoding, errors="ignore")[:65536]
    dialect = _detect_dialect(sample, args.delimiter, args.quotechar)

    out_path = _resolve_output_path(in_path, out_arg, args.inplace)
    rows, changed = fix_csv(
        in_path,
        out_path,
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        dialect=dialect,
        newline_replacement=args.newline_replacement,
    )

    if args.inplace:
        out_path.replace(in_path)
        out_path = in_path

    print(f"[DONE] rows={rows} cells_changed={changed} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
