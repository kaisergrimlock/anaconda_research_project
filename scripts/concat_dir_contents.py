#!/usr/bin/env python
"""
Concatenate contents of all files under a directory into one output file.

Usage:
  python scripts/concat_dir_contents.py <input_dir> <output_file>
  python scripts/concat_dir_contents.py <input_dir> <output_file> --no-recursive
"""

from __future__ import annotations

import os
from pathlib import Path

# Set these to override environment variables.
TEMP = r"outputs\llm_label\trec_dl_2021\gpt-oss-20b\criterion\_tmp_20260118_003008_openai.gpt-oss-20b-1_0_sw"
DEST = r"outputs\llm_label\trec_dl_2021\gpt-oss-20b\criterion\gpt-oss-20b_trecdl_2021_sw_contextuality_labels"

def iter_files(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file()]
    else:
        files = [p for p in root.iterdir() if p.is_file()]
    return sorted(files, key=lambda p: str(p))


def main() -> int:
    input_dir_value = TEMP or os.getenv("TEMP")
    output_file_value = DEST or os.getenv("DEST")

    if not input_dir_value or not output_file_value:
        raise SystemExit("Set TEMP/DEST in this file or via environment variables.")

    input_dir = Path(input_dir_value)
    output_file = Path(output_file_value)
    recursive = True

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_resolved = output_file.resolve()
    files = [p for p in iter_files(input_dir, recursive) if p.resolve() != output_resolved]

    with output_file.open("w", encoding="utf-8", errors="replace", newline="\n") as out_f:
        for idx, path in enumerate(files):
            header = f"===== {path} ====="
            out_f.write(header + "\n")
            with path.open("r", encoding="utf-8", errors="replace") as in_f:
                out_f.write(in_f.read())
            if idx != len(files) - 1:
                out_f.write("\n\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
