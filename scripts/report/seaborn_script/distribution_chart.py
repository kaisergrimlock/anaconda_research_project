#!/usr/bin/env python3
"""
Count docs per (label, collection) and write CSV summary.

Supports:
- CSV/TSV (auto-detects delimiter). Expects columns: `label`, `collection`.
- JSONL (one JSON object per line) with keys: `label`, `collection`.

Usage:
  python make_label_counts.py INPUT_FILE --outdir path/to/output [--outfile label_counts.csv]
  # If your columns have different names, specify them:
  python make_label_counts.py data.csv --outdir out --label-col rel --collection-col source
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

def iter_rows_from_csv(path, label_col, collection_col):
    with open(path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            # Fallback: assume comma
            dialect = csv.get_dialect("excel")
        reader = csv.DictReader(f, dialect=dialect)
        for i, row in enumerate(reader, 1):
            if label_col not in row or collection_col not in row:
                raise KeyError(
                    f"Missing required columns in CSV (need '{label_col}' and '{collection_col}'). "
                    f"Available: {list(row.keys())}"
                )
            yield str(row[label_col]).strip(), str(row[collection_col]).strip()

def iter_rows_from_jsonl(path, label_col, collection_col):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i} is not valid JSON: {e}")
            if label_col not in obj or collection_col not in obj:
                raise KeyError(
                    f"Missing keys at line {i} (need '{label_col}' and '{collection_col}'). "
                    f"Available: {list(obj.keys())}"
                )
            yield str(obj[label_col]).strip(), str(obj[collection_col]).strip()

def main():
    ap = argparse.ArgumentParser(description="Create CSV: label,no_of_docs,collection")
    ap.add_argument("input_file", help="Path to input CSV/TSV or JSONL")
    ap.add_argument("--outdir", required=True, help="Directory to write the output CSV")
    ap.add_argument("--outfile", default="label_counts.csv", help="Output filename (default: label_counts.csv)")
    ap.add_argument("--label-col", default="label", help="Column/key name for label (default: label)")
    ap.add_argument("--collection-col", default="collection", help="Column/key name for collection (default: collection)")
    args = ap.parse_args()

    inp = Path(args.input_file)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    # Choose reader based on extension
    ext = inp.suffix.lower()
    if ext in {".csv", ".tsv"}:
        row_iter = iter_rows_from_csv(inp, args.label_col, args.collection_col)
    elif ext in {".jsonl", ".ndjson"}:
        row_iter = iter_rows_from_jsonl(inp, args.label_col, args.collection_col)
    else:
        # Try CSV first; if it fails hard, suggest JSONL
        try:
            row_iter = iter_rows_from_csv(inp, args.label_col, args.collection_col)
        except Exception:
            row_iter = iter_rows_from_jsonl(inp, args.label_col, args.collection_col)

    counts = Counter()
    total = 0
    for label, collection in row_iter:
        counts[(label, collection)] += 1
        total += 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / args.outfile

    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "no. of docs", "collection"])
        for (label, collection), cnt in sorted(counts.items(), key=lambda x: (str(x[0][1]), str(x[0][0]))):
            writer.writerow([label, cnt, collection])

    print(f"Wrote {len(counts)} rows (from {total} docs) to: {outpath}")

if __name__ == "__main__":
    main()
