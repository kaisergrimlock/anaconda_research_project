#!/usr/bin/env python3
"""
Read a CSV with columns: query, docid, passage, relevance
Count docs per label (from the 'relevance' column) and write:
  label,no. of docs,collection
to an output CSV in a specified folder.

Usage:
  python label_counts_csv.py INPUT.csv --outdir out --collection "TREC-DL 2019" \
         [--label-col relevance] [--outfile label_counts.csv]
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file", help="Path to input CSV")
    ap.add_argument("--outdir", required=True, help="Directory to write output CSV")
    ap.add_argument("--collection", required=True, help="Collection name to record in output")
    ap.add_argument("--label-col", default="relevance", help="Label column name (default: relevance)")
    ap.add_argument("--outfile", default="label_counts.csv", help="Output filename (default: label_counts.csv)")
    args = ap.parse_args()

    inp = Path(args.input_file)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    # Try to detect delimiter (comma/tsv)
    with open(inp, "r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.get_dialect("excel")
        reader = csv.DictReader(f, dialect=dialect)

        if args.label_col not in reader.fieldnames:
            raise KeyError(
                f"Column '{args.label_col}' not found. Available: {reader.fieldnames}"
            )

        counts = Counter()
        total_rows = 0
        for row in reader:
            label = str(row[args.label_col]).strip()
            counts[label] += 1
            total_rows += 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / args.outfile

    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "no. of docs", "collection"])
        # sort by numeric label if possible, else lexicographic
        def sort_key(k):
            try:
                return (0, int(k))
            except ValueError:
                return (1, k)
        for label in sorted(counts.keys(), key=sort_key):
            w.writerow([label, counts[label], args.collection])

    print(f"Wrote {len(counts)} rows (from {total_rows} docs) to: {outpath}")

if __name__ == "__main__":
    main()
