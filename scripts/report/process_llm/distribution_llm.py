#!/usr/bin/env python3
# Read a single CSV and write: label,no. of docs,collection
# Expects columns: query, docid, passage, relevance

from pathlib import Path
from collections import Counter
import csv

# ==== Configure these ====
INPUT_FILE   = Path("outputs/trec_dl_llm_label/processed/all_docs_label_cleaned.csv")
OUTPUT_FILE  = Path("outputs/trec_dl_llm_label/processed/label_counts.csv")
COLLECTION   = "llm"   # whatever label you want in the output
LABEL_COLUMN = "relevance"
# =========================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

counts = Counter()
total_rows = 0

with open(INPUT_FILE, "r", newline="", encoding="utf-8-sig") as f:
    sample = f.read(4096)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.get_dialect("excel")  # default comma
    reader = csv.DictReader(f, dialect=dialect)

    if LABEL_COLUMN not in (reader.fieldnames or []):
        raise KeyError(f"Column '{LABEL_COLUMN}' not found. Available: {reader.fieldnames}")

    for row in reader:
        if not row:
            continue
        label = str(row[LABEL_COLUMN]).strip()
        counts[label] += 1
        total_rows += 1

header_needed = (not OUTPUT_FILE.exists()) or (OUTPUT_FILE.stat().st_size == 0)
with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    if header_needed:
        w.writerow(["label", "no. of docs", "collection"])

    def sort_key(k):
        try:
            return (0, int(k))  # numeric sort if labels 0..3
        except ValueError:
            return (1, k)

    for label in sorted(counts.keys(), key=sort_key):
        w.writerow([label, counts[label], COLLECTION])

print(f"Wrote {len(counts)} rows (from {total_rows} docs) to: {OUTPUT_FILE}")
