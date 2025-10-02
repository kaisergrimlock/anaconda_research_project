#!/usr/bin/env python3
# Clean a CSV by:
#  1) removing rows with missing/blank 'relevance'
#  2) removing duplicate docids (keeps first occurrence)
#
# Expected columns: query, docid, passage, relevance

from pathlib import Path
import csv

# ---- configure ----
INPUT_FILE  = Path("outputs/trec_dl_llm_label/llm_labels/all_docs_labels_utility.csv")
OUTPUT_FILE = Path("outputs/trec_dl_llm_label/processed/all_docs_label.cleaned.csv")
# -------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

kept = 0
skipped_missing_label = 0
skipped_duplicate_docid = 0
seen_docids = set()

with open(INPUT_FILE, "r", newline="", encoding="utf-8-sig") as fin:
    sample = fin.read(4096)
    fin.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.get_dialect("excel")

    reader = csv.DictReader(fin, dialect=dialect)
    fieldnames = reader.fieldnames or []
    required = {"query", "docid", "passage", "relevance"}
    missing_cols = required - set(map(str.strip, fieldnames))
    if missing_cols:
        raise KeyError(f"Missing columns: {sorted(missing_cols)}; found: {fieldnames}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            # treat None, empty string, or whitespace as missing label
            label = (row.get("relevance") or "").strip()
            if label == "":
                skipped_missing_label += 1
                continue

            docid = (row.get("docid") or "").strip()
            if docid in seen_docids:
                skipped_duplicate_docid += 1
                continue

            seen_docids.add(docid)
            writer.writerow(row)
            kept += 1

print(f"Done. Kept {kept} rows.")
print(f"Removed {skipped_missing_label} rows with missing labels.")
print(f"Removed {skipped_duplicate_docid} duplicate docid rows (kept first occurrence).")
