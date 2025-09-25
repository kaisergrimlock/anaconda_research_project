#!/usr/bin/env python3
# Aggregate label counts from many CSV files in a folder.
# Input CSVs must have a 'relevance' column (or fallback to 'label').

from pathlib import Path
from collections import Counter
import csv

# ==== Configure these ====
INPUT_DIR    = Path("outputs/trec_dl_2019/retrieved/all_topics_in_parts")  # folder with many CSVs
GLOB_PATTERN = "*.csv"                      # which files to include
OUTPUT_FILE  = Path("outputs/trec_dl_llm_label/processed/label_counts.csv")
JUDGE   = "NIST"               # constant written to the output
LABEL_COLUMN = "relevance"                  # will fall back to 'label' if missing
# =========================

files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
if not files:
    raise FileNotFoundError(f"No files matching {GLOB_PATTERN} in {INPUT_DIR}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

counts = Counter()
total_rows = 0
files_read = 0

def detect_reader(path: Path):
    """Return a csv.DictReader with a sniffed dialect (fallback to comma)."""
    f = open(path, "r", newline="", encoding="utf-8-sig")
    sample = f.read(4096)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.get_dialect("excel")
    return f, csv.DictReader(f, dialect=dialect)

for fp in files:
    with open(fp, "r", newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.get_dialect("excel")
        reader = csv.DictReader(f, dialect=dialect)

        # choose label column per-file (relevance preferred; fallback to label)
        fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        if LABEL_COLUMN in fieldnames:
            lbl_col = LABEL_COLUMN
        elif "label" in fieldnames:
            lbl_col = "label"
        else:
            raise KeyError(
                f"Neither '{LABEL_COLUMN}' nor 'label' found in {fp}. "
                f"Available columns: {fieldnames}"
            )

        for row in reader:
            if not row:
                continue
            label = str(row[lbl_col]).strip()
            counts[label] += 1
            total_rows += 1
        files_read += 1

# write summary
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["label", "no. of docs", "judge"])

    def sort_key(k):
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)

    for label in sorted(counts, key=sort_key):
        w.writerow([label, counts[label], JUDGE])

print(f"Processed {files_read} files, {total_rows} rows.")
print(f"Wrote summary to: {OUTPUT_FILE}")
