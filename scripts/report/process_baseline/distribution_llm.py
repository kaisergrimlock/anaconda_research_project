#!/usr/bin/env python3
# Aggregate label counts from ONE CSV file and APPEND to the summary CSV.
# Only labels {0,1,2,3} are considered; anything else => 0 (un-relevant).

from pathlib import Path
from collections import Counter
import csv

# ==== Configure these ====
INPUT_FILE   = Path("outputs/llm_label/trec_dl_2023_raw.csv")
OUTPUT_FILE  = Path("outputs/baseline/label_counts.csv")
JUDGE        = "llm"
LABEL_COLUMN = "relevance"   # fallback to 'label' if missing
# =========================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def detect_reader(path: Path):
    """Return a (file_handle, csv.DictReader) with a sniffed dialect (fallback to comma)."""
    f = open(path, "r", newline="", encoding="utf-8-sig")
    sample = f.read(4096)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.get_dialect("excel")
    return f, csv.DictReader(f, dialect=dialect)

# --- count labels (normalize to 0..3) ---
ALLOWED = {"0", "1", "2", "3"}
counts = Counter({k: 0 for k in ALLOWED})
rows_read = 0

fh, reader = detect_reader(INPUT_FILE)
with fh:
    fieldnames = [h.strip() for h in (reader.fieldnames or [])]
    if LABEL_COLUMN in fieldnames:
        lbl_col = LABEL_COLUMN
    elif "label" in fieldnames:
        lbl_col = "label"
    else:
        raise KeyError(
            f"Neither '{LABEL_COLUMN}' nor 'label' found in {INPUT_FILE}. "
            f"Available columns: {fieldnames}"
        )

    for row in reader:
        if not row:
            continue
        raw = str(row.get(lbl_col, "")).strip()
        norm = raw if raw in ALLOWED else "0"  # map anything else to 0
        counts[norm] += 1
        rows_read += 1

# --- append summary (always 0..3) ---
need_header = not OUTPUT_FILE.exists()

with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    if need_header:
        w.writerow(["label", "no. of docs", "judge"])
    for label in ("0", "1", "2", "3"):
        w.writerow([label, counts[label], JUDGE])

print(f"Processed {rows_read} rows from: {INPUT_FILE}")
print(f"Appended summary rows to: {OUTPUT_FILE}")
