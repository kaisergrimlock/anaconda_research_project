#!/usr/bin/env python3
# Check for missing values in the "relevance" (or fallback "label") column
# of a single CSV, and write a report including docid, query, and passage.

from pathlib import Path
import csv

# ---- configure ----
INPUT_FILE  = Path("outputs/trec_dl_llm_label/processed/all_llm_labels.csv")
LABEL_COLUMN = "relevance"   # fallback to 'label' if needed
REPORT_FILE = INPUT_FILE.parent / "relevance_missing_rows.csv"
# -------------------

def detect_reader(path: Path):
    """Return (file_handle, DictReader) with a sniffed dialect (fallback to comma)."""
    f = open(path, "r", newline="", encoding="utf-8-sig")
    sample = f.read(4096)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.get_dialect("excel")
    return f, csv.DictReader(f, dialect=dialect)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

missing = []
total = 0

fh, reader = detect_reader(INPUT_FILE)
with fh:
    headers = [h.strip() for h in (reader.fieldnames or [])]
    if LABEL_COLUMN not in headers and "label" in headers:
        lbl_col = "label"
    elif LABEL_COLUMN in headers:
        lbl_col = LABEL_COLUMN
    else:
        raise KeyError(
            f"Neither '{LABEL_COLUMN}' nor 'label' found in {INPUT_FILE}. "
            f"Available columns: {headers}"
        )

    for row in reader:
        if not row:
            continue
        total += 1
        val = (row.get(lbl_col, "") or "").strip()
        if val == "":
            missing.append({
                "docid": row.get("docid", ""),
                "query": row.get("query", ""),
                "passage": row.get("passage", ""),
                lbl_col: val
            })

missing_count = len(missing)
if missing_count == 0:
    print(f"OK ✅  No missing '{lbl_col}' values found in {INPUT_FILE}. Scanned {total} data rows.")
else:
    with REPORT_FILE.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=["docid", "query", "passage", lbl_col])
        w.writeheader()
        w.writerows(missing)

    pct = (missing_count / max(total, 1)) * 100.0
    print(f"Found {missing_count} missing '{lbl_col}' values out of {total} rows ({pct:.2f}%).")
    print(f"Wrote details to: {REPORT_FILE}")
