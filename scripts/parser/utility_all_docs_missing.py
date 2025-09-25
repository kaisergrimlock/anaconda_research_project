#!/usr/bin/env python3
# Find docids that exist in "parts" CSVs but are missing from "all_docs" CSVs.

from pathlib import Path
import csv

# ================== Configure these ==================
PARTS_DIR       = Path("outputs/trec_dl_2019/retrieved/all_topics_in_parts")
PARTS_PATTERN   = "*.csv"

ALL_DOCS_DIR    = Path("outputs/trec_dl_llm_label/llm_labels")   # folder containing the all_docs*.csv files
ALL_DOCS_PATTERN= "all_docs_label_cleaned.csv"

OUTPUT_FILE     = Path("outputs/trec_dl_llm_label/processed/missing_from_all_docs.csv")
DOCID_COL       = "docid"       # column name for document id
# =====================================================

def dict_reader(path: Path):
    """Return (file_handle, DictReader) with sniffed dialect; caller must close fh."""
    fh = open(path, "r", newline="", encoding="utf-8-sig")
    sample = fh.read(4096)
    fh.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.get_dialect("excel")
    return fh, csv.DictReader(fh, dialect=dialect)

# --- Collect docids from parts (also remember where we first saw each docid and its query if available)
parts_docids = set()
first_seen_in_parts = {}  # docid -> relative path string
query_from_parts = {}     # docid -> query (if present)

part_files = sorted(PARTS_DIR.glob(PARTS_PATTERN))
if not part_files:
    raise FileNotFoundError(f"No parts files matching {PARTS_PATTERN} in {PARTS_DIR}")

for fp in part_files:
    fh, reader = dict_reader(fp)
    try:
        if DOCID_COL not in (reader.fieldnames or []):
            raise KeyError(f"{fp}: missing '{DOCID_COL}'. Columns: {reader.fieldnames}")
        for row in reader:
            docid = (row.get(DOCID_COL) or "").strip()
            if not docid:
                continue
            if docid not in parts_docids:
                parts_docids.add(docid)
                first_seen_in_parts[docid] = str(fp)
                if "query" in reader.fieldnames:
                    query_from_parts[docid] = (row.get("query") or "").strip()
    finally:
        fh.close()

# --- Collect docids from all_docs files
all_docs_files = sorted(ALL_DOCS_DIR.glob(ALL_DOCS_PATTERN))
if not all_docs_files:
    raise FileNotFoundError(f"No all_docs files matching {ALL_DOCS_PATTERN} in {ALL_DOCS_DIR}")

all_docs_docids = set()
for fp in all_docs_files:
    fh, reader = dict_reader(fp)
    try:
        if DOCID_COL not in (reader.fieldnames or []):
            raise KeyError(f"{fp}: missing '{DOCID_COL}'. Columns: {reader.fieldnames}")
        for row in reader:
            docid = (row.get(DOCID_COL) or "").strip()
            if docid:
                all_docs_docids.add(docid)
    finally:
        fh.close()

# --- Compute missing set (present in parts but not in all_docs)
missing_docids = sorted(parts_docids - all_docs_docids, key=lambda x: (len(x), x))

# --- Write report
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["docid", "first_seen_in_parts", "query_from_parts"])
    for did in missing_docids:
        w.writerow([did, first_seen_in_parts.get(did, ""), query_from_parts.get(did, "")])

print(f"Parts docids:     {len(parts_docids)} from {len(part_files)} files")
print(f"All-docs docids:  {len(all_docs_docids)} from {len(all_docs_files)} files")
print(f"Missing docids:   {len(missing_docids)}")
print(f"Wrote: {OUTPUT_FILE}")
