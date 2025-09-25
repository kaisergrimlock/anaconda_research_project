#!/usr/bin/env python3
# Collect docids that exist in parts but are missing from all_docs,
# and write their docid, query, passage.

from pathlib import Path
import csv

# ========= configure =========
PARTS_DIR    = Path("outputs/trec_dl_2019/retrieved/all_topics_in_parts")
PARTS_GLOB   = "*.csv"

ALL_DOCS_FILE = Path("outputs/trec_dl_llm_label/processed/all_docs_label_cleaned.csv")

OUTPUT_FILE  = Path("outputs/trec_dl_llm_label/processed/missing_labels.csv")

DOCID_COL    = "docid"
QUERY_COL    = "query"
PASSAGE_COL  = "passage"
# ============================

def dict_reader(path: Path):
    fh = open(path, "r", newline="", encoding="utf-8-sig")
    sample = fh.read(4096); fh.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.get_dialect("excel")
    return fh, csv.DictReader(fh, dialect=dialect)

# --- gather doc -> (query, passage) from parts
parts_info = {}  # docid -> (query, passage)
part_files = sorted(PARTS_DIR.glob(PARTS_GLOB))
if not part_files:
    raise FileNotFoundError(f"No files matching {PARTS_GLOB} in {PARTS_DIR}")

for fp in part_files:
    fh, rdr = dict_reader(fp)
    try:
        cols = rdr.fieldnames or []
        for need in (DOCID_COL, QUERY_COL, PASSAGE_COL):
            if need not in cols:
                raise KeyError(f"{fp}: missing column '{need}'. Columns: {cols}")
        for row in rdr:
            did = (row.get(DOCID_COL) or "").strip()
            if not did or did in parts_info:
                continue
            q = (row.get(QUERY_COL) or "").strip()
            p = (row.get(PASSAGE_COL) or "").strip()
            parts_info[did] = (q, p)
    finally:
        fh.close()

# --- read all_docs docids
if not ALL_DOCS_FILE.exists():
    raise FileNotFoundError(f"All-docs file not found: {ALL_DOCS_FILE}")

all_docs_ids = set()
fh, rdr = dict_reader(ALL_DOCS_FILE)
try:
    if DOCID_COL not in (rdr.fieldnames or []):
        raise KeyError(f"{ALL_DOCS_FILE}: missing '{DOCID_COL}'. Columns: {rdr.fieldnames}")
    for row in rdr:
        did = (row.get(DOCID_COL) or "").strip()
        if did:
            all_docs_ids.add(did)
finally:
    fh.close()

# --- compute missing and write details
missing_ids = sorted(set(parts_info.keys()) - all_docs_ids, key=lambda x: (len(x), x))
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow([DOCID_COL, QUERY_COL, PASSAGE_COL])  # header
    for did in missing_ids:
        q, p = parts_info[did]
        w.writerow([did, q, p])

print(f"Parts files: {len(part_files)} | parts docids: {len(parts_info)}")
print(f"All-docs docids: {len(all_docs_ids)}")
print(f"Missing: {len(missing_ids)}")
print(f"Wrote: {OUTPUT_FILE}")
