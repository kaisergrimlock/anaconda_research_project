#!/usr/bin/env python3
"""
Find rows from retrieved/trec_dl_2023/** that are missing in the labels file.

- Keys used to match: (query, docid)  [configurable]
- Labels file: outputs/llm_label/trec_dl_2023_raw.csv
- Retrieved dir: retrieved/trec_dl_2023 (recurses into subfolders)
- Output: outputs/llm_label/missing_or_unjudged_formatted.csv
  Columns: query, docid, passage, relevance (empty)

Label normalization rule (for presence check only):
  only '0','1','2','3' are valid; anything else is coerced to '0' (still counts
  as "present", so it won't be flagged as missing).
"""

from pathlib import Path
import sys
import csv

def allow_huge_csv_fields():
    # Robustly raise the CSV field size limit (works on Windows where sys.maxsize may overflow)
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

allow_huge_csv_fields()

# ==== Configure these ====
LABELS_FILE   = Path("outputs/llm_label/gpt-oss-20b/gpt-oss-20b_trec_dl_2023_eng_raw.csv")
RETRIEVED_DIR = Path("retrieved/trec_dl_2023")   # searches **/*.csv
OUTPUT_MISSING = Path("outputs/llm_label/missing_or_unjudged_formatted.csv")

# Column aliases
DOCID_COL_CHOICES   = ("docid", "pid", "pid_resolved", "pid_qrels", "docno", "doc_id", "id")
QUERY_COL_CHOICES   = ("query", "query_en", "topic", "question", "qid")
PASSAGE_COL_CHOICES = ("passage", "text", "body")

ALLOWED_LABELS = {"0", "1", "2", "3"}  # kept for normalization (not needed for presence)
# =========================


def detect_reader(path: Path):
    """Open a CSV with sniffed dialect; returns (file_handle, DictReader)."""
    f = open(path, "r", newline="", encoding="utf-8-sig")
    sample = f.read(4096)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.get_dialect("excel")
    return f, csv.DictReader(f, dialect=dialect)


def pick_first(fields, choices):
    fs = [h.strip() for h in (fields or [])]
    for c in choices:
        if c in fs:
            return c
    return None


def pick_docid_col(fieldnames):
    col = pick_first(fieldnames, DOCID_COL_CHOICES)
    if not col:
        raise KeyError(f"Could not find any doc-id column in: {fieldnames}")
    return col


def pick_query_col(fieldnames):
    # optional
    return pick_first(fieldnames, QUERY_COL_CHOICES)


def pick_passage_col(fieldnames):
    # optional
    return pick_first(fieldnames, PASSAGE_COL_CHOICES)


# ---- 1) Collect all retrieved DOCIDs (and keep query/passage for output) ----
retrieved_docids = set()
docid_to_details = {}  # docid -> (query_opt, passage_opt)

csv_files = sorted(RETRIEVED_DIR.rglob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found under {RETRIEVED_DIR}")

for p in csv_files:
    fh, reader = detect_reader(p)
    with fh:
        dcol = pick_docid_col(reader.fieldnames)
        qcol = pick_query_col(reader.fieldnames)
        pcol = pick_passage_col(reader.fieldnames)
        for row in reader:
            if not row:
                continue
            did = str(row.get(dcol, "")).strip()
            if not did:
                continue
            retrieved_docids.add(did)
            if did not in docid_to_details:
                qval = str(row.get(qcol, "")).strip() if qcol else ""
                pval = str(row.get(pcol, "")).strip() if pcol else ""
                docid_to_details[did] = (qval, pval)

# ---- 2) Collect all labeled DOCIDs (label value itself irrelevant for presence) ----
if not LABELS_FILE.exists():
    raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")

labeled_docids = set()
fh, reader = detect_reader(LABELS_FILE)
with fh:
    dcol = pick_docid_col(reader.fieldnames)
    fields = [h.strip() for h in (reader.fieldnames or [])]
    lbl_col = "relevance" if "relevance" in fields else ("label" if "label" in fields else None)

    for row in reader:
        if not row:
            continue
        did = str(row.get(dcol, "")).strip()
        if not did:
            continue
        # keep normalization for completeness, though not used for presence
        if lbl_col:
            raw = str(row.get(lbl_col, "")).strip()
            _norm = raw if raw in ALLOWED_LABELS else "0"
        labeled_docids.add(did)

# ---- 3) Compute missing docids ----
missing_docids = sorted(retrieved_docids - labeled_docids)

# ---- 4) Write output (formatted for later labeling) ----
OUTPUT_MISSING.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_MISSING, "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["query", "docid", "passage", "relevance"])
    for did in missing_docids:
        q, psg = docid_to_details.get(did, ("", ""))
        w.writerow([q, did, psg, ""])

print(f"Retrieved docids: {len(retrieved_docids)}")
print(f"Labeled docids:   {len(labeled_docids)}")
print(f"Missing docids:   {len(missing_docids)}")
print(f"Wrote missing rows to: {OUTPUT_MISSING}")
