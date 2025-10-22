#!/usr/bin/env python3
"""
Count unique queries (by QID) across many CSV files.

Configure FOLDER and PATTERNS below.
- PATTERNS accepts multiple glob patterns (e.g., ["*.csv", "*.csv.gz"])
- Works with large CSV cells
- Counts unique QIDs, total rows, and optional per-QID counts
"""

from __future__ import annotations
import sys, csv, gzip
from pathlib import Path
from collections import Counter

# =========================
# Variables (edit these)
# =========================
TRECDL_YEAR = "2022"
FOLDER = Path("retrieved") / f"trec_dl_{TRECDL_YEAR}" / "judged"   # folder containing your chunked CSVs
PATTERNS = ["*.csv", "*.csv.gz"]              # one or more filename patterns
QID_COL  = "qid"                              # column name for query id
SHOW_BY_QID = False                           # True -> print "qid,count" breakdown
# =========================


def allow_huge_csv_fields():
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

def open_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")

def iter_csv_rows(path: Path):
    with open_maybe_gz(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row

def main():
    allow_huge_csv_fields()

    folder = Path(FOLDER)
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    # Collect files matching any of the patterns
    files = []
    for pat in PATTERNS:
        files.extend(folder.glob(pat))
    files = sorted(p for p in files if p.is_file())

    if not files:
        print(f"No files matched in {folder} with patterns {PATTERNS}", file=sys.stderr)
        sys.exit(1)

    total_rows = 0
    missing_qid_rows = 0
    qid_counts: Counter[str] = Counter()
    unique_qid_query_pairs = set()

    # Optionally capture a few sample headers to help debug wrong QID column names
    sample_headers = {}

    for f in files:
        # Peek header (non-fatal if it fails)
        try:
            with open_maybe_gz(f) as fh:
                r = csv.reader(fh)
                header = next(r, None)
                if header and len(sample_headers) < 3:
                    sample_headers[f.name] = header
        except Exception:
            pass

        try:
            for row in iter_csv_rows(f):
                total_rows += 1
                qid = (row.get(QID_COL) or "").strip()
                if not qid:
                    missing_qid_rows += 1
                    continue
                qid_counts[qid] += 1

                qtxt = (row.get("query") or "").strip()
                unique_qid_query_pairs.add((qid, qtxt))
        except Exception as e:
            print(f"WARNING: Failed reading {f.name}: {e}", file=sys.stderr)

    unique_qids = len(qid_counts)
    unique_queries_by_text = len({q for (_qid, q) in unique_qid_query_pairs if q})

    print("\n=== Query Tally ===")
    print(f"Folder                   : {folder}")
    print(f"Patterns                 : {PATTERNS}")
    print(f"Files matched            : {len(files)}")
    print(f"Total rows read          : {total_rows}")
    print(f"Rows missing '{QID_COL}' : {missing_qid_rows}")
    print(f"Unique QIDs (queries)    : {unique_qids}")
    print(f"Unique (qid, query) pairs: {len(unique_qid_query_pairs)}")
    print(f"Unique queries by text   : {unique_queries_by_text}")

    if SHOW_BY_QID:
        print("\nqid,count")
        for qid, cnt in qid_counts.most_common():
            print(f"{qid},{cnt}")

    if unique_qids == 0 and sample_headers:
        print("\n[Hint] No QIDs found. Check QID_COL (currently:", QID_COL, ")")
        print("Sample headers seen:")
        for fname, header in sample_headers.items():
            print(f"  {fname}: {header}")

if __name__ == "__main__":
    main()
