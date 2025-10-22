#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import pandas as pd
import csv, sys

# ==== Configure these ====
TREC_DL_YEAR = "2019"
INPUT_DIR    = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged")
GLOB_PATTERN = "*.csv"
OUTPUT_FILE  = Path(f"outputs/baseline/label_counts_{TREC_DL_YEAR}.csv")
JUDGE        = "NIST"
LABEL_COLUMN = "relevance"   # fallback to 'label' if missing
# =========================

files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
if not files:
    raise FileNotFoundError(f"No files matching {GLOB_PATTERN} in {INPUT_DIR}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

ALLOWED = {"0", "1", "2", "3"}
counts = Counter({k: 0 for k in ALLOWED})
total_rows = 0
files_read = 0

# track unique queries across all files
unique_queries = set()

def _bump_field_limit():
    limit = min(2_000_000_000, getattr(sys, "maxsize", 2_000_000_000))
    while limit >= 131072:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2

_bump_field_limit()

def _detect_col(fp: Path, preferred: str, fallback: str) -> str | None:
    """Read header only and find preferred or fallback column name (exact match)."""
    hdr = pd.read_csv(fp, nrows=0, encoding="utf-8-sig", engine="python")
    cols = [c.strip() for c in hdr.columns]
    if preferred in cols:
        return preferred
    if fallback in cols:
        return fallback
    return None

def load_label_series(fp: Path) -> pd.Series:
    """
    Read just the label column from a CSV, robust to very long fields.
    Tries 'relevance' then falls back to 'label'. Uses chunks to limit memory.
    """
    col = _detect_col(fp, LABEL_COLUMN, "label")
    if col is None:
        hdr = pd.read_csv(fp, nrows=0, encoding="utf-8-sig", engine="python")
        raise KeyError(f"Neither '{LABEL_COLUMN}' nor 'label' found in {fp}. Available: {list(hdr.columns)}")

    ser_parts = []
    for chunk in pd.read_csv(
        fp,
        usecols=[col],
        dtype=str,
        encoding="utf-8-sig",
        engine="python",
        chunksize=200_000,
        on_bad_lines="skip"
    ):
        ser_parts.append(chunk[col].astype(str).str.strip())
    if not ser_parts:
        return pd.Series(dtype=str)
    return pd.concat(ser_parts, ignore_index=True)

def add_unique_queries(fp: Path):
    """
    Pull a 'query-like' column and add to the global unique set.
    Prefers 'query', falls back to 'qid'. Missing -> skipped.
    """
    qcol = _detect_col(fp, "query", "qid")
    if qcol is None:
        return
    for chunk in pd.read_csv(
        fp,
        usecols=[qcol],
        dtype=str,
        encoding="utf-8-sig",
        engine="python",
        chunksize=200_000,
        on_bad_lines="skip"
    ):
        # normalize string representation
        qs = chunk[qcol].astype(str).str.strip()
        unique_queries.update(qs.dropna().tolist())

for fp in files:
    # labels
    s = load_label_series(fp)
    s_norm = s.where(s.isin(ALLOWED), other="0")
    vc = s_norm.value_counts()
    for k, v in vc.items():
        if k in ALLOWED:
            counts[k] += int(v)
    total_rows += int(vc.sum())
    files_read += 1

    # queries
    add_unique_queries(fp)

# write summary (fixed order 0..3), plus unique_queries row at the end
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
    out.write("label,no. of docs,judge\n")
    for label in ("0", "1", "2", "3"):
        out.write(f"{label},{counts[label]},{JUDGE}\n")
    # If you don't want this mixed into the label CSV, comment this next line and write a separate file instead
    out.write(f"unique_queries,{len(unique_queries)},{JUDGE}\n")

print(f"Processed {files_read} files, {total_rows} rows.")
print(f"Unique queries: {len(unique_queries)}")
print(f"Wrote summary to: {OUTPUT_FILE}")
