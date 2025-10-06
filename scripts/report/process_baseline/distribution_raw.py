#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import pandas as pd
import csv, sys

# ==== Configure these ====
TREC_DL_YEAR = "2023"
INPUT_DIR    = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged")
GLOB_PATTERN = "*.csv"
OUTPUT_FILE  = Path("outputs/baseline/label_counts.csv")
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

def _bump_field_limit():
    limit = min(2_000_000_000, getattr(sys, "maxsize", 2_000_000_000))
    while limit >= 131072:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2

_bump_field_limit()

def load_label_series(fp: Path) -> pd.Series:
    """
    Read just the label column from a CSV, robust to very long fields.
    Tries 'relevance' then falls back to 'label'. Uses chunks to limit memory.
    """
    # First read the header only to detect the correct column name
    hdr = pd.read_csv(fp, nrows=0, encoding="utf-8-sig", engine="python")
    cols = [c.strip() for c in hdr.columns]
    col = LABEL_COLUMN if LABEL_COLUMN in cols else ("label" if "label" in cols else None)
    if col is None:
        raise KeyError(f"Neither '{LABEL_COLUMN}' nor 'label' found in {fp}. Available: {cols}")

    # Stream in chunks to avoid loading the huge 'passage' column
    ser_parts = []
    for chunk in pd.read_csv(
        fp,
        usecols=[col],                 # <- only read the label column
        dtype=str,                    # keep labels as strings
        encoding="utf-8-sig",
        engine="python",              # more tolerant for messy rows/quotes
        chunksize=200_000,            # tune if needed
        on_bad_lines="skip"           # skip any malformed lines
    ):
        ser_parts.append(chunk[col].astype(str).str.strip())
    if not ser_parts:
        return pd.Series(dtype=str)
    return pd.concat(ser_parts, ignore_index=True)

for fp in files:
    s = load_label_series(fp)
    # normalize to {0,1,2,3}, everything else -> "0"
    s_norm = s.where(s.isin(ALLOWED), other="0")
    vc = s_norm.value_counts()
    for k, v in vc.items():
        if k in ALLOWED:
            counts[k] += int(v)
    total_rows += int(vc.sum())
    files_read += 1

# write summary (fixed order 0..3)
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
    out.write("label,no. of docs,judge\n")
    for label in ("0", "1", "2", "3"):
        out.write(f"{label},{counts[label]},{JUDGE}\n")

print(f"Processed {files_read} files, {total_rows} rows.")
print(f"Wrote summary to: {OUTPUT_FILE}")
