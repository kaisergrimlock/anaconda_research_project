#!/usr/bin/env python3
# Aggregate label counts from many CSV files in a folder.
# Only labels {0,1,2,3} are considered; anything else => 0 (un-relevant).

from pathlib import Path
from collections import Counter
import csv
import re  # ### NEW

# ==== Configure these ====
TREC_DL_YEAR = "2023"
INPUT_DIR    = Path("retrieved/trec_dl_" + TREC_DL_YEAR + "/judged")  # folder with many CSVs
GLOB_PATTERN = "*.csv"                      # which files to include

# Base output file name (without model suffix)
OUTPUT_FILE  = Path("outputs/baseline/trec_dl_{TREC_DL_YEAR}_raw.csv")

JUDGE        = "NIST"                       # constant written to the output
LABEL_COLUMN = "relevance"                  # will fall back to 'label' if missing

# Append this model name to the output file name
#MODEL_NAME = "openai.gpt-oss-20b-1_0"    # ### NEW (set this however you like)
MODEL_NAME = "anthropic.claude-3-5-haiku-20241022-v1:0"
# =========================

def _sanitize_for_path(s: str) -> str:      # ### NEW
    """Keep alnum, dot, dash, underscore; replace others with '-'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")

# Compute final output path with model suffix                    ### NEW
if MODEL_NAME:
    suffix = OUTPUT_FILE.suffix
    stem   = OUTPUT_FILE.stem
    parent = OUTPUT_FILE.parent
    safe_model = _sanitize_for_path(MODEL_NAME)
    OUTPUT_FILE = parent / f"{stem}_{safe_model}{suffix}"

files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
if not files:
    raise FileNotFoundError(f"No files matching {GLOB_PATTERN} in {INPUT_DIR}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Pre-seed to guarantee keys exist and to enforce 0..3 only
ALLOWED = {"0", "1", "2", "3"}
counts = Counter({k: 0 for k in ALLOWED})
total_rows = 0
files_read = 0

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

for fp in files:
    f, reader = detect_reader(fp)

    # choose label column per-file (relevance preferred; fallback to label)
    fieldnames = [h.strip() for h in (reader.fieldnames or [])]
    if LABEL_COLUMN in fieldnames:
        lbl_col = LABEL_COLUMN
    elif "label" in fieldnames:
        lbl_col = "label"
    else:
        f.close()
        raise KeyError(
            f"Neither '{LABEL_COLUMN}' nor 'label' found in {fp}. "
            f"Available columns: {fieldnames}"
        )

    for row in reader:
        if not row:
            continue
        raw = str(row.get(lbl_col, "")).strip()

        # Normalize: only keep {0,1,2,3}; everything else -> "0"
        norm = raw if raw in ALLOWED else "0"

        counts[norm] += 1
        total_rows += 1

    f.close()
    files_read += 1

# write summary (fixed order 0..3)
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["label", "no. of docs", "judge"])
    for label in ("0", "1", "2", "3"):
        w.writerow([label, counts[label], JUDGE])

print(f"Processed {files_read} files, {total_rows} rows.")
print(f"Wrote summary to: {OUTPUT_FILE}")
