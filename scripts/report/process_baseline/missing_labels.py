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
import csv

# ==== Configure these ====
LABELS_FILE = Path("outputs/llm_label/trec_dl_2023_raw.csv")
RETRIEVED_DIR = Path("retrieved/trec_dl_2023")     # searches **/*.csv
OUTPUT_MISSING = Path("outputs/llm_label/missing_or_unjudged_formatted.csv")

# Candidate key-column pairs to try (first that exists in a file is used)
# e.g., some files use 'topic' instead of 'query'.
KEY_CHOICES = [
    ("query", "docid"),
    ("topic", "docid"),
]
PASSAGE_COL_CHOICES = ("passage", "text", "body")

ALLOWED_LABELS = {"0", "1", "2", "3"}
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


def pick_keys(fieldnames):
    """Pick (key1,key2) present in fieldnames."""
    fields = [h.strip() for h in (fieldnames or [])]
    for a, b in KEY_CHOICES:
        if a in fields and b in fields:
            return a, b
    raise KeyError(f"Could not find any usable key pair in columns: {fields}")


def pick_passage_col(fieldnames):
    fields = [h.strip() for h in (fieldnames or [])]
    for c in PASSAGE_COL_CHOICES:
        if c in fields:
            return c
    # Passage isn't strictly required for keying; fallback to empty
    return None


# ---- 1) Collect all retrieved keys (and keep a passage for output) ----
retrieved_keys = set()
key_to_passage = {}

csv_files = sorted(RETRIEVED_DIR.rglob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found under {RETRIEVED_DIR}")

for p in csv_files:
    fh, reader = detect_reader(p)
    with fh:
        k1, k2 = pick_keys(reader.fieldnames)
        pcol = pick_passage_col(reader.fieldnames)
        for row in reader:
            if not row:
                continue
            k = (row[k1].strip(), row[k2].strip())
            retrieved_keys.add(k)
            # Keep first seen passage
            if k not in key_to_passage:
                key_to_passage[k] = row.get(pcol, "").strip() if pcol else ""

# ---- 2) Collect all labeled keys (coerce non-0..3 labels to '0', still present) ----
if not LABELS_FILE.exists():
    raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")

labeled_keys = set()
fh, reader = detect_reader(LABELS_FILE)
with fh:
    k1, k2 = pick_keys(reader.fieldnames)
    # choose label column
    fields = [h.strip() for h in (reader.fieldnames or [])]
    if "relevance" in fields:
        lbl_col = "relevance"
    elif "label" in fields:
        lbl_col = "label"
    else:
        raise KeyError(
            f"Neither 'relevance' nor 'label' found in {LABELS_FILE}. "
            f"Available columns: {fields}"
        )

    for row in reader:
        if not row:
            continue
        raw = str(row.get(lbl_col, "")).strip()
        _norm = raw if raw in ALLOWED_LABELS else "0"  # normalized (still present)
        # Regard row as labeled regardless of original value (after normalization)
        k = (row[k1].strip(), row[k2].strip())
        labeled_keys.add(k)

# ---- 3) Compute missing keys ----
missing = sorted(retrieved_keys - labeled_keys)

# ---- 4) Write output (formatted for later labeling) ----
OUTPUT_MISSING.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_MISSING, "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["query", "docid", "passage", "relevance"])
    for q, d in missing:
        w.writerow([q, d, key_to_passage.get((q, d), ""), ""])

print(f"Retrieved keys: {len(retrieved_keys)}")
print(f"Labeled keys:   {len(labeled_keys)}")
print(f"Missing keys:   {len(missing)}")
print(f"Wrote missing rows to: {OUTPUT_MISSING}")
