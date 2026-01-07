#!/usr/bin/env python3
import csv
from pathlib import Path

src = Path(r"retrieved/trec_dl_2022/translate_cache/query_map_ko_corrected.csv")
dst = Path(r"retrieved/trec_dl_2022/translate_cache/query_map_ko_corrected.clean.csv")

# Read raw lines (for debugging)
raw = src.read_text(encoding="utf-8-sig", errors="replace")
print("Total characters:", len(raw))
print("Total quotes in file:", raw.count('"'))

# Parse what DictReader can parse (will be 25 for your current file)
rows = []
with src.open("r", encoding="utf-8-sig", newline="") as fh:
    r = csv.DictReader(fh)
    for row in r:
        q = (row.get("query") or "").strip()
        t = (row.get("translated") or "").strip()
        if q:
            rows.append((q, t))

print("Rows parsed:", len(rows))

# Write a canonical clean CSV (proper quoting/escaping)
with dst.open("w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
    w.writerow(["query", "translated"])
    for q, t in rows:
        w.writerow([q, t])

print("Wrote:", dst)
