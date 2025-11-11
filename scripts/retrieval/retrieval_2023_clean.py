#!/usr/bin/env python3
"""
Clean noisy passages in TREC-DL CSV chunks.

What it does
------------
- Loops over CSVs (e.g., retrieved/trec_dl_2023/judged/all_topics_trecdl_2023_part*.csv)
- Detects junk passages (Shopify/product JSON blobs, script-y data dumps)
- Two modes:
    1) FILTER: drop the row entirely
    2) SANITIZE: keep the row but replace 'passage' with a cleaned, human-readable chunk
- Writes cleaned files to: <INPUT_DIR>/cleaned/<same filename>
- Prints a per-file and grand summary

Tweak the heuristics in `looks_like_junk()` and `sanitize_passage()` to be stricter/looser.

Usage
-----
Just run the script. Adjust CONFIG below if needed.
"""

from __future__ import annotations
import csv
import re
from html import unescape
from pathlib import Path
from typing import Iterable

# =========================
# CONFIG
# =========================
TREC_DL_YEAR = "2023"
INPUT_DIR    = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged")
GLOB_PATTERN = "all_topics_trecdl_" + TREC_DL_YEAR + "_part*.csv"
OUTPUT_DIR   = INPUT_DIR / "cleaned"

# Choose cleaning behavior:
MODE = "SANITIZE"  # "FILTER" to drop rows, "SANITIZE" to keep row but replace passage text

# Optional: cap maximum cleaned passage length (characters)
MAX_PASSAGE_CHARS = 2000

# =========================
# Heuristics
# =========================

# Big JSON-ish blocks (array/object with 200+ chars inside) — typical of product feeds
JSON_LIKE = re.compile(r'[\{\[][^{}\[\]]{200,}[\}\]]')
MULTISPACE = re.compile(r'\s+')

# Shopify / product-ish telltales
PRODUCT_KEYWORDS = (
    "variant_ids", "product_id", "Add to cart", "shopify.com",
    "\"options\": [", "\"featured_image\": {", "\"inventory_quantity\"",
    "\"available\":true", "\"title\":", "\"price\":", "\"sku\":"
)

def looks_like_junk(text: str) -> bool:
    """Return True if the passage is likely a product/script/JSON dump."""
    if not text:
        return False

    s = text.strip()
    # Direct keyword hits (very reliable for Shopify blobs)
    lower = s.lower()
    for kw in PRODUCT_KEYWORDS:
        if kw.lower() in lower:
            return True

    # Too many braces/quotes/colons relative to letters → data blob
    punct = sum(ch in '{}[],:;"\'\\/|' for ch in s)
    letters = sum(ch.isalpha() for ch in s)
    if letters > 0 and punct / (letters + 1e-6) > 0.6:
        return True

    # Contains a big JSON-like block
    if JSON_LIKE.search(s) is not None:
        return True

    return False

def sanitize_passage(text: str) -> str:
    """
    Try to extract a human-readable chunk from noisy text:
      1) Remove giant JSON-ish hunks
      2) Drop lines dominated by punctuation
      3) HTML unescape + collapse whitespace
      4) Keep the longest letter-dense sentence-like chunk
    """
    if not text:
        return ""

    # 1) Remove JSON hunks
    cleaned = JSON_LIKE.sub(" ", text)

    # 2) Drop very punctuation-heavy lines
    lines = []
    for line in cleaned.splitlines():
        ln = line.strip()
        if not ln:
            continue
        punct = sum(ch in '{}[],:;"\'\\/\t' for ch in ln)
        if len(ln) >= 40 and punct / len(ln) > 0.35:
            continue
        lines.append(ln)
    cleaned = " ".join(lines)

    # 3) Unescape + whitespace
    cleaned = unescape(cleaned)
    cleaned = MULTISPACE.sub(" ", cleaned).strip()

    # 4) Pick a readable chunk
    chunks = re.split(r'(?<=[.!?])\s+', cleaned) or [cleaned]
    best = max(chunks, key=lambda s: sum(c.isalpha() for c in s))
    if MAX_PASSAGE_CHARS:
        best = best[:MAX_PASSAGE_CHARS]
    return best

# =========================
# IO helpers
# =========================
def csv_iter_rows(path: Path):
    # Bump field size for very long passages
    try:
        import sys
        max_int = sys.maxsize
        import csv as _csv
        while True:
            try:
                _csv.field_size_limit(max_int)
                break
            except OverflowError:
                max_int = int(max_int / 10)
                if max_int < 131072:
                    _csv.field_size_limit(10_000_000)
                    break
    except Exception:
        pass

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        yield from r

def write_header(path: Path):
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["query", "docid", "passage", "relevance"])

# =========================
# Main cleaning routine
# =========================
def clean_file(in_path: Path, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header(out_path)

    kept = 0
    filtered = 0
    sanitized = 0

    with out_path.open("a", encoding="utf-8", newline="") as outf:
        w = csv.writer(outf)
        for row in csv_iter_rows(in_path):
            query   = row.get("query", "")
            docid   = row.get("docid", "")
            passage = row.get("passage", "") or ""
            rel     = row.get("relevance", row.get("label", ""))

            if looks_like_junk(passage):
                if MODE.upper() == "FILTER":
                    filtered += 1
                    continue
                else:  # SANITIZE
                    cleaned = sanitize_passage(passage)
                    w.writerow([query, docid, cleaned, rel])
                    kept += 1
                    sanitized += 1
            else:
                # Non-junk: still trim whitespace explosions
                cleaned = MULTISPACE.sub(" ", (passage or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")).strip()
                w.writerow([query, docid, cleaned, rel])
                kept += 1

    return {
        "file": in_path.name,
        "kept": kept,
        "filtered": filtered,
        "sanitized": sanitized,
        "mode": MODE.upper()
    }

def main():
    files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
    if not files:
        print(f"[INFO] No files matching {GLOB_PATTERN} in {INPUT_DIR}")
        return

    grand_kept = grand_filtered = grand_sanitized = 0
    print(f"[CLEAN] Mode={MODE}  in={INPUT_DIR}  out={OUTPUT_DIR}")
    for p in files:
        outp = OUTPUT_DIR / p.name
        stats = clean_file(p, outp)
        grand_kept      += stats["kept"]
        grand_filtered  += stats["filtered"]
        grand_sanitized += stats["sanitized"]
        print(f" - {stats['file']}: kept={stats['kept']}, filtered={stats['filtered']}, sanitized={stats['sanitized']}")

    print("\n[SUMMARY]")
    print(f"Kept rows     : {grand_kept:,}")
    print(f"Filtered rows : {grand_filtered:,}")
    print(f"Sanitized rows: {grand_sanitized:,}")
    print(f"Output folder : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
