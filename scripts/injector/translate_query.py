#!/usr/bin/env python3
from __future__ import annotations
import csv, sys
from pathlib import Path
import boto3

# =========================
# HARD-CODED SETTINGS
# =========================
# This file lives at .../scripts/report/process_baseline/*.py
# Go up 3 levels to reach repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

TRECDL_YEAR  = "2023"          # "2019", "2020", or "2023"
TARGET_LANG  = "ru"            # e.g., "vi" (Vietnamese), "fr" (French)
AWS_REGION   = "ap-southeast-2"

INPUT_DIR    = REPO_ROOT / f"retrieved/trec_dl_{TRECDL_YEAR}/judged"
FILE_GLOB    = "all_topics_trecdl_*_part*.csv"   # safe pattern
OUTPUT_DIR   = REPO_ROOT / "outputs/queries"
OUTPUT_CSV   = OUTPUT_DIR / f"trec_dl_{TRECDL_YEAR}_queries_{TARGET_LANG}.csv"

DEDUP_BEFORE_TRANSLATE = True  # save cost & time; preserves first-seen order
# =========================

def _allow_huge_csv_fields():
    limit = min(2_000_000_000, getattr(sys, "maxsize", 2_000_000_000))
    while limit >= 131072:
        try:
            csv.field_size_limit(limit); return
        except OverflowError:
            limit //= 2
_allow_huge_csv_fields()

def _find_query_idx(header: list[str]) -> int:
    for i, h in enumerate(header or []):
        if h is not None and h.strip().lower() == "query":
            return i
    raise KeyError("No 'query' column found.")

def _extract_queries_from_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return []
        qidx = _find_query_idx(header)
        out: list[str] = []
        for row in r:
            if qidx < len(row):
                q = (row[qidx] or "").strip()
                if q:
                    out.append(q)
        return out

def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def main():
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    files = sorted(INPUT_DIR.glob(FILE_GLOB))
    if not files:
        print(f"ERROR: No CSV files matching '{FILE_GLOB}' in {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    # 1) Collect queries
    all_queries: list[str] = []
    for fp in files:
        try:
            qs = _extract_queries_from_file(fp)
            all_queries.extend(qs)
            print(f"[OK] {fp.name}: +{len(qs)} queries")
        except Exception as e:
            print(f"[SKIP] {fp.name}: {e}", file=sys.stderr)

    if not all_queries:
        print("No queries found.", file=sys.stderr)
        sys.exit(1)

    if DEDUP_BEFORE_TRANSLATE:
        queries = _dedup_preserve_order(all_queries)
        print(f"De-duplicated: {len(all_queries)} → {len(queries)} unique queries")
    else:
        queries = all_queries

    # 2) Translate
    translate = boto3.client("translate", region_name=AWS_REGION)
    translations: dict[str, str] = {}
    col_trans = f"query_{TARGET_LANG}"

    for i, q in enumerate(queries, 1):
        try:
            resp = translate.translate_text(
                Text=q,
                SourceLanguageCode="auto",
                TargetLanguageCode=TARGET_LANG
            )
            translations[q] = resp["TranslatedText"]
        except Exception as e:
            print(f"[WARN] translate failed on #{i}: {e}", file=sys.stderr)
            translations[q] = ""  # keep row with blank translation

        if i % 100 == 0 or i == len(queries):
            print(f"Translated {i}/{len(queries)}")

    # 3) Write output (two columns)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", col_trans])
        if DEDUP_BEFORE_TRANSLATE:
            # write unique queries only (common for reference lists)
            for q in queries:
                w.writerow([q, translations.get(q, "")])
        else:
            # write in original (possibly duplicate) order
            for q in all_queries:
                w.writerow([q, translations.get(q, "")])

    print(f"\nDone. Wrote {len(queries) if DEDUP_BEFORE_TRANSLATE else len(all_queries)} rows to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
