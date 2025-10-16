#!/usr/bin/env python3
from __future__ import annotations
import csv, sys
from pathlib import Path
from typing import Dict

# ========= EDIT THESE PATHS IF NEEDED =========
JUDGED_DIR   = Path("retrieved/trec_dl_2023/judged")
PART_PATTERN = "all_topics_trecdl_2023_part*.csv"
OUTPUT_CSV   = Path("outputs/queries/first_nonrelevant_per_query.csv")
# ==============================================

COL_QID        = "qid"
COL_QUERY      = "query"
COL_PID_QRELS  = "pid_qrels"
COL_PID_RES    = "pid_resolved"
COL_PASSAGE    = "passage"
COL_RELEVANCE  = "relevance"

def _bump_field_limit():
    try:
        limit = min(2_000_000_000, getattr(sys, "maxsize", 2_000_000_000))
        while limit >= 131072:
            try:
                csv.field_size_limit(limit); return
            except OverflowError:
                limit //= 2
    except Exception:
        pass
_bump_field_limit()

def _to_int(x) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return 0

def main():
    if not JUDGED_DIR.exists():
        sys.exit(f"[FATAL] Folder not found: {JUDGED_DIR}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Keep first non-relevant row per query (by file order, then row order)
    first_zero: Dict[str, Dict[str, str]] = {}

    files = sorted(JUDGED_DIR.glob(PART_PATTERN))
    if not files:
        sys.exit(f"[FATAL] No CSVs matched {JUDGED_DIR / PART_PATTERN}")

    for fp in files:
        with fp.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            hdr = reader.fieldnames or []

            required = [COL_QID, COL_QUERY, COL_PASSAGE, COL_RELEVANCE]
            missing = [c for c in required if c not in hdr]
            if missing:
                print(f"[WARN] {fp.name}: missing required columns {missing}. Skipping.")
                continue

            has_qrels = COL_PID_QRELS in hdr
            has_res   = COL_PID_RES in hdr

            for row in reader:
                try:
                    rel = int((row.get(COL_RELEVANCE) or "0").strip())
                except ValueError:
                    rel = 0
                if rel != 0:
                    continue

                q = (row.get(COL_QUERY) or "").strip()
                if not q or q in first_zero:
                    continue

                payload = {
                    COL_QID:       (row.get(COL_QID) or "").strip(),
                    COL_QUERY:     q,
                    COL_PID_QRELS: (row.get(COL_PID_QRELS) or "").strip() if has_qrels else "",
                    COL_PID_RES:   (row.get(COL_PID_RES) or "").strip()   if has_res   else "",
                    COL_PASSAGE:   (row.get(COL_PASSAGE) or "").strip(),
                }
                if not payload[COL_PASSAGE]:
                    continue

                first_zero[q] = payload

    if not first_zero:
        sys.exit("[FATAL] No relevance==0 rows found in judged files.")

    # Write exactly: qid,query,pid_qrels,pid_resolved,passage
    out_fields = [COL_QID, COL_QUERY, COL_PID_QRELS, COL_PID_RES, COL_PASSAGE]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as out:
        w = csv.DictWriter(out, fieldnames=out_fields)
        w.writeheader()
        for _, payload in sorted(first_zero.items(), key=lambda kv: _to_int(kv[1][COL_QID])):
            w.writerow(payload)

    print(f"[DONE] Wrote {len(first_zero)} rows → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
