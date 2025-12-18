#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# -----------------------
# Config (edit if needed)
# -----------------------
THIS_DIR = Path(__file__).resolve().parent
VI_PATH  = THIS_DIR / "query_map_vi_corrected.csv"
KO_PATH  = THIS_DIR / "query_map_ko.csv"
OUT_PATH = THIS_DIR / "query_map_vi_ko.csv"

# Candidate column names (case-insensitive)
ID_CANDIDATES = ["query"]
TRANSLATED_CANDIDATES = ["translated"]

# -----------------------
# Helpers
# -----------------------
def norm(s: str) -> str:
    return (s or "").strip().lower()

def pick_col(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    """
    Return the *actual* column name from fieldnames that matches any candidate (case-insensitive).
    """
    fn_map = {norm(f): f for f in fieldnames}
    for c in candidates:
        if norm(c) in fn_map:
            return fn_map[norm(c)]
    return None

def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    # utf-8-sig handles BOM (common when CSV saved from Excel)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row.")
        rows = []
        for r in reader:
            # Ensure all values are strings (DictReader can return None)
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
        return list(reader.fieldnames), rows

def build_map(
    rows: List[Dict[str, str]],
    id_col: str,
    translated_col: str,
    label: str,
) -> Dict[str, str]:
    """
    Build id -> translated mapping. Warn on duplicates (keeps last non-empty).
    """
    m: Dict[str, str] = {}
    dup_count = 0
    for r in rows:
        key = (r.get(id_col) or "").strip()
        val = (r.get(translated_col) or "").strip()
        if not key:
            continue
        if key in m and val and m[key] != val:
            dup_count += 1
        # Keep last non-empty if possible
        if val:
            m[key] = val
        else:
            m.setdefault(key, "")
    if dup_count:
        print(f"[WARN] {label}: found {dup_count} duplicate {id_col} with differing translations.", file=sys.stderr)
    return m

def main() -> int:
    if not VI_PATH.exists():
        print(f"[ERROR] Missing: {VI_PATH}", file=sys.stderr)
        return 2
    if not KO_PATH.exists():
        print(f"[ERROR] Missing: {KO_PATH}", file=sys.stderr)
        return 2

    vi_fields, vi_rows = read_csv_rows(VI_PATH)
    ko_fields, ko_rows = read_csv_rows(KO_PATH)

    vi_id = pick_col(vi_fields, ID_CANDIDATES)
    ko_id = pick_col(ko_fields, ID_CANDIDATES)
    if not vi_id or not ko_id:
        print(f"[ERROR] Could not auto-detect ID column.\n"
              f"  VI fields: {vi_fields}\n  KO fields: {ko_fields}\n"
              f"Edit ID_CANDIDATES or set vi_id/ko_id manually.", file=sys.stderr)
        return 2

    vi_tr = pick_col(vi_fields, TRANSLATED_CANDIDATES)
    ko_tr = pick_col(ko_fields, TRANSLATED_CANDIDATES)
    if not vi_tr or not ko_tr:
        print(f"[ERROR] Could not auto-detect translated column.\n"
              f"  VI fields: {vi_fields}\n  KO fields: {ko_fields}\n"
              f"Edit TRANSLATED_CANDIDATES or set vi_tr/ko_tr manually.", file=sys.stderr)
        return 2

    vi_map = build_map(vi_rows, vi_id, vi_tr, "VI")
    ko_map = build_map(ko_rows, ko_id, ko_tr, "KO")

    all_ids = sorted(set(vi_map.keys()) | set(ko_map.keys()), key=lambda x: (len(x), x))

    out_fields = ["qid", "translated_vi", "translated_ko"]
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for qid in all_ids:
            w.writerow({
                "qid": qid,
                "translated_vi": vi_map.get(qid, ""),
                "translated_ko": ko_map.get(qid, ""),
            })

    print(f"[OK] Wrote: {OUT_PATH}")
    print(f"[INFO] VI rows: {len(vi_rows)} | KO rows: {len(ko_rows)} | merged ids: {len(all_ids)}")
    print(f"[INFO] Missing VI for {sum(1 for i in all_ids if not vi_map.get(i,''))} ids; "
          f"Missing KO for {sum(1 for i in all_ids if not ko_map.get(i,''))} ids.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
