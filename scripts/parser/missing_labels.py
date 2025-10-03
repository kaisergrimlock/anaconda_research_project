#!/usr/bin/env python3
"""
Output all docs that are either:
  - missing from the LLM raw file, OR
  - present in LLM but have missing/non-numeric relevance

Format (exactly):
  query,docid,passage,relevance

Sources:
  NIST parts: retrieved/trec_dl_2019/judged/all_topics_trecdl_2019_part*.csv
  LLM raw   : outputs/llm_label/gpt_oss_20b_trec_dl_2019_raw.csv
"""

from __future__ import annotations
from pathlib import Path
import csv
from typing import Dict, Tuple, Any, List

# -------- Paths --------
NIST_DIR  = Path("retrieved/trec_dl_2019/judged")
NIST_GLOB = "all_topics_trecdl_2019_part*.csv"
LLM_CSV   = Path("outputs/llm_label/gpt_oss_20b_trec_dl_2019_raw.csv")
OUT_PATH  = Path("outputs/llm_label/missing_or_unjudged_formatted.csv")

# -------- Helpers --------
def _norm_headers(cols: List[str]) -> Dict[str, str]:
    return {c.lower().strip(): c for c in (cols or [])}

def _pick_key(h: Dict[str, str]) -> Tuple[str, ...]:
    for a, b in [("query","docid"), ("topic","docid"), ("qid","docid")]:
        if a in h and b in h:
            return (h[a], h[b])
    if "docid" in h: return (h["docid"],)
    if "doc"   in h: return (h["doc"],)
    if "id"    in h: return (h["id"],)
    raise KeyError("No suitable key columns (need docid or (query/topic/qid, docid)).")

def _row_key(row: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple((row.get(k, "") or "").strip() for k in keys)

def _get(row: Dict[str, Any], h: Dict[str,str], logical: str) -> str:
    if logical in h:
        return (row.get(h[logical], "") or "").strip()
    return ""

def _rel_as_str(row: Dict[str, Any], h: Dict[str,str]) -> str:
    # prefer 'relevance', fall back to 'label'
    for name in ("relevance", "label"):
        if name in h:
            v = (row.get(h[name], "") or "").strip()
            if v == "":
                return ""
            try:
                return str(int(v))
            except Exception:
                return v
    return ""

def _rel_is_missing_in_llm(row: Dict[str, Any], h: Dict[str,str]) -> bool:
    for name in ("relevance", "label"):
        if name in h:
            v = (row.get(h[name], "") or "").strip()
            if v == "":
                return True
            try:
                int(v)
                return False
            except Exception:
                return True
    # if neither column exists, treat as missing
    return True

def _load_csv_map(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            return (), {}, {}
        h = _norm_headers(rdr.fieldnames)
        kcols = _pick_key(h)
        mm = {}
        for r in rdr:
            mm[_row_key(r, kcols)] = r
        return kcols, mm, h

# -------- Main --------
def main():
    if not LLM_CSV.exists():
        raise FileNotFoundError(f"LLM CSV not found: {LLM_CSV}")

    # Load LLM (presence + relevance check)
    llm_kcols, llm_map, llm_h = _load_csv_map(LLM_CSV)

    # Load + union NIST parts
    nist_files = sorted(NIST_DIR.glob(NIST_GLOB))
    if not nist_files:
        raise FileNotFoundError(f"No part files in {NIST_DIR} matching {NIST_GLOB}")

    nist_map = {}
    nist_h = None
    for pf in nist_files:
        kcols, mm, hh = _load_csv_map(pf)
        if nist_h is None:
            nist_h = hh
        nist_map.update(mm)  # later files override

    # Determine keys to include:
    #  - Missing from LLM: in NIST but not in LLM
    #  - Unjudged in LLM: in both, but LLM relevance is blank/non-numeric
    nist_keys = set(nist_map.keys())
    llm_keys  = set(llm_map.keys())

    missing_keys  = nist_keys - llm_keys
    overlap_keys  = nist_keys & llm_keys
    unjudged_keys = {k for k in overlap_keys if _rel_is_missing_in_llm(llm_map[k], llm_h)}

    target_keys = sorted(missing_keys | unjudged_keys)

    # Emit rows with NIST-side data, formatted exactly
    rows_out: List[List[str]] = []
    for k in target_keys:
        n_row = nist_map[k]
        query   = _get(n_row, nist_h, "query") or _get(n_row, nist_h, "topic") or _get(n_row, nist_h, "qid")
        docid   = _get(n_row, nist_h, "docid") or _get(n_row, nist_h, "doc") or _get(n_row, nist_h, "id")
        passage = _get(n_row, nist_h, "passage")
        rel     = _rel_as_str(n_row, nist_h)
        rows_out.append([query, docid, passage, rel])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "docid", "passage", "relevance"])
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows -> {OUT_PATH}")

if __name__ == "__main__":
    main()
