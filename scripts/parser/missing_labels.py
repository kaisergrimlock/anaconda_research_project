#!/usr/bin/env python3
"""
Emit all docs that are present in NIST parts but missing from the LLM raw CSV,
in the exact format:

query,docid,passage,relevance

- NIST parts: retrieved/trec_dl_2019/judged/all_topics_trecdl_2019_part*.csv
- LLM raw   : outputs/llm_label/gpt_oss_20b_trec_dl_2019_raw.csv

Writes:
  outputs/llm_label/missing_from_llm_formatted.csv
"""

from __future__ import annotations
from pathlib import Path
import csv
from typing import Dict, Tuple, Any, List

# -------- Paths --------
NIST_DIR  = Path("retrieved/trec_dl_2019/judged")
NIST_GLOB = "all_topics_trecdl_2019_part*.csv"
LLM_CSV   = Path("outputs/llm_label/gpt_oss_20b_trec_dl_2019_raw.csv")

OUT_PATH  = Path("outputs/llm_label/missing_from_llm_formatted.csv")

# -------- Helpers --------
def _norm_headers(cols: List[str]) -> Dict[str, str]:
    """Map logical lowercased names to actual header names."""
    return {c.lower().strip(): c for c in (cols or [])}

def _pick_key(h: Dict[str, str]) -> Tuple[str, ...]:
    """Key priority: (query,docid) -> (topic,docid) -> (qid,docid) -> (docid,)"""
    for a, b in [("query", "docid"), ("topic", "docid"), ("qid", "docid")]:
        if a in h and b in h:
            return (h[a], h[b])
    if "docid" in h: return (h["docid"],)
    if "doc"   in h: return (h["doc"],)
    if "id"    in h: return (h["id"],)
    raise KeyError("No suitable key columns (need docid or (query/topic/qid, docid)).")

def _row_key(row: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple((row.get(k, "") or "").strip() for k in keys)

def _as_int(s):
    try:
        return int(str(s).strip())
    except Exception:
        return None

def _get_field(row: Dict[str, Any], h: Dict[str, str], logical: str) -> str:
    """Return a common logical field if present, else empty string."""
    if logical in h:
        return (row.get(h[logical], "") or "").strip()
    return ""

def _rel_val_str(row: Dict[str, Any], h: Dict[str, str]) -> str:
    """Prefer 'relevance', fall back to 'label', return as string ('' if missing)."""
    for name in ("relevance", "label"):
        if name in h:
            v = (row.get(h[name], "") or "").strip()
            # keep as-is (string) but normalize numeric-looking values
            if v == "":
                return ""
            try:
                return str(int(v))
            except Exception:
                return v  # keep original if non-numeric
    return ""

def _load_csv_map(path: Path):
    """Return (keycols, row_map, header_map)."""
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

    # Load LLM (for presence only)
    llm_kcols, llm_map, llm_h = _load_csv_map(LLM_CSV)

    # Load + union NIST parts
    nist_files = sorted(NIST_DIR.glob(NIST_GLOB))
    if not nist_files:
        raise FileNotFoundError(f"No part files in {NIST_DIR} matching {NIST_GLOB}")

    nist_map = {}
    nist_h_ref = None
    nist_kcols_ref = None

    for pf in nist_files:
        kcols, mm, hh = _load_csv_map(pf)
        if nist_h_ref is None:
            nist_h_ref = hh
            nist_kcols_ref = kcols
        # later files override earlier on same key
        nist_map.update(mm)

    # Determine missing keys (present in NIST but not in LLM)
    missing_keys = sorted(set(nist_map.keys()) - set(llm_map.keys()))

    # Prepare rows in required order/headers
    # We will output query text even if source used 'topic' or 'qid'—we'll map to 'query'.
    rows_out = []
    for k in missing_keys:
        n_row = nist_map[k]
        # query text: prefer 'query', then 'topic', then 'qid'
        query_txt = (
            _get_field(n_row, nist_h_ref, "query")
            or _get_field(n_row, nist_h_ref, "topic")
            or _get_field(n_row, nist_h_ref, "qid")
        )
        docid = (
            _get_field(n_row, nist_h_ref, "docid")
            or _get_field(n_row, nist_h_ref, "doc")
            or _get_field(n_row, nist_h_ref, "id")
        )
        passage = _get_field(n_row, nist_h_ref, "passage")
        relevance = _rel_val_str(n_row, nist_h_ref)

        rows_out.append([query_txt, docid, passage, relevance])

    # Write CSV (exact header & order)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "docid", "passage", "relevance"])
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} missing docs to: {OUT_PATH}")

if __name__ == "__main__":
    main()
