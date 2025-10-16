#!/usr/bin/env python3
from __future__ import annotations
import csv, sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# ====== EDIT PATHS IF NEEDED ======
SRC_FIRST_NONREL = Path("outputs/queries/first_nonrelevant_per_query.csv")
LLM_JUDGED_FILE  = Path("outputs/llm_label/gpt-oss-20b/gpt-oss-20b_trec_dl_2023_raw_with_ids.csv")
OUTPUT_CSV       = Path("outputs/queries/first_nonrelevant_with_llm_relevance.csv")
# ==================================

# Expected/alias columns
COL_QID         = "qid"
COL_TOPIC       = "topic"       # sometimes used instead of qid
COL_QUERY       = "query"
COL_PID_RES     = "pid_resolved"
COL_PID_QRELS   = "pid_qrels"
COL_DOCID       = "docid"
COL_DOC_ID      = "doc_id"
COL_PID         = "pid"
COL_PASSAGE     = "passage"
COL_RELEVANCE   = "relevance"
COL_LABEL       = "label"

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

def _get(row: dict, *names: str) -> str:
    for n in names:
        if n in row and row[n] is not None:
            v = str(row[n]).strip()
            if v != "":
                return v
    return ""

def _coalesce_pid(row: dict) -> str:
    # Use the same coalescing order on both sides
    return _get(row, COL_PID_RES, COL_PID_QRELS, COL_DOCID, COL_PID, COL_DOC_ID)

def _get_qid(row: dict) -> str:
    return _get(row, COL_QID, COL_TOPIC)

def _parse_rel(s: str) -> Optional[str]:
    s = (s or "").strip()
    if s in {"0", "1", "2", "3"}:
        return s
    # fallback: find a lone digit 0-3
    for ch in s:
        if ch in "0123":
            return ch
    return None

def load_first_nonrel(path: Path) -> Dict[Tuple[str, str], dict]:
    """
    Load rows from first_nonrelevant_per_query.csv (qid,query,pid_qrels,pid_resolved,passage)
    Return map: (qid, pid) -> payload {qid, query, pid, passage}
    """
    if not path.exists():
        sys.exit(f"[FATAL] Source list not found: {path}")
    by_key: Dict[Tuple[str, str], dict] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, skipinitialspace=True)
        hdr = r.fieldnames or []
        if COL_QUERY not in hdr or COL_PASSAGE not in hdr:
            sys.exit(f"[FATAL] Source must have at least ['{COL_QUERY}','{COL_PASSAGE}']. Header={hdr}")

        for row in r:
            if not row:
                continue
            qid   = _get_qid(row)
            pid   = _coalesce_pid(row)
            query = (row.get(COL_QUERY) or "").strip()
            passage = (row.get(COL_PASSAGE) or "").strip()
            if not qid or not pid:
                continue
            key = (qid, pid)
            # keep first occurrence
            if key not in by_key:
                by_key[key] = {
                    "qid": qid,
                    "query": query,
                    "pid": pid,
                    "passage": passage,
                }
    return by_key

def load_llm_labels(path: Path) -> Dict[Tuple[str, str], str]:
    """
    Load labels from the LLM judged file.
    Return map: (qid, pid) -> relevance (as '0'..'3'), first hit wins.
    """
    if not path.exists():
        sys.exit(f"[FATAL] LLM judged file not found: {path}")
    labels: Dict[Tuple[str, str], str] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, skipinitialspace=True)
        hdr = r.fieldnames or []
        # We need at least some id columns + a label-like column
        has_any_label = (COL_RELEVANCE in hdr) or (COL_LABEL in hdr)
        if not has_any_label:
            sys.exit(f"[FATAL] LLM judged file must have '{COL_RELEVANCE}' or '{COL_LABEL}'. Header={hdr}")

        for row in r:
            if not row:
                continue
            qid = _get_qid(row)
            pid = _coalesce_pid(row)
            if not qid or not pid:
                continue
            rel = _parse_rel(_get(row, COL_RELEVANCE, COL_LABEL))
            if rel is None:
                continue
            labels.setdefault((qid, pid), rel)

    return labels

def main():
    out_dir = OUTPUT_CSV.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    src_map = load_first_nonrel(SRC_FIRST_NONREL)
    lbl_map = load_llm_labels(LLM_JUDGED_FILE)

    total = len(src_map)
    matched = 0
    missing = 0

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=[COL_QID, COL_QUERY, "pid", COL_PASSAGE, COL_RELEVANCE])
        w.writeheader()

        # stable order: numeric qid if possible
        def _to_int(s: str) -> int:
            try: return int(s)
            except Exception: return 0

        for (qid, pid) in sorted(src_map.keys(), key=lambda k: (_to_int(k[0]), k[1])):
            base = src_map[(qid, pid)]
            rel = lbl_map.get((qid, pid))
            if rel is None:
                missing += 1
                rel = ""  # leave blank if not found
            else:
                matched += 1
            w.writerow({
                COL_QID: qid,
                COL_QUERY: base["query"],
                "pid": pid,
                COL_PASSAGE: base["passage"],
                COL_RELEVANCE: rel,
            })

    print(f"[DONE] wrote {OUTPUT_CSV}")
    print(f"[STATS] input rows={total} | matched labels={matched} | missing labels={missing}")

if __name__ == "__main__":
    main()
