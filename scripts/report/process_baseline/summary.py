#!/usr/bin/env python3
"""
Compare NIST (parts) vs LLM (raw combined) and save summary CSV.

Now joins LLM rows with qid from retrieved/<LANG> parts, then compares on (qid, pidlike).
"""

from __future__ import annotations
from pathlib import Path
import csv
from collections import Counter
from typing import Dict, Tuple, List, Any
from datetime import datetime
import numpy as np
from sklearn.metrics import cohen_kappa_score   # pip install scikit-learn
import krippendorff as kd                       # pip install krippendorff
from helper import allow_huge_csv_fields

# --------- Paths ---------
TREC_DL_YEAR = "2023"   # '2019', '2020', or '2023'
LANG = "fr"

# NIST baseline (judged parts)
NIST_DIR   = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}/judged")
NIST_GLOB  = f"all_topics_trecdl_{TREC_DL_YEAR}_part*.csv"

# Retrieved parts for qid lookup (LANG folder)
TOPICS_DIR  = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / LANG
TOPICS_GLOB = f"all_topics_trecdl_{TREC_DL_YEAR}_part*.csv"

# LLM combined file (raw)
LLM_CSV = Path(f"outputs/llm_label/gpt-oss-20b/gpt-oss-20b_trec_dl_{TREC_DL_YEAR}_{LANG}_raw.csv")

# preferred output dir (use 'output/llm_label' if it exists, else 'outputs/llm_label')
PREF_OUT_DIR = Path("output/llm_label")
OUT_DIR = PREF_OUT_DIR if PREF_OUT_DIR.exists() else Path("outputs/llm_label")
OUT_CSV = OUT_DIR / "results_summary.csv"
allow_huge_csv_fields()


# --------- Header normalization ---------
def _kclean(s: str) -> str:
    return (s or "").strip().lstrip("\ufeff").lower()

def _norm_headers(cols: List[str]) -> Dict[str, str]:
    """
    Normalize headers to canonical keys. Return map canonical -> original.
    Canonical keys: qid, query, pid, pid_resolved, pid_qrels, docid, relevance, label
    """
    if not cols:
        return {}
    base = {_kclean(c): c for c in cols}
    # Build a map from canonical to original header if present
    def pick(*names: str) -> str | None:
        for n in names:
            nn = _kclean(n)
            if nn in base:
                return base[nn]
        return None

    canon = dict(base)  # keep originals accessible too
    # explicit canonical aliases
    mapping = {
        "qid":         pick("qid", "topic", "topicid", "topic_id", "query_id"),
        "query":       pick("query", "query_text", "title"),
        "pid":         pick("pid", "passage_id", "passageid", "passage-id"),
        "pid_resolved":pick("pid_resolved"),
        "pid_qrels":   pick("pid_qrels","pid-qrels"),
        "docid":       pick("docid", "docno", "doc_id", "documentid", "document_id"),
        "relevance":   pick("relevance","judgment","judgement","label","grade"),
        "label":       pick("label"),  # kept for _rel_val fallback
    }
    for k, v in mapping.items():
        if v:
            canon[k] = v
    return canon


# --------- CSV loading helpers ---------
def _row_key(row: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple((row.get(k, "") or "").strip() for k in keys)

def _as_int(s):
    try:
        return int(str(s).strip())
    except:
        return None

def _rel_val(row: Dict[str, Any], h: Dict[str,str]) -> int | None:
    for name in ("relevance", "label"):
        if name in h:
            return _as_int(row.get(h[name], ""))
    return None

def _load_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (original fieldnames, list of row dicts) with utf-8-sig."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        return (rdr.fieldnames or []), list(rdr)


# --------- Build pidlike -> qid index from retrieved/<LANG> parts ---------
def build_pid_to_qid_index() -> tuple[Dict[str, str], Counter]:
    """
    From retrieved/<LANG>/all_topics_trecdl_* parts, build a map:
      pidlike (pid | pid_resolved | pid_qrels | docid) -> qid
    If multiple qids appear for the same pidlike, last one wins; duplicates counted.
    """
    files = sorted(TOPICS_DIR.glob(TOPICS_GLOB))
    if not files:
        raise FileNotFoundError(f"No retrieved LANG part files in {TOPICS_DIR} matching {TOPICS_GLOB}")

    idx: Dict[str, str] = {}
    dupe_stats = Counter()

    for pf in files:
        _, rows = _load_csv(pf)
        if not rows:
            continue
        h = _norm_headers(list(rows[0].keys()))
        qid_col = h.get("qid")
        if not qid_col:
            raise KeyError(f"{pf.name}: missing qid column")

        # pick any pidlike present in this file
        pid_cols = [h.get(k) for k in ("pid", "pid_resolved", "pid_qrels", "docid") if h.get(k)]
        if not pid_cols:
            raise KeyError(f"{pf.name}: missing any of pid|pid_resolved|pid_qrels|docid")

        for r in rows:
            qid = (r.get(qid_col, "") or "").strip()
            if not qid:
                continue
            for pc in pid_cols:
                pidv = (r.get(pc, "") or "").strip()
                if not pidv:
                    continue
                if pidv in idx and idx[pidv] != qid:
                    dupe_stats["conflict_pid_to_qid"] += 1
                idx[pidv] = qid
    return idx, dupe_stats


# --------- Load NIST judged parts into (qid, pidlike) -> rel ---------
def load_nist_map() -> tuple[Dict[Tuple[str, str], int], Counter]:
    files = sorted(NIST_DIR.glob(NIST_GLOB))
    if not files:
        raise FileNotFoundError(f"No NIST part files in {NIST_DIR} matching {NIST_GLOB}")

    nmap: Dict[Tuple[str, str], int] = {}
    dups = Counter()

    for pf in files:
        _, rows = _load_csv(pf)
        if not rows:
            continue
        h = _norm_headers(list(rows[0].keys()))

        qid_col = h.get("qid")
        if not qid_col:
            raise KeyError(f"{pf.name}: missing qid column")

        # prefer pid_resolved/pid_qrels/pid/docid in that order
        pid_col = h.get("pid_resolved") or h.get("pid_qrels") or h.get("pid") or h.get("docid")
        if not pid_col:
            raise KeyError(f"{pf.name}: missing any of pid_resolved|pid_qrels|pid|docid")

        for r in rows:
            qid = (r.get(qid_col, "") or "").strip()
            pid = (r.get(pid_col, "") or "").strip()
            if not qid or not pid:
                continue
            rel = _rel_val(r, h)
            key = (qid, pid)
            if key in nmap:
                dups["nist_dupe_keys"] += 1
            nmap[key] = rel
    return nmap, dups


# --------- Load LLM file and enrich with qid via index ---------
def load_llm_with_qid(pid_to_qid: Dict[str, str]) -> tuple[Dict[Tuple[str, str], int], Counter]:
    if not LLM_CSV.exists():
        raise FileNotFoundError(f"Missing LLM CSV: {LLM_CSV}")

    _, rows = _load_csv(LLM_CSV)
    if not rows:
        return {}, Counter()

    h = _norm_headers(list(rows[0].keys()))
    relmap: Dict[Tuple[str, str], int] = {}
    stats = Counter()

    # find pid/docid candidates present in the LLM file
    pid_cols = [h.get(k) for k in ("pid", "docid") if h.get(k)]
    if not pid_cols:
        raise KeyError(f"{LLM_CSV.name}: missing pid/docid")

    for r in rows:
        rel = _rel_val(r, h)
        if rel is None:
            stats["llm_missing_rel"] += 1
            continue

        # try both pid and docid to look up qid
        pid_candidates = []
        for pc in pid_cols:
            v = (r.get(pc, "") or "").strip()
            if v:
                pid_candidates.append(v)

        if not pid_candidates:
            stats["llm_missing_pidlike"] += 1
            continue

        # choose the first candidate that maps to a qid
        qid = None
        pid_used = None
        for pidv in pid_candidates:
            qid = pid_to_qid.get(pidv)
            if qid:
                pid_used = pidv
                break

        if not qid:
            stats["llm_pid_not_found_in_index"] += 1
            continue

        key = (qid, pid_used)
        if key in relmap:
            stats["llm_dupe_keys"] += 1
        relmap[key] = rel

    return relmap, stats


# --------- Main ---------
def main():
    # 1) Build pidlike -> qid index from retrieved/<LANG> parts
    pid_to_qid, idx_stats = build_pid_to_qid_index()

    # 2) Load NIST judged parts as (qid, pidlike) -> rel
    nist_map, nist_stats = load_nist_map()

    # 3) Load LLM file, enrich with qid via index, (qid, pidlike) -> rel
    llm_map, llm_stats = load_llm_with_qid(pid_to_qid)

    # 4) Intersect keys
    inter = sorted(set(llm_map.keys()) & set(nist_map.keys()))
    if not inter:
        print("No overlapping comparable rows with numeric relevance after qid join.")
        print(f"Index size: {len(pid_to_qid)}, LLM keyed: {len(llm_map)}, NIST keyed: {len(nist_map)}")
        print(f"Index stats: {dict(idx_stats)}")
        print(f"LLM stats  : {dict(llm_stats)}")
        print(f"NIST stats : {dict(nist_stats)}")
        return

    # 5) Collect graded labels
    nist_vals, llm_vals = [], []
    more = less = equal = 0
    for k in inter:
        n = nist_map.get(k)
        l = llm_map.get(k)
        if n is None or l is None:
            continue
        nist_vals.append(n)
        llm_vals.append(l)
        if l > n:  more += 1
        elif l < n: less += 1
        else:       equal += 1

    total = len(nist_vals)
    if total == 0:
        print("Overlap exists, but no rows with numeric relevance.")
        return

    # 6) Metrics
    alpha_ord = kd.alpha(
        reliability_data=np.array([nist_vals, llm_vals]),
        level_of_measurement="ordinal"
    )
    def _binarize_threshold(vals: List[int], thr: int = 1) -> List[int]:
        return [0 if v <= thr else 1 for v in vals]
    kappa_bin = cohen_kappa_score(_binarize_threshold(nist_vals), _binarize_threshold(llm_vals))

    # 7) Console
    print("=" * 72)
    print(f"Compared NIST (judged parts) vs LLM ({LLM_CSV.name}) with qid-join via {TOPICS_DIR.name}/parts")
    print(f"NIST keyed rows : {len(nist_map)}   LLM keyed rows : {len(llm_map)}   Index size : {len(pid_to_qid)}")
    if idx_stats:  print(f"Index stats    : {dict(idx_stats)}")
    if llm_stats:  print(f"LLM stats      : {dict(llm_stats)}")
    if nist_stats: print(f"NIST stats     : {dict(nist_stats)}")
    print("-" * 72)
    print(f"Total docs compared         : {total}")
    print(f"More relevant (LLM > NIST)  : {more}")
    print(f"Less relevant (LLM < NIST)  : {less}")
    print(f"Equal                       : {equal}")
    print("-" * 72)
    print(f"Krippendorff's alpha (ordinal, graded): {alpha_ord:.4f}")
    print(f"Cohen's kappa (binary thr=1)          : {kappa_bin:.4f}")
    print("=" * 72)

    # 8) Save CSV summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_CSV.exists()
    with OUT_CSV.open("a", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        if write_header:
            w.writerow([
                "timestamp",
                "nist_parts_files",
                "llm_file",
                "total_docs",
                "more_relevant_llm_gt_nist",
                "less_relevant_llm_lt_nist",
                "equal",
                "krippendorff_alpha_ordinal",
                "cohen_kappa_binary_thr1"
            ])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            len(sorted(NIST_DIR.glob(NIST_GLOB))),
            LLM_CSV.name,
            total,
            more,
            less,
            equal,
            f"{alpha_ord:.6f}",
            f"{kappa_bin:.6f}",
        ])
    print(f"Saved summary -> {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
