#!/usr/bin/env python3
"""
Compare NIST (parts) vs LLM (raw combined) and save summary CSV.

- NIST: retrieved/trec_dl_2019/judged/all_topics_trecdl_2019_part*.csv
- LLM : outputs/llm_label/gpt_oss_20b_trec_dl_2019_raw.csv

Console + CSV outputs:
  - total docs compared (intersection)
  - more relevant (LLM > NIST)
  - less relevant  (LLM < NIST)
  - equal
  - Krippendorff's alpha (ordinal, graded)
  - Cohen's kappa (binary, <=1 -> 0, >1 -> 1)
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
TREC_DL_YEAR = "2023"  # '2019', '2020', or '2023'
NIST_DIR  = Path("retrieved/trec_dl_" + TREC_DL_YEAR + "/judged")
NIST_GLOB = "all_topics_trecdl_" + TREC_DL_YEAR + "_part*.csv"
LLM_CSV   = Path("outputs/llm_label/gpt-oss-20b/gpt-oss-20b_trec_dl_2023_raw_with_ids.csv")

# preferred output dir (use 'output/llm_label' if it exists, else 'outputs/llm_label')
PREF_OUT_DIR = Path("output/llm_label")
OUT_DIR = PREF_OUT_DIR if PREF_OUT_DIR.exists() else Path("outputs/llm_label")
OUT_CSV = OUT_DIR / "results_summary.csv"
allow_huge_csv_fields()


# --------- Helpers ---------
# --- replace _norm_headers ---
def _norm_headers(cols: List[str]) -> Dict[str, str]:
    """
    Normalize headers (lower, strip, remove BOM) and return a map
    normalized_name -> original_header_name, with useful aliases.
    """
    if not cols:
        return {}
    def kclean(s: str) -> str:
        return (s or "").strip().lstrip("\ufeff").lower()

    base = {kclean(c): c for c in cols}

    # canonical -> alternatives (normalized)
    aliases = {
        # passage/document ids
        "pid":   ("pid", "pid_resolved", "pid-qrels", "pid_qrels", "passage_id", "passageid", "passage-id"),
        "docid": ("docid", "docno", "doc_id", "documentid", "document_id"),
        # query ids / topic ids
        "qid":   ("qid", "topic", "topicid", "topic_id", "query_id"),
        "query": ("query", "query_text", "title"),
        # labels
        "relevance": ("relevance", "judgment", "judgement", "label", "grade"),
    }

    final_map = dict(base)
    for canon, alts in aliases.items():
        for alt in alts:
            if alt in base:
                final_map[canon] = base[alt]
                break

    return final_map



# --- replace _pick_key ---
def _pick_key(h: Dict[str, str]) -> Tuple[str, ...]:
    """
    Identifier columns for joining.
    For 2023, prefer (qid/query/topic, pid*) or just pid*.
    For other years, prefer (qid/query/topic, docid*) or just docid*.
    Accept pid* anywhere as a fallback too.
    """
    year = str(TREC_DL_YEAR)

    # find any available pid-like and docid-like originals
    pid_like   = [h[k] for k in ("pid", "pid_resolved", "pid_qrels") if k in h]
    docid_like = [h[k] for k in ("docid",) if k in h]  # 'docid' may already be aliased

    # helpers: first available column name
    pid_col   = pid_like[0] if pid_like else None
    docid_col = docid_like[0] if docid_like else None

    # paired keys in priority order
    if year == "2023":
        for a in ("query", "qid", "topic"):
            if a in h and pid_col:
                return (h[a], pid_col)
        if pid_col:
            return (pid_col,)

    # non-2023 preference
    for a in ("query", "qid", "topic"):
        if a in h and docid_col:
            return (h[a], docid_col)

    # broad fallbacks
    if docid_col:
        return (docid_col,)
    if pid_col:
        return (pid_col,)

    # last-resort debug
    raise KeyError(
        f"No key columns found. Have headers: {sorted(h.keys())}. "
        "Need pid/pid_resolved/pid_qrels or docid, or (query/qid/topic with one of those)."
    )



def _row_key(row: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple((row.get(k, "") or "").strip() for k in keys)

def _as_int(s):
    try: return int(str(s).strip())
    except: return None

def _rel_val(row: Dict[str, Any], h: Dict[str,str]) -> int | None:
    # prefer 'relevance', fall back to 'label'
    for name in ("relevance", "label"):
        if name in h:
            return _as_int(row.get(h[name], ""))
    return None

# --- replace your _load_csv open() line with this (utf-8-sig removes BOM) ---
def _load_csv(path: Path) -> Tuple[Tuple[str,...], Dict[Tuple[str,...], int], Counter]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            return (), {}, Counter()
        h = _norm_headers(rdr.fieldnames)
        kcols = _pick_key(h)
        rows: Dict[Tuple[str,...], int] = {}
        dups = Counter()
        for r in rdr:
            key = _row_key(r, kcols)
            rel = _rel_val(r, h)
            if key in rows:
                dups[key] += 1
            rows[key] = rel
        return kcols, rows, dups


def _binarize_threshold(vals: List[int], thr: int = 1) -> List[int]:
    # <=thr -> 0,  >thr -> 1
    return [0 if v <= thr else 1 for v in vals]

# --------- Main ---------
def main():
    if not LLM_CSV.exists():
        raise FileNotFoundError(f"Missing LLM CSV: {LLM_CSV}")

    # Load LLM (combined)
    llm_kcols, llm_map, llm_dups = _load_csv(LLM_CSV)

    # Load and union NIST (all parts)
    nist_files = sorted(NIST_DIR.glob(NIST_GLOB))
    if not nist_files:
        raise FileNotFoundError(f"No NIST part files in {NIST_DIR} matching {NIST_GLOB}")

    nist_map: Dict[Tuple[str,...], int] = {}
    nist_dups_total = Counter()
    nist_kcols_ref: Tuple[str,...] | None = None

    for pf in nist_files:
        kcols, m, dups = _load_csv(pf)
        nist_dups_total.update(dups)
        if nist_kcols_ref is None:
            nist_kcols_ref = kcols
        elif nist_kcols_ref != kcols:
            print(f"[WARN] Key columns differ in {pf.name}: {kcols} vs {nist_kcols_ref}")
        nist_map.update(m)  # later files win

    # Intersection keys
    inter = sorted(set(llm_map.keys()) & set(nist_map.keys()))

    # Collect graded labels (skip Nones)
    nist_vals: List[int] = []
    llm_vals:  List[int] = []
    more = less = equal = 0  # relative to NIST baseline

    for k in inter:
        n = nist_map.get(k)
        l = llm_map.get(k)
        if n is None or l is None:
            continue
        nist_vals.append(n)
        llm_vals.append(l)
        if l > n:  more += 1       # LLM > NIST
        elif l < n: less += 1      # LLM < NIST
        else:       equal += 1

    total = len(nist_vals)

    if total == 0:
        print("No overlapping comparable rows with numeric relevance.")
        return

    # Metrics
    alpha_ord = kd.alpha(
        reliability_data=np.array([nist_vals, llm_vals]),
        level_of_measurement="ordinal"
    )

    nist_bin = _binarize_threshold(nist_vals, thr=1)
    llm_bin  = _binarize_threshold(llm_vals,  thr=1)
    kappa_bin = cohen_kappa_score(nist_bin, llm_bin)

    # Console
    print("=" * 68)
    print(f"Compared NIST (parts, {len(nist_files)} files) vs LLM ({LLM_CSV.name})")
    print(f"Key cols NIST: {nist_kcols_ref}")
    print(f"Key cols LLM : {llm_kcols}")
    print("-" * 68)
    print(f"Total docs compared         : {total}")
    print(f"More relevant (LLM > NIST)  : {more}")
    print(f"Less relevant (LLM < NIST)  : {less}")
    print(f"Equal                       : {equal}")
    print("-" * 68)
    print(f"Krippendorff's alpha (ordinal, graded): {alpha_ord:.4f}")
    print(f"Cohen's kappa (binary thr=1)          : {kappa_bin:.4f}")
    print("=" * 68)

    # ---- Save CSV summary ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_CSV.exists()  # append-friendly: write header if new
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
            len(nist_files),
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
