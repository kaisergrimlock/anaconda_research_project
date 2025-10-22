#!/usr/bin/env python3
from __future__ import annotations
import csv, sys
from pathlib import Path

# =========================
# CONFIG — edit these
# =========================
TREC_DL_YEAR = "2023"
MODEL        = "gpt-oss-20b"
LANG         = "eng"   # "eng", "vi", etc.

LLM_FILE = Path("outputs/llm_label") / MODEL / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{LANG}_raw.csv"
TOPICS_DIR  = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / LANG
TOPICS_GLOB = f"all_topics_trecdl_{TREC_DL_YEAR}_part*.csv"

OUT_DIR             = Path("outputs") / "llm_label" / MODEL
TOPICS_NOT_IN_LLM   = OUT_DIR / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{LANG}_topics_not_in_llm.csv"
LLM_NOT_IN_TOPICS   = OUT_DIR / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{LANG}_llm_not_in_topics.csv"
WRITE_INVERSE_CHECK = True
# =========================

# Desired column order for topics_not_in_llm.csv
TOPICS_OUT_HEADERS = [
    "qid","query","pid_qrels","pid_resolved","passage","relevance","query_eng","passage_injected"
]

def _bump_field_limit():
    limit = min(2_000_000_000, getattr(sys, "maxsize", 2_000_000_000))
    while limit >= 131072:
        try:
            csv.field_size_limit(limit); return
        except OverflowError:
            limit //= 10
_bump_field_limit()

def _pick(cols, *cands):
    s = {c.lower() for c in cols}
    for group in cands:
        for c in group:
            if c.lower() in s:
                return c
    return None

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def load_llm_pairs(fp: Path):
    """Return: (pair_set, rows_by_pair, header). Pair is (pid, norm(passage_eng))."""
    with fp.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        hdr = [h.strip() for h in (r.fieldnames or [])]
        pid_col = _pick(hdr, ("pid",))
        pen_col = _pick(hdr, ("passage_eng","passage_injected","passage_en"))
        if not pid_col or not pen_col:
            raise RuntimeError("LLM file needs 'pid' and 'passage_eng' (or alias).")

        pair_set, rows_by_pair = set(), {}
        total = 0
        for row in r:
            total += 1
            pid = (row.get(pid_col) or "").strip()
            pe  = _norm(row.get(pen_col) or "")
            if pid and pe:
                key = (pid, pe)
                pair_set.add(key)
                rows_by_pair.setdefault(key, row)
        print(f"[llm] rows scanned: {total:,}; unique (pid,passage_eng): {len(pair_set):,}")
        return pair_set, rows_by_pair, hdr

def load_topics_pairs(dirpath: Path, pattern: str):
    """
    Return: (pair_set, rows_by_pair)
    Pair is (pid_resolved|pid_qrels, norm(passage_injected)).
    Also retain fields so we can write rows in TOPICS_OUT_HEADERS order.
    """
    files = sorted(dirpath.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No topic files matching {pattern!r} in {dirpath}")

    pair_set, rows_by_pair = set(), {}
    total = 0
    for fp in files:
        with fp.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            hdr = [h.strip() for h in (r.fieldnames or [])]
            if not hdr:
                continue

            qid_col   = _pick(hdr, ("qid","topic_id"))
            pid_res   = _pick(hdr, ("pid_resolved",))
            pid_q     = _pick(hdr, ("pid_qrels",))
            inj_col   = _pick(hdr, ("passage_injected","passage_eng"))
            # optional but in your parts:
            pass_col  = _pick(hdr, ("passage",))
            rel_col   = _pick(hdr, ("relevance",))
            query_col = _pick(hdr, ("query",))
            qeng_col  = _pick(hdr, ("query_eng",))

            if not inj_col or not (pid_res or pid_q):
                continue

            for row in r:
                total += 1
                pid = (row.get(pid_res) or row.get(pid_q) or "").strip()
                inj = _norm(row.get(inj_col) or "")
                if not pid or not inj:
                    continue
                key = (pid, inj)
                if key in pair_set:
                    continue
                pair_set.add(key)

                # Build a normalized record in the requested output format
                rec = {
                    "qid":               (row.get(qid_col) or "") if qid_col else "",
                    "query":             (row.get(query_col) or "") if query_col else "",
                    "pid_qrels":         (row.get(pid_q) or "") if pid_q else "",
                    "pid_resolved":      (row.get(pid_res) or "") if pid_res else "",
                    "passage":           (row.get(pass_col) or "") if pass_col else "",
                    "relevance":         (row.get(rel_col) or "") if rel_col else "",
                    "query_eng":         (row.get(qeng_col) or "") if qeng_col else "",
                    "passage_injected":  (row.get(inj_col) or "") if inj_col else "",
                }
                rows_by_pair[key] = rec

    print(f"[topics] files: {len(files)}; rows scanned: {total:,}; unique (pid,passage_injected): {len(pair_set):,}")
    return pair_set, rows_by_pair

def write_topics_not_in_llm(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TOPICS_OUT_HEADERS)
        w.writeheader()
        for r in rows:
            # ensure all columns exist
            out = {k: r.get(k, "") for k in TOPICS_OUT_HEADERS}
            w.writerow(out)
    print(f"[write] {path}  rows={len(rows):,}")

def write_llm_not_in_topics(path: Path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep original LLM columns
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {path}  rows={len(rows):,}")

def main():
    if not LLM_FILE.exists():
        raise FileNotFoundError(f"LLM file not found: {LLM_FILE}")
    if not TOPICS_DIR.exists():
        raise FileNotFoundError(f"Topics dir not found: {TOPICS_DIR}")

    llm_pairs, llm_rows_by_pair, llm_hdr = load_llm_pairs(LLM_FILE)
    topics_pairs, topics_rows_by_pair = load_topics_pairs(TOPICS_DIR, TOPICS_GLOB)

    # topics missing in LLM (by (pid, normalized passage))
    missing_in_llm_keys = topics_pairs - llm_pairs
    missing_rows = [topics_rows_by_pair[k] for k in missing_in_llm_keys]
    write_topics_not_in_llm(TOPICS_NOT_IN_LLM, missing_rows)

    if WRITE_INVERSE_CHECK:
        llm_not_in_topics_keys = llm_pairs - topics_pairs
        rows = [llm_rows_by_pair[k] for k in llm_not_in_topics_keys]
        write_llm_not_in_topics(LLM_NOT_IN_TOPICS, rows, llm_hdr)

    print("\nSummary:")
    print(f"  topics unique pairs .......... {len(topics_pairs):,}")
    print(f"  llm unique pairs ............. {len(llm_pairs):,}")
    print(f"  topics NOT in llm ............ {len(missing_in_llm_keys):,}")
    if WRITE_INVERSE_CHECK:
        print(f"  llm NOT in topics ............ {len(llm_not_in_topics_keys):,}")

if __name__ == "__main__":
    main()
