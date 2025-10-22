#!/usr/bin/env python3
from __future__ import annotations
import csv, sys
from pathlib import Path

# =========================
# CONFIG — edit these
# =========================
TREC_DL_YEAR = "2023"
MODEL        = "gpt-oss-20b"
LANG         = "vi"   # "eng", "vi", etc.

# LLM file (the single CSV you labeled)
LLM_FILE = Path("outputs/llm_label") / MODEL / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{LANG}_raw.csv"

# Topics folder (your “eng” folder with many all_topics_* parts)
TOPICS_DIR  = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / LANG
TOPICS_GLOB = f"all_topics_trecdl_{TREC_DL_YEAR}_part*.csv"

# Outputs
OUT_DIR             = Path("outputs") / "llm_label" / MODEL
TOPICS_NOT_IN_LLM   = OUT_DIR / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{LANG}_topics_not_in_llm.csv"
LLM_NOT_IN_TOPICS   = OUT_DIR / f"{MODEL}_trec_dl_{TREC_DL_YEAR}_{LANG}_llm_not_in_topics.csv"  # sanity check
WRITE_INVERSE_CHECK = True  # set False if you only want the first report
# =========================


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
    """Return: (pair_set, rows_by_pair). Pair is (pid, norm(passage_eng))."""
    with fp.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        hdr = [h.strip() for h in (r.fieldnames or [])]
        pid_col = _pick(hdr, ("pid",))
        pen_col = _pick(hdr, ("passage_" + LANG,"passage_injected"))
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
    Return: (pair_set, rows_by_pair, merged_header)
    Pair is (pid_resolved|pid_qrels, norm(passage_injected)).
    We keep useful columns (qid, query, pid_qrels, pid_resolved, passage_injected, query_eng).
    """
    files = sorted(dirpath.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No topic files matching {pattern!r} in {dirpath}")

    keep_cols = ["qid","query","query_eng","pid_qrels","pid_resolved","passage_injected"]
    pair_set, rows_by_pair = set(), {}
    merged_header = list(dict.fromkeys(keep_cols))  # preserve order

    total = 0
    for fp in files:
        with fp.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            hdr = [h.strip() for h in (r.fieldnames or [])]
            if not hdr:
                continue
            pid_res = _pick(hdr, ("pid_resolved",))
            pid_q   = _pick(hdr, ("pid_qrels",))
            inj_col = _pick(hdr, ("passage_injected","passage_eng"))
            if not inj_col or not (pid_res or pid_q):
                continue

            for row in r:
                total += 1
                pid = (row.get(pid_res) or row.get(pid_q) or "").strip()
                inj = _norm(row.get(inj_col) or "")
                if not pid or not inj:
                    continue
                key = (pid, inj)
                if key not in pair_set:
                    pair_set.add(key)
                    # store a slim row with useful columns
                    slim = {k: row.get(k, "") for k in keep_cols if k in hdr}
                    # ensure all expected columns exist
                    for k in keep_cols:
                        slim.setdefault(k, "")
                    rows_by_pair[key] = slim
    print(f"[topics] files: {len(files)}; rows scanned: {total:,}; unique (pid,passage_injected): {len(pair_set):,}")
    return pair_set, rows_by_pair, merged_header

def write_csv(path: Path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
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
    topics_pairs, topics_rows_by_pair, topics_hdr = load_topics_pairs(TOPICS_DIR, TOPICS_GLOB)

    # A) What you asked for:
    # topics that DON'T appear in the LLM file by (pid, passage)
    missing_in_llm_keys = topics_pairs - llm_pairs
    missing_rows = [topics_rows_by_pair[k] for k in missing_in_llm_keys]
    write_csv(TOPICS_NOT_IN_LLM, missing_rows, topics_hdr)

    # Optional sanity: LLM pairs not seen in topics
    if WRITE_INVERSE_CHECK:
        llm_not_in_topics_keys = llm_pairs - topics_pairs
        # write the raw LLM rows for inspection
        headers = list(llm_hdr) + (["note"] if "note" not in llm_hdr else [])
        rows = []
        for k in llm_not_in_topics_keys:
            r = dict(llm_rows_by_pair[k])
            r.setdefault("note", "not_in_topics")
            rows.append(r)
        write_csv(LLM_NOT_IN_TOPICS, rows, headers)

    print(f"\nSummary:")
    print(f"  topics unique pairs .......... {len(topics_pairs):,}")
    print(f"  llm unique pairs ............. {len(llm_pairs):,}")
    print(f"  topics NOT in llm ............ {len(missing_in_llm_keys):,}")
    if WRITE_INVERSE_CHECK:
        print(f"  llm NOT in topics ............ {len(llm_pairs - topics_pairs):,}")

if __name__ == "__main__":
    main()
