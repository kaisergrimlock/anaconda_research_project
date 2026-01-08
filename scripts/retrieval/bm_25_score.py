#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import tempfile
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher


# ============================================================
# Config (edit these)
# ============================================================
LANG = "vi"  # "raw", "vi", "fr", ...

YEAR = "2021"
MODEL = "gpt-oss-20b"

IN_CSV = (
    f"outputs/llm_label/trec_dl_{YEAR}/{MODEL}/"
    f"{MODEL}_trecdl_{YEAR}_{LANG}_labels.csv"
)

OUT_CSV = (
    f"outputs/analysis/"
    f"bm25_proxy_{MODEL}_trecdl_{YEAR}_{LANG}_labels.csv"
)

# Which query column to use for BM25 scoring (must exist in the CSV)
QUERY_COL = "query"

# Passage columns (must exist in the CSV)
PASSAGE_COL = "passage"
INJECTED_PASSAGE_COL = "passage_injected"

# BM25 params for mini-index
BM25_K1 = 0.82
BM25_B = 0.68

# Mini-index retrieval depth (should be >= number of docs in mini-index)
# Safe default: large enough to cover all 2*N docs per qid.
MINI_SEARCH_K = 2000

# Assumption: rows are grouped by qid (often true). If unsure, set False.
ASSUME_GROUPED_BY_QID = True

# Keep all original columns in the output?
KEEP_ALL_INPUT_COLUMNS = True
# ============================================================


def normalize_cell(s: str) -> str:
    """Collapse CR/LF/TAB to spaces so each CSV row stays single-line."""
    return " ".join((s or "").replace("\r", " ").replace("\n", " ").replace("\t", " ").split())


# ----------------------------
# Mini-index helpers
# ----------------------------
def write_jsonl(docs: List[Tuple[str, str]], jsonl_path: Path) -> None:
    with jsonl_path.open("w", encoding="utf-8") as f:
        for docid, contents in docs:
            f.write(json.dumps({"id": docid, "contents": contents}, ensure_ascii=False) + "\n")


def build_mini_index(docs: List[Tuple[str, str]], *, k1: float, b: float) -> Tuple[LuceneSearcher, Path]:
    """
    Build a temporary Lucene index using Pyserini CLI (version-stable),
    return (searcher, tmp_dir). Caller must delete tmp_dir.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="bm25_mini_index_"))

    collection_dir = tmp_dir / "collection"
    index_dir = tmp_dir / "index"
    collection_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Write a JSONL collection file
    data_jsonl = collection_dir / "docs.jsonl"
    write_jsonl(docs, data_jsonl)

    # Build index using the CLI module (works across Pyserini versions)
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(collection_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storeRaw",
    ]

    # Optional: if you want closer to typical retrieval configs, uncomment:
    # cmd += ["--storePositions", "--storeDocvectors"]

    subprocess.run(cmd, check=True)

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=k1, b=b)
    return searcher, tmp_dir

def score_wanted(searcher: LuceneSearcher, query: str, wanted_ids: List[str], *, k: int) -> Dict[str, float]:
    wanted = set(wanted_ids)
    out: Dict[str, float] = {d: 0.0 for d in wanted_ids}

    hits = searcher.search(query, k=max(k, len(wanted_ids) * 2))
    for h in hits:
        if h.docid in wanted:
            out[h.docid] = float(h.score)

    return out


# ----------------------------
# Per-qid scoring
# ----------------------------
def score_group(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rows:
        return []

    query = (rows[0].get(QUERY_COL) or "").strip()
    if not query:
        # If query is empty, return zeros for the group
        out = []
        for r in rows:
            rr = dict(r)
            rr["bm25_orig_mini"] = "0.0"
            rr["bm25_inj_mini"] = "0.0"
            rr["bm25_delta_mini"] = "0.0"
            rr["bm25_query_col_used"] = QUERY_COL
            out.append(rr)
        return out

    docs: List[Tuple[str, str]] = []
    orig_ids: List[str] = []
    inj_ids: List[str] = []

    for r in rows:
        pid = (r.get("pid") or "").strip()
        p_orig = normalize_cell(r.get(PASSAGE_COL, ""))
        p_inj = normalize_cell(r.get(INJECTED_PASSAGE_COL, ""))

        oid = f"{pid}__orig"
        iid = f"{pid}__inj"

        docs.append((oid, p_orig))
        docs.append((iid, p_inj))
        orig_ids.append(oid)
        inj_ids.append(iid)

    mini_searcher, tmp_dir = build_mini_index(docs, k1=BM25_K1, b=BM25_B)
    try:
        scores_orig = score_wanted(mini_searcher, query, orig_ids, k=MINI_SEARCH_K)
        scores_inj = score_wanted(mini_searcher, query, inj_ids, k=MINI_SEARCH_K)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    out: List[Dict[str, str]] = []
    for r in rows:
        pid = (r.get("pid") or "").strip()
        oid = f"{pid}__orig"
        iid = f"{pid}__inj"

        bm25_orig = float(scores_orig.get(oid, 0.0))
        bm25_inj = float(scores_inj.get(iid, 0.0))
        bm25_delta = bm25_inj - bm25_orig

        rr = dict(r)
        rr["bm25_orig_mini"] = f"{bm25_orig:.6f}"
        rr["bm25_inj_mini"] = f"{bm25_inj:.6f}"
        rr["bm25_delta_mini"] = f"{bm25_delta:.6f}"
        rr["bm25_query_col_used"] = QUERY_COL
        out.append(rr)

    return out


def main() -> None:
    in_path = Path(IN_CSV)
    if not in_path.exists():
        raise RuntimeError(f"Input CSV not found: {in_path}")

    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise RuntimeError("Input CSV has no header row.")

        required = {"qid", "pid", QUERY_COL, PASSAGE_COL, INJECTED_PASSAGE_COL}
        missing = required - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"Missing required columns: {sorted(missing)}\n"
                f"CSV columns: {reader.fieldnames}"
            )

        input_fields = list(reader.fieldnames)

        bm25_fields = ["bm25_orig_mini", "bm25_inj_mini", "bm25_delta_mini", "bm25_query_col_used"]
        if KEEP_ALL_INPUT_COLUMNS:
            fieldnames = input_fields + [c for c in bm25_fields if c not in input_fields]
        else:
            fieldnames = ["qid", QUERY_COL, "pid", PASSAGE_COL, INJECTED_PASSAGE_COL] + bm25_fields

        cur_qid: Optional[str] = None
        buffer: List[Dict[str, str]] = []

        with out_path.open("w", encoding="utf-8", newline="") as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            if ASSUME_GROUPED_BY_QID:
                for rec in reader:
                    qid = (rec.get("qid") or "").strip()
                    if cur_qid is None:
                        cur_qid = qid

                    if qid != cur_qid:
                        for out_rec in score_group(buffer):
                            writer.writerow(out_rec)
                        buffer = []
                        cur_qid = qid

                    buffer.append({k: (rec.get(k) or "") for k in input_fields})

                # flush last group
                if buffer:
                    for out_rec in score_group(buffer):
                        writer.writerow(out_rec)

            else:
                # Robust mode: group everything in memory
                by_qid: Dict[str, List[Dict[str, str]]] = {}
                for rec in reader:
                    qid = (rec.get("qid") or "").strip()
                    by_qid.setdefault(qid, []).append({k: (rec.get(k) or "") for k in input_fields})

                for qid in sorted(by_qid.keys()):
                    for out_rec in score_group(by_qid[qid]):
                        writer.writerow(out_rec)

    print(f"Wrote BM25 proxy scores to: {out_path}")
    print(f"Query col used: {QUERY_COL}")


if __name__ == "__main__":
    main()
