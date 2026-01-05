#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Iterable, Tuple, Optional

from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels

# ----------------------------
# Config
# ----------------------------
os.environ["PYSERINI_CACHE"] = r"D:\PyseriniCache"
Path(r"D:\PyseriniCache").mkdir(parents=True, exist_ok=True)

TRECDL_YEAR = "2021"       # '2019', '2020', '2021', '2022', or '2023'
LEVEL       = "passage"    # 'passage' or 'document'
FETCH_TEXT  = True
CHUNK_SIZE  = 500          # rows per output CSV chunk

# Mode: dump judged qrels ('qrels') OR retrieve top-K and annotate with qrels ('topk')
MODE     = "topk"          # 'qrels' or 'topk'
K_DEPTH  = 30              # number of docs per query when MODE == 'topk'

# Optional: force a subset of qids; otherwise take the first N judged
FORCE_QIDS: Iterable[Any] | None = None
N_QUERIES = 100000000

# ---- Output location ----
OUT_DIR = Path(f"retrieved/trec_dl_{TRECDL_YEAR}/judged_new")

# ----------------------------
# Helpers
# ----------------------------
def topic_key_for(year: str, level: str) -> str:
    # DL21+ topics keys are dl21/dl22/dl23 (no passage/doc suffix)
    if year in ("2021", "2022", "2023"):
        return f"dl{year[2:]}"  # "2021" -> "dl21"
    mapping = {
        "2019": {"passage": "dl19-passage", "document": "dl19-doc"},
        "2020": {"passage": "dl20-passage", "document": "dl20-doc"},
    }
    return mapping[year][level]

def qrels_key_for(year: str, level: str) -> str:
    return {
        "2019": {"passage": "dl19-passage", "document": "dl19-doc"},
        "2020": {"passage": "dl20-passage", "document": "dl20-doc"},
        "2021": {"passage": "dl21-passage", "document": "dl21-doc"},
        "2022": {"passage": "dl22-passage", "document": "dl22-doc"},
        "2023": {"passage": "dl23-passage", "document": "dl23-doc"},
    }[year][level]

def index_name_for(year: str, level: str) -> str:
    # DL19/DL20 use MS MARCO v1; DL21+ use MS MARCO v2
    if year in ("2019", "2020"):
        return "msmarco-v1-passage" if level == "passage" else "msmarco-v1-doc"
    return "msmarco-v2-passage" if level == "passage" else "msmarco-v2.1-doc-segmented"

def qid_sort_key(x: Any):
    sx = str(x)
    return int(sx) if sx.isdigit() else sx

def topic_text(rec: Any) -> str:
    if isinstance(rec, dict):
        return rec.get("title") or rec.get("text") or rec.get("query") or str(rec)
    for attr in ("title", "text", "query"):
        if hasattr(rec, attr):
            v = getattr(rec, attr)
            if v:
                return v
    return str(rec)

def qrels_for(qrels_by_qid: Dict[Any, Dict[str, Any]], qid_any: Any) -> Dict[str, Any]:
    """Return qrels for qid whether keys are ints or strs; strips whitespace."""
    if qid_any in qrels_by_qid:
        return qrels_by_qid[qid_any]
    s = str(qid_any).strip()
    if s in qrels_by_qid:
        return qrels_by_qid[s]
    try:
        i = int(s)
        if i in qrels_by_qid:
            return qrels_by_qid[i]
    except ValueError:
        pass
    return {}

def alt_docid_forms(docid: str, level: str) -> List[str]:
    """Return both prefixed and bare MS MARCO forms for robust matching."""
    s = str(docid).strip()
    if level == "passage":
        if s.startswith("msmarco_passage_"):
            return [s, s.replace("msmarco_passage_", "", 1)]
        return [s, f"msmarco_passage_{s}"]
    else:
        if s.startswith("msmarco_doc_"):
            return [s, s.replace("msmarco_doc_", "", 1)]
        return [s, f"msmarco_doc_{s}"]

def as_int_grade(g: Any) -> int:
    """Normalize qrels grade to int (handles '0','1','2','3' as strings)."""
    try:
        return int(g)
    except Exception:
        s = str(g).strip()
        if s.isdigit():
            return int(s)
        return 0

def extract_text_from_doc(doc) -> str:
    """Return plain passage/doc text (not the whole JSON)."""
    if not doc:
        return ""
    raw = doc.raw() or ""
    # Try JSON payload first
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for k in ("contents", "passage", "passage_text", "text", "body", "raw"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v
    except Exception:
        pass
    # Fallback to contents() if available
    try:
        c = doc.contents()
        if isinstance(c, str) and c.strip():
            return c
    except Exception:
        pass
    return raw

def normalize_csv_cell(s: str) -> str:
    """Collapse CR/LF/TAB to spaces so each CSV row is single-line."""
    return " ".join((s or "").replace("\r", " ").replace("\n", " ").replace("\t", " ").split())

class RollingCsvWriter:
    """
    Writes rows into multiple CSV files, rotating every `chunk_size` rows.
    Files are named: {prefix}{part}.csv (no zero padding), each with a header.
    """
    def __init__(self, out_dir: Path, prefix: str, header: List[str], chunk_size: int = 500):
        self.out_dir = out_dir
        self.prefix = prefix
        self.header = header
        self.chunk_size = max(1, int(chunk_size))
        self.part = 0
        self.rows_in_part = 0
        self.fh = None
        self.writer = None
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _open_next(self):
        if self.fh:
            self.fh.close()
        self.part += 1
        self.rows_in_part = 0
        filename = f"{self.prefix}{self.part}.csv"
        self.fh = (self.out_dir / filename).open("w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.fh)
        self.writer.writerow(self.header)

    def write(self, row: List[Any]):
        if self.writer is None or self.rows_in_part >= self.chunk_size:
            self._open_next()
        self.writer.writerow(row)
        self.rows_in_part += 1

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None
            self.writer = None

def fetch_doc_by_any_form(searcher: LuceneSearcher, docid: str, level: str) -> Tuple[Optional[str], Optional[Any]]:
    """Try both bare and prefixed forms; return (resolved_id, doc) or (None, None)."""
    for form in alt_docid_forms(docid, level):
        d = searcher.doc(form)
        if d is not None:
            return form, d
    return None, None

def grade_for_doc(qrels_for_qid: Dict[str, Any], docid: str, level: str) -> int:
    """Return qrels grade for docid if known (robust to msmarco id forms), else 0."""
    s = str(docid).strip()
    for form in alt_docid_forms(s, level):
        if form in qrels_for_qid:
            return as_int_grade(qrels_for_qid[form])
    if s in qrels_for_qid:
        return as_int_grade(qrels_for_qid[s])
    return 0

def pick_qids_to_run(
    all_topics: Dict[Any, Any],
    qrels_by_qid: Dict[Any, Dict[str, Any]],
    force_qids: Iterable[Any] | None,
    n_queries: int,
) -> List[Any]:
    """Choose at least n_queries judged topics (or all if fewer)."""
    judged = [qid for qid in all_topics.keys() if len(qrels_for(qrels_by_qid, qid)) > 0]
    judged_sorted = sorted(judged, key=qid_sort_key)

    if force_qids:
        forced = []
        for q in force_qids:
            if q in all_topics and len(qrels_for(qrels_by_qid, q)) > 0:
                forced.append(q)
        if not forced:
            raise RuntimeError("None of the FORCE_QIDS have qrels.")
        return forced

    if not judged_sorted:
        raise RuntimeError("No judged topics found for this topics key.")
    if len(judged_sorted) < n_queries:
        print(f"Warning: only {len(judged_sorted)} judged topics available; running all.")
    return judged_sorted[:max(1, min(len(judged_sorted), n_queries))]

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    tkey       = topic_key_for(TRECDL_YEAR, LEVEL)
    qrels_key  = qrels_key_for(TRECDL_YEAR, LEVEL)
    index_name = index_name_for(TRECDL_YEAR, LEVEL)

    # Light sanity check
    if LEVEL == "passage" and "passage" not in index_name:
        raise RuntimeError(f"LEVEL={LEVEL} must match index={index_name}")
    if LEVEL == "document" and "doc" not in index_name:
        raise RuntimeError(f"LEVEL={LEVEL} must match index={index_name}")

    topics = get_topics(tkey)
    qrels  = get_qrels(qrels_key)

    qids_to_run = pick_qids_to_run(
        all_topics=topics,
        qrels_by_qid=qrels,
        force_qids=FORCE_QIDS,
        n_queries=N_QUERIES,
    )

    print(f"Output dir : {OUT_DIR}")
    print(f"Topics key : {tkey}")
    print(f"Qrels key  : {qrels_key}")
    print(f"Index      : {index_name}")
    print(f"Mode       : {MODE} (K={K_DEPTH if MODE=='topk' else 'n/a'})")
    print(f"Running {len(qids_to_run)} queries.")

    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    searcher.set_bm25(k1=0.82, b=0.68)

    # ✅ New CSV schema: qid,query,pid,passage,relevance
    rolling = RollingCsvWriter(
        out_dir=OUT_DIR,
        prefix=f"trecdl_{LEVEL}_{TRECDL_YEAR}_part",
        header=["qid", "query", "pid", "passage", "relevance"],
        chunk_size=CHUNK_SIZE,
    )

    total_rows = 0

    try:
        for qid_key in qids_to_run:
            query_text = topic_text(topics[qid_key])
            qrels_for_qid = qrels_for(qrels, qid_key)  # dict[docid] -> grade

            if MODE == "qrels":
                # Dump judged qrels only (pid comes from qrels, resolved to index id if possible)
                for did, grade in qrels_for_qid.items():
                    pid_qrels = str(did).strip()

                    if FETCH_TEXT:
                        pid_resolved, doc = fetch_doc_by_any_form(searcher, pid_qrels, LEVEL)
                        pid_out = pid_resolved or pid_qrels
                        text = extract_text_from_doc(doc) if doc is not None else ""
                    else:
                        pid_resolved, _ = fetch_doc_by_any_form(searcher, pid_qrels, LEVEL)
                        pid_out = pid_resolved or pid_qrels
                        text = ""

                    rolling.write([
                        str(qid_key),
                        query_text,
                        pid_out,
                        normalize_csv_cell(text),
                        as_int_grade(grade),
                    ])
                    total_rows += 1

            else:
                # MODE == 'topk': retrieve top-K and annotate with qrels (grade=0 if unjudged)
                hits = searcher.search(query_text, k=K_DEPTH)
                for h in hits:
                    pid_out = str(h.docid).strip()

                    doc = searcher.doc(pid_out) if FETCH_TEXT else None
                    text = extract_text_from_doc(doc) if doc is not None else ""

                    grade = grade_for_doc(qrels_for_qid, pid_out, LEVEL)

                    rolling.write([
                        str(qid_key),
                        query_text,
                        pid_out,
                        normalize_csv_cell(text),
                        grade,  # 0 if unjudged
                    ])
                    total_rows += 1

    finally:
        rolling.close()

    print(f"\nWrote {total_rows} rows into chunked CSVs under: {OUT_DIR}")
