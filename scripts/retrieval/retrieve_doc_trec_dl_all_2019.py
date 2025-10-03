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
os.environ['PYSERINI_CACHE'] = r'D:\PyseriniCache'
Path(r'D:\PyseriniCache').mkdir(parents=True, exist_ok=True)

TRECDL_YEAR = '2019'      # '2019' or '2020'
LEVEL       = 'passage'   # 'passage' or 'document'
FETCH_TEXT  = True        # set False if you only need (qid, docid, rel)

# Optional: force a subset of qids; otherwise we take the first N judged
FORCE_QIDS: Iterable[Any] | None = None
N_QUERIES   = 25

# ---- Output location ----
OUT_DIR = Path("outputs/trec_dl")
COMBINED_CSV = OUT_DIR / f"trecdl_{LEVEL}_{TRECDL_YEAR}_judged_only.csv"

# ----------------------------
# Helpers
# ----------------------------
def topic_key_for(year: str, level: str) -> str:
    return {
        '2019': {'passage': 'dl19-passage', 'document': 'dl19-doc'},
        '2020': {'passage': 'dl20-passage', 'document': 'dl20-doc'},
    }[year][level]

def index_name_for(level: str) -> str:
    return 'msmarco-v1-passage' if level == 'passage' else 'msmarco-v1-doc'

def qid_sort_key(x: Any):
    sx = str(x)
    return int(sx) if sx.isdigit() else sx

def topic_text(rec: Any) -> str:
    if isinstance(rec, dict):
        return rec.get('title') or rec.get('text') or rec.get('query') or str(rec)
    for attr in ('title', 'text', 'query'):
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
    if level == 'passage':
        if s.startswith('msmarco_passage_'):
            return [s, s.replace('msmarco_passage_', '', 1)]
        return [s, f'msmarco_passage_{s}']
    else:  # document
        if s.startswith('msmarco_doc_'):
            return [s, s.replace('msmarco_doc_', '', 1)]
        return [s, f'msmarco_doc_{s}']

def as_int_grade(g: Any) -> int:
    """Normalize qrels grade to int (handles '0','1','2','3' as strings)."""
    try:
        return int(g)
    except Exception:
        return 1 if str(g).strip().isdigit() and int(str(g).strip()) > 0 else 0

def extract_text_from_doc(doc) -> str:
    """Best-effort extraction of passage/document text."""
    if not doc:
        return ""
    raw = doc.raw() or ""
    # Try JSON payload (common in msmarco prebuilt indexes)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            if "contents" in obj and isinstance(obj["contents"], str):
                return obj["contents"]
            for k in ("raw", "text", "body"):
                if k in obj and isinstance(obj[k], str):
                    return obj[k]
    except Exception:
        pass
    try:
        c = doc.contents()
        if isinstance(c, str) and c.strip():
            return c
    except Exception:
        pass
    return raw

def fetch_doc_by_any_form(searcher: LuceneSearcher, docid: str, level: str) -> Tuple[Optional[str], Optional[Any]]:
    """Try both bare and prefixed forms; return (resolved_id, doc) or (None, None)."""
    for form in alt_docid_forms(docid, level):
        d = searcher.doc(form)
        if d is not None:
            return form, d
    return None, None

def pick_qids_to_run(all_topics: Dict[Any, Any],
                     qrels_by_qid: Dict[Any, Dict[str, Any]],
                     force_qids: Iterable[Any] | None,
                     n_queries: int) -> List[Any]:
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
    tkey = topic_key_for(TRECDL_YEAR, LEVEL)
    index_name = index_name_for(LEVEL)

    if (LEVEL == 'passage' and not index_name.endswith('passage')) or \
       (LEVEL == 'document' and not index_name.endswith('doc')):
        raise RuntimeError(f"LEVEL={LEVEL} must match index={index_name}")

    topics = get_topics(tkey)   # qids usually ints
    qrels  = get_qrels(tkey)    # qids can be int or str depending on bundle

    qids_to_run = pick_qids_to_run(
        all_topics=topics,
        qrels_by_qid=qrels,
        force_qids=FORCE_QIDS,
        n_queries=N_QUERIES
    )

    print(f"Topics key : {tkey}")
    print(f"Index      : {index_name}")
    print(f"Running {len(qids_to_run)} queries: {sorted(map(str, qids_to_run), key=qid_sort_key)}")

    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    # Keep your BM25 settings in case you later want retrieval again
    searcher.set_bm25(k1=0.82, b=0.68)

    OUT_DIR.mkdir(exist_ok=True)

    with COMBINED_CSV.open("w", encoding="utf-8", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["query", "docid", "passage", "relevance"])

        for qid_key in qids_to_run:
            raw_qrels_for_qid = qrels_for(qrels, qid_key)  # dict[docid] -> grade
            query_text = topic_text(topics[qid_key])

            total_judged = len(raw_qrels_for_qid)
            found, missing = 0, 0

            print("\n" + "="*80)
            print(f"qid        : {qid_key} (type={type(qid_key).__name__})")
            print(f"Query      : {query_text}")
            print(f"Total judged in qrels for qid: {total_judged}")

            for did, grade in raw_qrels_for_qid.items():
                resolved_id, doc = (None, None)
                if FETCH_TEXT:
                    resolved_id, doc = fetch_doc_by_any_form(searcher, did, LEVEL)
                    if doc is not None:
                        found += 1
                        text = extract_text_from_doc(doc)
                        writer.writerow([query_text, resolved_id, text, as_int_grade(grade)])
                    else:
                        missing += 1
                        # Still write a row so you can see what's missing from the index
                        writer.writerow([query_text, did, "", as_int_grade(grade)])
                else:
                    # Not fetching text; resolve id if possible (optional), else use did
                    resolved_id, _ = fetch_doc_by_any_form(searcher, did, LEVEL)
                    writer.writerow([query_text, resolved_id or did, "", as_int_grade(grade)])
                    found += 1 if resolved_id else 0
                    missing += 0 if resolved_id else 1

            print(f"Found in index: {found} / {total_judged} | Missing: {missing}")

    print(f"\nWrote judged-only CSV: {COMBINED_CSV}")
