import os
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Iterable

from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels

# ----------------------------
# Config
# ----------------------------
os.environ['PYSERINI_CACHE'] = r'D:\PyseriniCache'
Path(r'D:\PyseriniCache').mkdir(parents=True, exist_ok=True)

TRECDL_YEAR = '2019'      # '2019' or '2020'
LEVEL       = 'passage'   # 'passage' or 'document'

# You can force specific QIDs by setting FORCE_QIDS to a list/tuple/set of qids (ints/strs).
# If left as None, the first N judged topics (sorted) will be used.
FORCE_QIDS: Iterable[Any] | None = None

# Run at least this many different queries (topics)
N_QUERIES   = 25
DOCS_PER_TOPIC = 4 
K_START     = 50     # initial retrieval depth
MIN_JUDGED  = 10     # minimum judged docs to retrieve
K_CAP       = 10000  # maximum retrieval depth

# ---- Output location ----
OUT_DIR = Path("outputs/trec_dl")
COMBINED_CSV = OUT_DIR / f"trecdl_{LEVEL}_{TRECDL_YEAR}_combined.csv"

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

def build_dual_qrels(raw_qrels: Dict[str, Any], level: str) -> Dict[str, int]:
    """Map both prefixed and bare docids to the same (int) grade."""
    dual: Dict[str, int] = {}
    for did, g in raw_qrels.items():
        grade = as_int_grade(g)
        for form in alt_docid_forms(did, level):
            dual[form] = max(grade, dual.get(form, -1))
    return dual

def retrieve_until_min_judged(searcher: LuceneSearcher, query_text: str,
                              judged_lookup: Dict[str, int],
                              k_start: int, min_judged: int, k_cap: int,
                              desired_k: int | None = None):
    k = k_start
    while True:
        k_effective = max(k, desired_k or 0)
        hits = searcher.search(query_text, k=k_effective)
        judged = sum(1 for h in hits if h.docid in judged_lookup)
        # stop when judged condition met AND we have at least desired_k (if specified)
        if (judged >= min_judged or k >= k_cap) and (desired_k is None or len(hits) >= desired_k):
            print(f"Final k={k_effective}: judged_in_topk={judged}, unjudged_in_topk={len(hits)-judged}")
            print("Sample hit docids   :", [h.docid for h in hits[:5]])
            print("Sample qrels docids :", list(judged_lookup.keys())[:5])
            return hits
        k = min(max(k * 2, desired_k or k), k_cap)

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
            # Some variants store under "raw" or similar keys
            for k in ("raw", "text", "body"):
                if k in obj and isinstance(obj[k], str):
                    return obj[k]
    except Exception:
        pass
    # Fallback: use .contents() if available, else raw
    try:
        c = doc.contents()
        if isinstance(c, str) and c.strip():
            return c
    except Exception:
        pass
    return raw

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
    searcher.set_bm25(k1=0.82, b=0.68)

    OUT_DIR.mkdir(exist_ok=True)

    with COMBINED_CSV.open("w", encoding="utf-8", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["query", "docid", "passage", "relevance"])

        for qid_key in qids_to_run:
            raw_qrels_for_qid = qrels_for(qrels, qid_key)
            qrels_dual = build_dual_qrels(raw_qrels_for_qid, LEVEL)

            query_text = topic_text(topics[qid_key])

            print("\n" + "="*80)
            print(f"qid        : {qid_key} (type={type(qid_key).__name__})")
            print(f"Query      : {query_text}")
            print(f"Total judged (raw qrels) for qid: {len(raw_qrels_for_qid)}")

            hits = retrieve_until_min_judged(
                searcher=searcher,
                query_text=query_text,
                judged_lookup=qrels_dual,
                k_start=K_START,
                min_judged=MIN_JUDGED,
                k_cap=K_CAP,
                desired_k=DOCS_PER_TOPIC,   # << ensure we retrieve at least this many
            )

            for h in hits[:DOCS_PER_TOPIC]:
                doc  = searcher.doc(h.docid)
                text = extract_text_from_doc(doc)
                rel  = qrels_dual.get(h.docid)
                writer.writerow([query_text, h.docid, text, "" if rel is None else rel])

    print(f"\nWrote combined CSV: {COMBINED_CSV}")
