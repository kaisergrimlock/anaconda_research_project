import os
from pathlib import Path
from typing import Any, Dict, List

from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels

# ----------------------------
# Config
# ----------------------------
os.environ['PYSERINI_CACHE'] = r'D:\PyseriniCache'
Path(r'D:\PyseriniCache').mkdir(parents=True, exist_ok=True)

TRECDL_YEAR = '2019'      # '2019' or '2020'
LEVEL       = 'passage'   # 'passage' or 'document'
FORCE_QID   = None        # set to an int/str to force that qid; otherwise auto-pick a judged one

K_START     = 50 #initial retrieval depth
MIN_JUDGED  = 10 #minimum judged docs to retrieve
K_CAP       = 2000 #maximum retrieval depth

# ---- Output locations (edit here) ----
OUT_DIR = Path("outputs/trec_dl")  # make sure this exists or can be created
LABELS_NAME_FMT = "labels_trecdl_{level}_{year}_q{qid}.tsv"
DOCS_NAME_FMT   = "topic_and_docs_trecdl_{level}_{year}_q{qid}.txt"
# Usage later:
# labels_path = OUT_DIR / LABELS_NAME_FMT.format(level=LEVEL, year=TRECDL_YEAR, qid=qid_str)
# docs_path   = OUT_DIR / DOCS_NAME_FMT.format(level=LEVEL, year=TRECDL_YEAR, qid=qid_str)

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

def pick_judged_qid(all_topics: Dict[Any, Any], qrels_by_qid: Dict[Any, Dict[str, Any]]) -> Any:
    judged_qids = [qid for qid in all_topics.keys() if len(qrels_for(qrels_by_qid, qid)) > 0]
    if judged_qids:
        return sorted(judged_qids, key=qid_sort_key)[0]
    return sorted(all_topics.keys(), key=qid_sort_key)[0]

def retrieve_until_min_judged(searcher: LuceneSearcher, query_text: str,
                              judged_lookup: Dict[str, int],
                              k_start: int, min_judged: int, k_cap: int):
    k = k_start
    while True:
        hits = searcher.search(query_text, k=k)
        judged = sum(1 for h in hits if h.docid in judged_lookup)
        if judged >= min_judged or k >= k_cap:
            print(f"Final k={k}: judged_in_topk={judged}, unjudged_in_topk={len(hits)-judged}")
            print("Sample hit docids   :", [h.docid for h in hits[:5]])
            print("Sample qrels docids :", list(judged_lookup.keys())[:5])
            return hits
        k = min(k * 2, k_cap)

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

    if FORCE_QID is not None and FORCE_QID in topics:
        qid_key = FORCE_QID
    else:
        qid_key = pick_judged_qid(topics, qrels)

    raw_qrels_for_qid = qrels_for(qrels, qid_key)
    qrels_dual = build_dual_qrels(raw_qrels_for_qid, LEVEL)

    query_text = topic_text(topics[qid_key])
    qid_str    = str(qid_key)

    print(f"Topics key : {tkey}")
    print(f"Index      : {index_name}")
    print(f"qid        : {qid_str} (type={type(qid_key).__name__})")
    print(f"Query      : {query_text}")
    print(f"Total judged (raw qrels) for qid: {len(raw_qrels_for_qid)}")

    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    searcher.set_bm25(k1=0.82, b=0.68)

    hits = retrieve_until_min_judged(
        searcher=searcher,
        query_text=query_text,
        judged_lookup=qrels_dual,
        k_start=K_START,
        min_judged=MIN_JUDGED,
        k_cap=K_CAP
    )

    # Write outputs (using the top-level variables)
    OUT_DIR.mkdir(exist_ok=True)
    labels_path = OUT_DIR / LABELS_NAME_FMT.format(level=LEVEL, year=TRECDL_YEAR, qid=qid_str)
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# Topics: {tkey}\n# Query ID: {qid_str}\n# Query: {query_text}\n")
        f.write("docid\trelevance\tscore\n")
        for h in hits:
            rel = qrels_dual.get(h.docid)
            f.write(f"{h.docid}\t{rel}\t{h.score:.4f}\n")

    docs_path = OUT_DIR / DOCS_NAME_FMT.format(level=LEVEL, year=TRECDL_YEAR, qid=qid_str)
    with docs_path.open("w", encoding="utf-8") as f:
        f.write(f"Topics: {tkey}\nQuery ID: {qid_str}\nQuery: {query_text}\n\n")
        for rank, h in enumerate(hits, 1):
            rel  = qrels_dual.get(h.docid)
            body = (searcher.doc(h.docid).raw() or "") if searcher.doc(h.docid) else ""
            f.write(f"Doc {rank}: {h.docid} (rel={rel}, score={h.score:.3f})\n")
            f.write(("Passage:\n" if LEVEL == "passage" else "Document:\n") + body + "\n" + "-"*80 + "\n\n")

    print(f"Wrote: {labels_path}")
    print(f"Wrote: {docs_path}")
