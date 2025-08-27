import os
from pathlib import Path

# --- Put all caches under D: (fewer lock/permission issues than %LOCALAPPDATA%)
os.environ['PYSERINI_CACHE']   = r'D:\PyseriniCache'
os.environ['IR_DATASETS_HOME'] = r'D:\ir_datasets'

# Critical: move temp out of C:\Users\...\AppData\Local\Temp where AV/locks happen
os.environ['TMP']  = r'D:\ir_tmp'
os.environ['TEMP'] = r'D:\ir_tmp'

# (Optional) single-threaded downloads reduce chance of Windows file lock races
os.environ['IR_DATASETS_DL_THREADS'] = '1'

# Make sure folders exist
for p in [r'D:\PyseriniCache', r'D:\ir_datasets', r'D:\ir_tmp']:
    Path(p).mkdir(parents=True, exist_ok=True)

from pyserini.search.lucene import LuceneSearcher
import ir_datasets

# --- TREC-DL 2019 (passage) topics + qrels
ds = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')

# --- Searcher: MS MARCO v1 passage index (matches the docids used by TREC-DL 2019)
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
searcher.set_bm25(k1=0.82, b=0.68)

# Build qrels dict {qid: {docid: relevance}}
qrels = {}
for q in ds.qrels_iter():
    qrels.setdefault(str(q.query_id), {})[q.doc_id] = int(q.relevance)

# Grab a query (TREC-DL 2019 has ~43)
queries = list(ds.queries_iter())
query = queries[2]  # pick one
qid = str(query.query_id)
topic = query.text
print(f"Query {qid}: {topic}\n")

# Retrieve
k = 10
hits = searcher.search(topic, k=k)

# Collect
results = []
for i, h in enumerate(hits, 1):
    doc = searcher.doc(h.docid)
    doc_text = doc.raw() if doc is not None else ""
    results.append({
        "query": topic,
        "passage": doc_text,
        "docid": h.docid,
        "score": h.score,
        "relevance": qrels.get(qid, {}).get(h.docid, None)
    })

# Write outputs
out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)

labels_path = out_dir / f"labels_trecdl2019_q{qid}.tsv"
with labels_path.open("w", encoding="utf-8", newline="") as f:
    f.write(f"# Dataset: TREC-DL 2019 (passage)\n# Query ID: {qid}\n# Topic: {topic}\n")
    f.write("docid\trelevance\tscore\n")
    for r in results:
        f.write(f"{r['docid']}\t{r['relevance']}\t{r['score']:.4f}\n")

docs_path = out_dir / f"topic_and_docs_trecdl2019_q{qid}.txt"
with docs_path.open("w", encoding="utf-8") as f:
    f.write(f"Dataset: TREC-DL 2019 (passage)\nQuery ID: {qid}\nTopic: {topic}\n\n")
    for i, r in enumerate(results, 1):
        f.write(f"Doc {i}: {r['docid']} (rel={r['relevance']}, score={r['score']:.3f})\n")
        f.write("Passage:\n" + (r["passage"] or "") + "\n" + "-"*80 + "\n\n")

print(f"Wrote: {labels_path}")
print(f"Wrote: {docs_path}")
