import os
os.environ['PYSERINI_CACHE'] = r'D:\PyseriniCache'
os.environ['IR_DATASETS_HOME'] = r'D:\ir_datasets'

from pyserini.search.lucene import LuceneSearcher
import ir_datasets

# 1) Searcher setup
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
searcher.set_bm25(k1=0.82, b=0.68)

# 2) Load dataset (queries + qrels)
ds = ir_datasets.load('msmarco-passage/dev/small')

# Build qrels dict {qid: {docid: relevance}}
qrels = {}
for q in ds.qrels_iter():
    qrels.setdefault(str(q.query_id), {})[q.doc_id] = int(q.relevance)

# 3) Pick one benchmark query
# Get all queries as a list
queries = list(ds.queries_iter())

# Pick the second query (index 1)
query = queries[2]
qid = str(query.query_id)
topic = query.text

print(f"Query {qid}: {topic}\n")

# 4) Retrieve top 10 docs for this query
hits = searcher.search(topic, k=100)

# 5) Print results with topic, docid, relevance, and raw doc text
print(f"Top 10 results for query {qid} ({topic}):\n")
results = []
for i, h in enumerate(hits, 1):
    doc_text = searcher.doc(h.docid).raw()
    results.append({
        "query": topic,
        "passage": doc_text,
        "docid": h.docid,
        "score": h.score,
        "relevance": qrels.get(qid, {}).get(h.docid, None)
    })

from pathlib import Path

# Create an output folder
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

# 1) File with topic at the top, then docid + relevance (and score for convenience)
labels_path = out_dir / f"labels_q{qid}.tsv"
with labels_path.open("w", encoding="utf-8", newline="") as f:
    f.write(f"# Query ID: {qid}\n")
    f.write(f"# Topic: {topic}\n")
    f.write("docid\trelevance\tscore\n")
    for r in results:
        f.write(f"{r['docid']}\t{r['relevance']}\t{r['score']:.4f}\n")

# 2) File with topic + each retrieved document’s text
docs_path = out_dir / f"topic_and_docs_q{qid}.txt"
with docs_path.open("w", encoding="utf-8") as f:
    f.write(f"Query ID: {qid}\nTopic: {topic}\n\n")
    for i, r in enumerate(results, 1):
        f.write(f"Doc {i}: {r['docid']} (rel={r['relevance']}, score={r['score']:.3f})\n")
        f.write("Passage:\n")
        f.write(r["passage"])
        f.write("\n" + "-"*80 + "\n\n")

print(f"Wrote: {labels_path}")
print(f"Wrote: {docs_path}")