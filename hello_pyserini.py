import os
os.environ['PYSERINI_CACHE'] = r'D:\PyseriniCache'
os.environ['IR_DATASETS_HOME'] = r'D:\ir_datasets'

# 1) Setup: searcher + datasets + evaluator
from pyserini.search.lucene import LuceneSearcher
import ir_datasets
import pytrec_eval

# Use a small, ready-made index
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

# Configure the BM25 ranking function with custom parameters:
#   k1 controls term saturation (how quickly term frequency gains level off).
#   b controls length normalization (how much document length affects the score).
#   Lower k1 (0.82) = term frequency has less impact.
#   Lower b (0.68) = document length normalization is reduced.
searcher.set_bm25(k1=0.82, b=0.68)

# Load a benchmark split that includes queries & qrels
ds = ir_datasets.load('msmarco-passage/dev/small')

# Build qrels dict: {qid: {docid: rel, ...}, ...}
qrels = {}
for q in ds.qrels_iter():
    qrels.setdefault(str(q.query_id), {})[q.doc_id] = int(q.relevance)

# 2) Search all benchmark queries to build a TREC-style "run" dict
run = {}  # {qid: {docid: score, ...}}
for query in ds.queries_iter():
    qid = str(query.query_id)
    text = query.text
    hits = searcher.search(text, k=1000)
    run[qid] = {h.docid: float(h.score) for h in hits}

# 3) Evaluate with pytrec_eval
metrics = {'map', 'ndcg_cut_10', 'recip_rank', 'P_10'}
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
results = evaluator.evaluate(run)

# Aggregate and print
import numpy as np

def mean_metric(m):
    return np.mean([res[m] for res in results.values()])

print(f"MAP       : {mean_metric('map'):.4f}")
print(f"nDCG@10   : {mean_metric('ndcg_cut_10'):.4f}")
print(f"MRR       : {mean_metric('recip_rank'):.4f}")
print(f"P@10      : {mean_metric('P_10'):.4f}")
