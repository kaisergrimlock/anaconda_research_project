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

# 2) Build a TREC-style "qrels" dict from the dataset
# Build a nested dictionary of relevance judgments (qrels).
# Structure will be: { query_id: { doc_id: relevance_score, ... }, ... }
qrels = {}  # initialize empty qrels dictionary

# Iterate through all relevance judgments from the dataset
for q in ds.qrels_iter():
    # Ensure the dictionary has an entry for this query_id,
    # then assign the relevance score for the current doc_id.
    # Example: qrels['123']['DOC456'] = 1
    qrels.setdefault(str(q.query_id), {})[q.doc_id] = int(q.relevance)

# 3) Search all benchmark queries to build a TREC-style "run" dict
# The "run" dict will store the search results for each query:
# Structure: { query_id: { doc_id: score, ... }, ... }
run = {}
for query in ds.queries_iter():
    qid = str(query.query_id)  # Convert query ID to string for consistency
    text = query.text          # The actual query text
    hits = searcher.search(text, k=1000)  # Retrieve top 1000 hits for the query
    # Store the document IDs and their scores for this query
    run[qid] = {h.docid: float(h.score) for h in hits}

# 4) Evaluate with pytrec_eval
# Define the set of evaluation metrics to compute
metrics = {'map', 'ndcg_cut_10', 'recip_rank', 'P_10'}
# Create a RelevanceEvaluator with the qrels and desired metrics
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
# Evaluate the run against the qrels to get per-query results
results = evaluator.evaluate(run)

# 5) Aggregate and print
import numpy as np

def mean_metric(m):
    # Compute the mean value of metric 'm' across all queries
    return np.mean([res[m] for res in results.values()])

# Print the average values for each metric
print(f"MAP       : {mean_metric('map'):.4f}")
print(f"nDCG@10   : {mean_metric('ndcg_cut_10'):.4f}")
print(f"MRR       : {mean_metric('recip_rank'):.4f}")
print(f"P@10      : {mean_metric('P_10'):.4f}")
