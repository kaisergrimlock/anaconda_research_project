from pyserini.search.lucene import LuceneSearcher

# Download and open a small, ready-made index (first run will fetch to your cache)
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

# Optional: tweak BM25 params
searcher.set_bm25(k1=0.82, b=0.68)

# Run a query
hits = searcher.search('what is the capital of australia?', k=5)

for i, h in enumerate(hits, 1):
    print(f'{i:2d}. docid={h.docid} score={h.score:.3f}')
    # If you want the raw text:
    # print(searcher.doc(h.docid).raw())
