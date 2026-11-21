# Evaluation: BEIR SciFact (test split)

- Corpus: 5,183 abstracts -> 7,610 chunks; 300 labelled test queries
- Embeddings: `BAAI/bge-small-en-v1.5`; reranker: `Xenova/ms-marco-MiniLM-L-6-v2`; 30 candidates per query
- Machine: Windows AMD64, CPU only, Python 3.12.10; indexing took 968.4 s

| Configuration | P@5 | Recall@10 | MRR@10 | nDCG@10 | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|---|
| BM25 (keyword) | 0.152 | 0.763 | 0.600 | 0.635 | 3.7 | 7.8 |
| Dense (vector) | 0.173 | 0.840 | 0.684 | 0.718 | 8.6 | 12.0 |
| Hybrid (BM25 + vector) | 0.169 | 0.850 | 0.685 | 0.721 | 12.9 | 20.9 |
| Hybrid + cross-encoder rerank | 0.165 | 0.842 | 0.666 | 0.703 | 1744.3 | 2225.3 |

Answer quality (extractive generator, first 100 queries, Hybrid + cross-encoder rerank):
groundedness 1.0, citation hit rate 0.6989, low-confidence fallbacks 7, p50 end-to-end 1675.9 ms, LLM tokens 0 (extractive mode, $0).

Reproduce with `python eval/evaluate.py`.
