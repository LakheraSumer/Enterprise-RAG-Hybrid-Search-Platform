"""Hybrid retrieval: dense (vector) + sparse (BM25) candidates, min-max score normalisation, weighted fusion,
duplicate removal, metadata filters and optional cross-encoder reranking."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .bm25 import BM25Index
from .embeddings import Embedder
from .reranker import Reranker
from .text import fingerprint
from .vector_store import StoredChunk, VectorStore, matches

MODES = ("hybrid", "dense", "sparse")


@dataclass
class RetrievedChunk:
    chunk: StoredChunk
    score: float  # final ranking score
    dense: float | None = None  # raw cosine similarity
    sparse: float | None = None  # raw BM25 score
    rerank: float | None = None  # raw cross-encoder logit

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata,
            "score": round(self.score, 6),
            "dense_score": None if self.dense is None else round(self.dense, 6),
            "bm25_score": None if self.sparse is None else round(self.sparse, 6),
            "rerank_score": None if self.rerank is None else round(self.rerank, 6),
        }


def min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def confidence(results: list[RetrievedChunk]) -> float:
    """Best evidence of relevance among the results: sigmoid(cross-encoder logit) if reranked,
    else the highest raw cosine similarity. Pure keyword results have no calibrated score (0.5)."""
    best = 0.0
    seen_signal = False
    for r in results:
        if r.rerank is not None:
            best, seen_signal = max(best, 1.0 / (1.0 + math.exp(-r.rerank))), True
        elif r.dense is not None:
            best, seen_signal = max(best, r.dense), True
    if not results:
        return 0.0
    return best if seen_signal else 0.5


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
        chunks: dict[str, StoredChunk],
        reranker: Reranker | None = None,
        dense_weight: float = 0.5,
        candidate_k: int = 30,
    ) -> None:
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1")
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.chunks = chunks
        self.reranker = reranker
        self.dense_weight = dense_weight
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        filters: dict[str, str] | None = None,
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        k = max(self.candidate_k, top_k)

        dense: dict[str, float] = {}
        sparse: dict[str, float] = {}
        if mode in ("dense", "hybrid"):
            dense = dict(self.store.search(self.embedder.embed_query(query), k, filters))
        if mode in ("sparse", "hybrid"):
            allowed = {cid for cid, c in self.chunks.items() if matches(c.metadata, filters)} if filters else None
            sparse = dict(self.bm25.search(query, k, allowed))

        weight = {"dense": 1.0, "sparse": 0.0}.get(mode, self.dense_weight)
        nd, ns = min_max(dense), min_max(sparse)
        fused = {cid: weight * nd.get(cid, 0.0) + (1.0 - weight) * ns.get(cid, 0.0) for cid in nd.keys() | ns.keys()}

        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        for cid in sorted(fused, key=lambda c: (-fused[c], c)):
            chunk = self.chunks.get(cid)
            if chunk is None:
                continue
            key = fingerprint(chunk.text)
            if key in seen:  # duplicate removal (same text from different sources)
                continue
            seen.add(key)
            results.append(RetrievedChunk(chunk, fused[cid], dense.get(cid), sparse.get(cid)))
            if len(results) >= k:
                break

        if rerank and self.reranker is not None and results:
            for r, s in zip(results, self.reranker.score(query, [f"{r.chunk.title}. {r.chunk.text}" for r in results])):
                r.rerank = s
                r.score = s
            results.sort(key=lambda r: -r.score)
        return results[:top_k]
