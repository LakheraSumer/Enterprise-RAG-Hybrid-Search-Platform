"""RAG pipeline: ingestion (chunk -> dedupe -> embed -> index) and querying (retrieve -> confidence gate -> answer)."""
from __future__ import annotations

import hashlib
import threading
import time
from typing import Iterable

from .bm25 import BM25Index
from .cache import Cache, cache_key, make_cache
from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder, make_embedder
from .generation import FALLBACK_ANSWER, Answer, Generator, make_generator
from .models import DocumentIn
from .reranker import Reranker, make_reranker
from .retrieval import HybridRetriever, RetrievedChunk, confidence
from .text import fingerprint
from .vector_store import StoredChunk, VectorStore, make_store

_DEFAULT = object()


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        reranker: Reranker | None | object = _DEFAULT,
        cache: Cache | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or make_embedder(settings)
        self.store = store or make_store(settings, self.embedder.dim)
        self.reranker = make_reranker(settings) if reranker is _DEFAULT else reranker
        self.cache = cache or make_cache(settings)
        self.generator = generator or make_generator(settings)
        self.bm25 = BM25Index()
        self._chunks: dict[str, StoredChunk] = {}
        self._lock = threading.RLock()
        self._version = 0
        # rebuild the in-process keyword index from whatever the vector store already holds
        for chunk in self.store.all_chunks():
            self._chunks[chunk.id] = chunk
            self.bm25.add(chunk.id, f"{chunk.title} {chunk.text}")
        self.retriever = HybridRetriever(
            self.store, self.bm25, self.embedder, self._chunks, self.reranker, settings.dense_weight, settings.candidate_k
        )

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _forget(self, chunk_ids: Iterable[str]) -> None:
        for cid in chunk_ids:
            self._chunks.pop(cid, None)
            self.bm25.remove(cid)

    def ingest(self, documents: list[DocumentIn]) -> dict:
        """Idempotent: re-ingesting a document id replaces its previous chunks."""
        started = time.perf_counter()
        with self._lock:
            prepared = []
            for doc in documents:
                doc_id = doc.id or hashlib.sha1(f"{doc.title}\n{doc.text}".encode()).hexdigest()[:16]
                self._forget(self.store.delete_document(doc_id))
                prepared.append((doc_id, doc))

            seen = {fingerprint(c.text) for c in self._chunks.values()}
            new_chunks: list[StoredChunk] = []
            duplicates = 0
            for doc_id, doc in prepared:
                for i, piece in enumerate(chunk_text(doc.text, self.settings.chunk_tokens, self.settings.chunk_overlap)):
                    key = fingerprint(piece)
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    new_chunks.append(StoredChunk(f"{doc_id}::{i}", doc_id, doc.title, piece, dict(doc.metadata)))

            if new_chunks:
                vectors = self.embedder.embed([f"{c.title}\n{c.text}" for c in new_chunks])
                self.store.upsert(new_chunks, vectors)
                for c in new_chunks:
                    self._chunks[c.id] = c
                    self.bm25.add(c.id, f"{c.title} {c.text}")
            self._version += 1  # invalidates cached answers
        return {
            "documents": len(documents),
            "chunks_indexed": len(new_chunks),
            "duplicates_skipped": duplicates,
            "seconds": round(time.perf_counter() - started, 3),
        }

    def search(
        self, query: str, top_k: int = 5, mode: str = "hybrid", filters: dict[str, str] | None = None, rerank: bool = True
    ) -> list[RetrievedChunk]:
        with self._lock:
            return self.retriever.retrieve(query, top_k, mode, filters, rerank)

    def query(
        self, question: str, top_k: int = 5, mode: str = "hybrid", filters: dict[str, str] | None = None, rerank: bool = True
    ) -> dict:
        key = cache_key(self._version, question.strip().lower(), top_k, mode, filters, rerank)
        cached = self.cache.get(key)
        if cached is not None:
            cached["cached"] = True
            return cached

        started = time.perf_counter()
        results = self.search(question, top_k, mode, filters, rerank)
        score = confidence(results)
        if not results or score < self.settings.min_confidence:
            answer = Answer(FALLBACK_ANSWER, mode="low-confidence-fallback")
        else:
            answer = self.generator.generate(question, results)
        payload = {
            "question": question,
            "answer": answer.answer,
            "citations": answer.citations,
            "confidence": round(score, 4),
            "answer_mode": answer.mode,
            "sources": [r.to_dict() for r in results],
            "usage": {
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
                "cost_usd": answer.cost_usd,
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "cached": False,
        }
        self.cache.set(key, payload)
        return payload
