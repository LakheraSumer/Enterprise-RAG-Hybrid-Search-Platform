"""Incremental Okapi BM25 keyword index."""
from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from .text import terms


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tf: dict[str, Counter] = {}
        self._len: dict[str, int] = {}
        self._df: Counter = Counter()
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._tf)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._tf:
            self.remove(doc_id)
        tokens = terms(text)
        tf = Counter(tokens)
        self._tf[doc_id] = tf
        self._len[doc_id] = len(tokens)
        self._total_len += len(tokens)
        for term in tf:
            self._df[term] += 1
            self._postings[term].add(doc_id)

    def remove(self, doc_id: str) -> None:
        tf = self._tf.pop(doc_id, None)
        if tf is None:
            return
        self._total_len -= self._len.pop(doc_id)
        for term in tf:
            self._df[term] -= 1
            self._postings[term].discard(doc_id)
            if self._df[term] <= 0:
                del self._df[term]
                del self._postings[term]

    def search(self, query: str, top_k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        n = len(self._tf)
        if n == 0:
            return []
        avgdl = (self._total_len / n) or 1.0
        scores: dict[str, float] = defaultdict(float)
        for term in set(terms(query)):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for doc_id in self._postings[term]:
                if allowed is not None and doc_id not in allowed:
                    continue
                f = self._tf[doc_id][term]
                norm = 1.0 - self.b + self.b * self._len[doc_id] / avgdl
                scores[doc_id] += idf * f * (self.k1 + 1.0) / (f + self.k1 * norm)
        return heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
