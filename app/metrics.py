"""Retrieval and answer-quality metrics used by the evaluation harness."""
from __future__ import annotations

import math
from typing import Collection, Sequence

from .text import split_sentences, strip_citations, terms


def precision_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def mrr_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def groundedness(answer: str, contexts: Sequence[str], threshold: float = 0.6) -> float:
    """Share of answer sentences whose content words are mostly (>= threshold) found in the retrieved context."""
    context_terms = set()
    for c in contexts:
        context_terms.update(terms(c))
    sentences = [s for s in split_sentences(strip_citations(answer)) if terms(s)]
    if not sentences:
        return 0.0
    supported = sum(1 for s in sentences if sum(t in context_terms for t in terms(s)) / len(terms(s)) >= threshold)
    return supported / len(sentences)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[idx]
