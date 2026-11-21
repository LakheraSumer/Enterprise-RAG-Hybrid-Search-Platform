"""Embedding providers. All return L2-normalised float32 vectors, so a dot product is cosine similarity."""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

from .config import Settings
from .text import words


class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic feature-hashing embedder (word unigrams/bigrams + character trigrams).

    It needs no model download, so tests and CI run offline. It captures lexical and sub-word
    similarity only; use the fastembed provider for semantic search.
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _features(text: str) -> list[str]:
        tokens = words(text)
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for token in tokens:
            padded = f"#{token}#"
            feats.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
                out[row, h % self._dim] += 1.0 if (h >> 63) == 0 else -1.0
        return _normalize(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class FastEmbedEmbedder:
    """Semantic embeddings (default BAAI/bge-small-en-v1.5, 384-d) running locally on ONNX Runtime."""

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed_query("dimension probe").shape[0])
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.asarray(list(self._model.passage_embed(list(texts))), dtype=np.float32)
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)
        return _normalize(vector[None, :])[0]


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hashing":
        return HashingEmbedder()
    if settings.embedding_provider == "fastembed":
        return FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
