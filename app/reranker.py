"""Cross-encoder reranking (ms-marco MiniLM via fastembed/ONNX): scores each (query, chunk) pair jointly."""
from __future__ import annotations

from typing import Protocol, Sequence

from .config import Settings


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(texts))]


def make_reranker(settings: Settings) -> Reranker | None:
    if settings.reranker == "none":
        return None
    if settings.reranker == "cross-encoder":
        return CrossEncoderReranker(settings.reranker_model, settings.model_cache_dir)
    raise ValueError(f"Unknown RERANKER: {settings.reranker!r}")
