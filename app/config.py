"""Runtime settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _get(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # "hashing" = dependency-free local embedder (tests/offline); "fastembed" = BAAI/bge-small-en-v1.5 via ONNX
    embedding_provider: str = "hashing"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # "none" or "cross-encoder" (ms-marco MiniLM cross-encoder via ONNX)
    reranker: str = "none"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_dir: str | None = None
    # PostgreSQL + pgvector when set, otherwise an in-memory vector store
    database_url: str | None = None
    # Redis answer cache when set, otherwise an in-memory TTL cache
    redis_url: str | None = None
    chunk_tokens: int = 200
    chunk_overlap: int = 40
    dense_weight: float = 0.5
    candidate_k: int = 30
    min_confidence: float = 0.2
    cache_ttl_seconds: int = 600
    # Optional OpenAI-compatible LLM; without it answers are extractive (quoted from sources, with citations)
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 20.0
    price_per_1k_input_tokens: float = 0.0
    price_per_1k_output_tokens: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            embedding_provider=_get("EMBEDDING_PROVIDER", d.embedding_provider),
            embedding_model=_get("EMBEDDING_MODEL", d.embedding_model),
            reranker=_get("RERANKER", d.reranker),
            reranker_model=_get("RERANKER_MODEL", d.reranker_model),
            model_cache_dir=_get("MODEL_CACHE_DIR"),
            database_url=_get("DATABASE_URL"),
            redis_url=_get("REDIS_URL"),
            chunk_tokens=int(_get("CHUNK_TOKENS", str(d.chunk_tokens))),
            chunk_overlap=int(_get("CHUNK_OVERLAP", str(d.chunk_overlap))),
            dense_weight=float(_get("DENSE_WEIGHT", str(d.dense_weight))),
            candidate_k=int(_get("CANDIDATE_K", str(d.candidate_k))),
            min_confidence=float(_get("MIN_CONFIDENCE", str(d.min_confidence))),
            cache_ttl_seconds=int(_get("CACHE_TTL_SECONDS", str(d.cache_ttl_seconds))),
            llm_base_url=_get("LLM_BASE_URL"),
            llm_api_key=_get("LLM_API_KEY"),
            llm_model=_get("LLM_MODEL", d.llm_model),
            llm_timeout_seconds=float(_get("LLM_TIMEOUT_SECONDS", str(d.llm_timeout_seconds))),
            price_per_1k_input_tokens=float(_get("PRICE_PER_1K_INPUT_TOKENS", "0")),
            price_per_1k_output_tokens=float(_get("PRICE_PER_1K_OUTPUT_TOKENS", "0")),
        )
