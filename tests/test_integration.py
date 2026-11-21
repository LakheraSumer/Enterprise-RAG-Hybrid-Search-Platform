"""Runs against real PostgreSQL + pgvector and Redis (docker compose up postgres redis), skipped otherwise.

    TEST_DATABASE_URL=postgresql://rag:rag@localhost:5433/rag TEST_REDIS_URL=redis://localhost:6380/0 pytest
"""
import os
import uuid

import pytest

from app.cache import RedisCache
from app.pipeline import RAGPipeline
from app.vector_store import PgVectorStore

DB_URL = os.getenv("TEST_DATABASE_URL")
REDIS_URL = os.getenv("TEST_REDIS_URL")


@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
def test_pgvector_store_round_trip(settings, sample_docs):
    store = PgVectorStore(DB_URL, dim=512, table=f"test_chunks_{uuid.uuid4().hex[:8]}")
    try:
        pipeline = RAGPipeline(settings, store=store)
        pipeline.ingest(sample_docs)
        results = pipeline.search("How long is bronze raw data retained?", top_k=3)
        assert results[0].chunk.doc_id == "policy-data-retention"
        filtered = pipeline.search("access", top_k=5, filters={"department": "governance"}, mode="dense")
        assert filtered and all(r.chunk.metadata["department"] == "governance" for r in filtered)

        # a new pipeline on the same table rebuilds its keyword index from Postgres
        reopened = RAGPipeline(settings, store=store)
        assert reopened.chunk_count == pipeline.chunk_count
        assert reopened.search("broadcast join", top_k=1, mode="sparse")[0].chunk.doc_id == "guide-spark-cost"
    finally:
        store.drop()


@pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")
def test_redis_cache(settings, sample_docs):
    from app.vector_store import InMemoryVectorStore

    pipeline = RAGPipeline(settings, store=InMemoryVectorStore(), cache=RedisCache(REDIS_URL, ttl_seconds=30))
    pipeline.ingest(sample_docs)
    question = f"What is the data quality target? {uuid.uuid4().hex[:6]}"
    assert pipeline.query(question)["cached"] is False
    assert pipeline.query(question)["cached"] is True


def test_redis_outage_does_not_break_queries(settings, sample_docs):
    from app.vector_store import InMemoryVectorStore

    pipeline = RAGPipeline(settings, store=InMemoryVectorStore(), cache=RedisCache("redis://localhost:1/0"))
    pipeline.ingest(sample_docs)
    assert "99 percent" in pipeline.query("What is the monthly data quality target?")["answer"]
