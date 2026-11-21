import numpy as np
import pytest

from app.bm25 import BM25Index
from app.embeddings import HashingEmbedder
from app.metrics import groundedness, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from app.retrieval import min_max


def test_bm25_ranks_exact_keyword_match_first():
    index = BM25Index()
    index.add("a", "Broadcast small dimension tables in Spark joins")
    index.add("b", "Data retention for bronze tables is 400 days")
    index.add("c", "Standups are at ten in the morning")
    assert index.search("retention days bronze", top_k=3)[0][0] == "b"


def test_bm25_remove_and_filter():
    index = BM25Index()
    index.add("a", "spark shuffle cost")
    index.add("b", "spark broadcast join")
    index.remove("a")
    assert [d for d, _ in index.search("spark")] == ["b"]
    assert index.search("spark", allowed={"a"}) == []


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder()
    v1, v2 = e.embed(["data quality checks", "data quality checks"])
    assert np.isclose(np.linalg.norm(v1), 1.0)
    assert np.allclose(v1, v2)
    near, far = e.embed(["data quality check", "team standup time"])
    assert v1 @ near > v1 @ far


def test_min_max():
    assert min_max({"a": 2.0, "b": 4.0, "c": 3.0}) == {"a": 0.0, "b": 1.0, "c": 0.5}
    assert min_max({"a": 5.0}) == {"a": 1.0}


@pytest.mark.parametrize("mode", ["hybrid", "dense", "sparse"])
def test_pipeline_finds_relevant_document(pipeline, mode):
    results = pipeline.search("How long is bronze raw data retained?", top_k=3, mode=mode)
    assert results[0].chunk.doc_id == "policy-data-retention"


def test_metadata_filter_restricts_results(pipeline):
    results = pipeline.search("data access and retention", top_k=10, filters={"department": "people"})
    assert results and all(r.chunk.metadata["department"] == "people" for r in results)


def test_duplicate_chunks_are_skipped(pipeline, sample_docs):
    copy = sample_docs[0].model_copy(update={"id": "duplicate-of-runbook"})
    result = pipeline.ingest([copy])
    assert result["chunks_indexed"] == 0 and result["duplicates_skipped"] > 0


def test_reingest_same_id_replaces_chunks(pipeline, sample_docs):
    before = pipeline.chunk_count
    pipeline.ingest([sample_docs[1]])
    assert pipeline.chunk_count == before


def test_metrics():
    ranked = ["d3", "d1", "d7"]
    assert precision_at_k(ranked, {"d1"}, 2) == 0.5
    assert recall_at_k(ranked, {"d1", "d9"}, 3) == 0.5
    assert mrr_at_k(ranked, {"d1"}, 10) == 0.5
    assert ndcg_at_k(["d1"], {"d1"}, 10) == 1.0
    assert groundedness("Bronze data is kept 400 days [1].", ["Raw bronze data is kept for 400 days."]) == 1.0
    assert groundedness("Pizza is served on Fridays [1].", ["Raw bronze data is kept for 400 days."]) == 0.0
