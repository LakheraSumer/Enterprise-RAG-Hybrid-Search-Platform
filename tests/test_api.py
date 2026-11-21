from fastapi.testclient import TestClient

from app.cache import MemoryTTLCache
from app.main import create_app
from app.pipeline import RAGPipeline
from app.vector_store import InMemoryVectorStore


def test_end_to_end_api(settings, sample_docs):
    pipeline = RAGPipeline(settings, store=InMemoryVectorStore(), cache=MemoryTTLCache())
    with TestClient(create_app(pipeline)) as client:
        assert client.get("/health").json()["chunks"] == 0

        body = {"documents": [d.model_dump() for d in sample_docs]}
        accepted = client.post("/documents", json=body)
        assert accepted.status_code == 202
        job = client.get(f"/jobs/{accepted.json()['job_id']}").json()
        assert job["status"] == "completed" and job["result"]["chunks_indexed"] > 0

        search = client.post("/search", json={"query": "broadcast join spark cost", "top_k": 3}).json()
        assert search["results"][0]["doc_id"] == "guide-spark-cost"

        answer = client.post("/query", json={"question": "Within how many days must deletion requests be completed?"}).json()
        assert "30 days" in answer["answer"]
        assert answer["citations"][0]["doc_id"] == "policy-data-retention"

        assert client.post("/query", json={"question": "x"}).status_code == 422
        assert client.get("/jobs/unknown").status_code == 404
