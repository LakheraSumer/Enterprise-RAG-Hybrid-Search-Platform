import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cache import MemoryTTLCache  # noqa: E402
from app.config import Settings  # noqa: E402
from app.models import DocumentIn  # noqa: E402
from app.pipeline import RAGPipeline  # noqa: E402
from app.vector_store import InMemoryVectorStore  # noqa: E402


@pytest.fixture
def sample_docs() -> list[DocumentIn]:
    raw = json.loads((ROOT / "data" / "sample" / "documents.json").read_text(encoding="utf-8"))
    return [DocumentIn(**d) for d in raw]


@pytest.fixture
def settings() -> Settings:
    return Settings(embedding_provider="hashing", reranker="none", chunk_tokens=60, chunk_overlap=10)


@pytest.fixture
def pipeline(settings, sample_docs) -> RAGPipeline:
    p = RAGPipeline(settings, store=InMemoryVectorStore(), cache=MemoryTTLCache())
    p.ingest(sample_docs)
    return p
