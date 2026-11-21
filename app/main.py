"""FastAPI service: async (background) document indexing, hybrid search and grounded question answering."""
from __future__ import annotations

import logging
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException

from . import __version__
from .config import Settings
from .models import IngestRequest, QueryRequest, SearchRequest
from .pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rag.api")


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    state: dict = {"pipeline": pipeline}
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["pipeline"] is None:
            state["pipeline"] = RAGPipeline(Settings.from_env())
        logger.info("pipeline ready: %d chunks indexed", state["pipeline"].chunk_count)
        yield

    app = FastAPI(title="Enterprise RAG & Hybrid Search Platform", version=__version__, lifespan=lifespan)

    def _pipeline() -> RAGPipeline:
        return state["pipeline"]

    def _run_ingest(job_id: str, request: IngestRequest) -> None:
        with jobs_lock:
            jobs[job_id]["status"] = "running"
        try:
            result = _pipeline().ingest(request.documents)
            with jobs_lock:
                jobs[job_id].update(status="completed", result=result)
            logger.info("job %s completed: %s", job_id, result)
        except Exception as exc:  # surfaced through GET /jobs/{id}
            logger.exception("job %s failed", job_id)
            with jobs_lock:
                jobs[job_id].update(status="failed", error=str(exc))

    @app.get("/health")
    def health() -> dict:
        p = _pipeline()
        return {
            "status": "ok",
            "version": __version__,
            "chunks": p.chunk_count,
            "vector_store": p.store.name,
            "embedder": p.embedder.name,
            "reranker": getattr(p.reranker, "name", None),
            "generator": p.generator.name,
            "cache": p.cache.name,
        }

    @app.post("/documents", status_code=202)
    def add_documents(request: IngestRequest, background: BackgroundTasks) -> dict:
        job_id = uuid.uuid4().hex
        with jobs_lock:
            jobs[job_id] = {"job_id": job_id, "status": "queued", "documents": len(request.documents)}
        background.add_task(_run_ingest, job_id, request)
        return {"job_id": job_id, "status": "queued"}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/search")
    def search(request: SearchRequest) -> dict:
        results = _pipeline().search(request.query, request.top_k, request.mode, request.filters, request.rerank)
        return {"query": request.query, "mode": request.mode, "results": [r.to_dict() for r in results]}

    @app.post("/query")
    def query(request: QueryRequest) -> dict:
        return _pipeline().query(request.question, request.top_k, request.mode, request.filters, request.rerank)

    return app


app = create_app()
