"""Vector stores: in-memory (NumPy) and PostgreSQL + pgvector (HNSW index, JSONB metadata filters)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .config import Settings


@dataclass
class StoredChunk:
    id: str
    doc_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def matches(metadata: dict[str, str], filters: dict[str, str] | None) -> bool:
    return all(metadata.get(k) == str(v) for k, v in (filters or {}).items())


class VectorStore(Protocol):
    name: str

    def upsert(self, chunks: Sequence[StoredChunk], vectors: np.ndarray) -> None: ...

    def delete_document(self, doc_id: str) -> list[str]: ...

    def all_chunks(self) -> list[StoredChunk]: ...

    def search(self, vector: np.ndarray, top_k: int, filters: dict[str, str] | None = None) -> list[tuple[str, float]]: ...


class InMemoryVectorStore:
    name = "memory"

    def __init__(self) -> None:
        self._chunks: dict[str, StoredChunk] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def upsert(self, chunks: Sequence[StoredChunk], vectors: np.ndarray) -> None:
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = np.asarray(vector, dtype=np.float32)
        self._matrix = None

    def delete_document(self, doc_id: str) -> list[str]:
        removed = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        for cid in removed:
            del self._chunks[cid]
            del self._vectors[cid]
        if removed:
            self._matrix = None
        return removed

    def all_chunks(self) -> list[StoredChunk]:
        return list(self._chunks.values())

    def search(self, vector: np.ndarray, top_k: int, filters: dict[str, str] | None = None) -> list[tuple[str, float]]:
        if self._matrix is None:
            self._ids = list(self._chunks)
            self._matrix = np.vstack([self._vectors[i] for i in self._ids]) if self._ids else None
        if self._matrix is None:
            return []
        sims = self._matrix @ np.asarray(vector, dtype=np.float32)
        out: list[tuple[str, float]] = []
        for i in np.argsort(-sims):
            cid = self._ids[i]
            if filters and not matches(self._chunks[cid].metadata, filters):
                continue
            out.append((cid, float(sims[i])))
            if len(out) >= top_k:
                break
        return out


class PgVectorStore:
    """Chunks, metadata and embeddings in one PostgreSQL table, searched by cosine distance (<=>)."""

    name = "pgvector"

    def __init__(self, dsn: str, dim: int, table: str = "rag_chunks") -> None:
        import psycopg
        from psycopg.types.json import Jsonb

        if not table.isidentifier():
            raise ValueError("table must be a plain identifier")
        self._Jsonb = Jsonb
        self.table = table
        self._conn = psycopg.connect(dsn, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {table} (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({int(dim)}) NOT NULL)"""
            )
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_doc_idx ON {table} (doc_id)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_meta_idx ON {table} USING GIN (metadata)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx ON {table} USING hnsw (embedding vector_cosine_ops)")

    @staticmethod
    def _vec(vector: np.ndarray) -> str:
        return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"

    def upsert(self, chunks: Sequence[StoredChunk], vectors: np.ndarray) -> None:
        rows = [(c.id, c.doc_id, c.title, c.text, self._Jsonb(c.metadata), self._vec(v)) for c, v in zip(chunks, vectors)]
        with self._conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table} (id, doc_id, title, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET doc_id = EXCLUDED.doc_id, title = EXCLUDED.title,
                        text = EXCLUDED.text, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                rows,
            )

    def delete_document(self, doc_id: str) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE doc_id = %s RETURNING id", (doc_id,))
            return [r[0] for r in cur.fetchall()]

    def all_chunks(self) -> list[StoredChunk]:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT id, doc_id, title, text, metadata FROM {self.table}")
            return [StoredChunk(r[0], r[1], r[2], r[3], dict(r[4] or {})) for r in cur.fetchall()]

    def search(self, vector: np.ndarray, top_k: int, filters: dict[str, str] | None = None) -> list[tuple[str, float]]:
        v = self._vec(vector)
        sql = f"SELECT id, 1 - (embedding <=> %s::vector) AS score FROM {self.table}"
        params: list = [v]
        if filters:
            sql += " WHERE metadata @> %s"
            params.append(self._Jsonb({k: str(val) for k, val in filters.items()}))
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params += [v, int(top_k)]
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [(r[0], float(r[1])) for r in cur.fetchall()]

    def drop(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table}")


def make_store(settings: Settings, dim: int) -> VectorStore:
    if settings.database_url:
        return PgVectorStore(settings.database_url, dim)
    return InMemoryVectorStore()
