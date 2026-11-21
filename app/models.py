"""API request/response schemas (validated by Pydantic)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["hybrid", "dense", "sparse"]


class DocumentIn(BaseModel):
    id: str | None = Field(default=None, max_length=200, description="Stable id; re-ingesting the same id replaces it")
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict, description="Exact-match filter fields, e.g. department")


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=1000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Mode = "hybrid"
    filters: dict[str, str] | None = None
    rerank: bool = True


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Mode = "hybrid"
    filters: dict[str, str] | None = None
    rerank: bool = True
