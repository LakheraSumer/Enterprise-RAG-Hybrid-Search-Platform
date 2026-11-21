"""Grounded answer generation with source citations.

* ExtractiveGenerator (default, no API key): quotes the most relevant sentences from the retrieved chunks.
* LLMGenerator: any OpenAI-compatible chat API, prompted to answer only from numbered context; retried with
  backoff, and falls back to the extractive answer if the API keeps failing.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .config import Settings
from .resilience import TransientError, retry
from .retrieval import RetrievedChunk
from .text import count_tokens, split_sentences, terms

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "I could not find enough information in the indexed documents to answer this confidently."


@dataclass
class Answer:
    answer: str
    citations: list[dict] = field(default_factory=list)
    mode: str = "extractive"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


def _citation(index: int, r: RetrievedChunk) -> dict:
    return {"index": index, "chunk_id": r.chunk.id, "doc_id": r.chunk.doc_id, "title": r.chunk.title}


class Generator(Protocol):
    name: str

    def generate(self, question: str, results: list[RetrievedChunk]) -> Answer: ...


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, results: list[RetrievedChunk]) -> Answer:
        q = set(terms(question))
        scored = []
        for rank, r in enumerate(results, start=1):
            for position, sentence in enumerate(split_sentences(r.chunk.text)):
                overlap = len(q & set(terms(sentence)))
                if overlap:
                    scored.append((overlap / max(len(q), 1) + 0.25 / rank, rank, position, sentence))
        if not scored and results:
            first = split_sentences(results[0].chunk.text)
            scored = [(0.0, 1, 0, first[0])] if first else []
        picked = sorted(sorted(scored, key=lambda s: -s[0])[: self.max_sentences], key=lambda s: (s[1], s[2]))
        parts = [f"{sentence} [{rank}]" for _, rank, _, sentence in picked]
        used = sorted({rank for _, rank, _, _ in picked})
        return Answer(" ".join(parts), [_citation(i, results[i - 1]) for i in used], mode=self.name)


class LLMGenerator:
    name = "llm"

    SYSTEM_PROMPT = (
        "You answer questions using ONLY the numbered context passages. Cite every claim with its passage "
        "number in square brackets, e.g. [2]. If the context does not contain the answer, say you don't know."
    )

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 20.0,
        client: httpx.Client | None = None,
        price_in: float = 0.0,
        price_out: float = 0.0,
        fallback: Generator | None = None,
        sleep=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = client or httpx.Client(timeout=timeout)
        self.price_in = price_in
        self.price_out = price_out
        self.fallback = fallback or ExtractiveGenerator()
        self._retry_kwargs = {"sleep": sleep} if sleep else {}

    def _call(self, messages: list[dict]) -> dict:
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json={"model": self.model, "messages": messages, "temperature": 0},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        except httpx.TransportError as exc:  # timeouts, connection resets
            raise TransientError(str(exc)) from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientError(f"HTTP {response.status_code}")
        response.raise_for_status()
        return response.json()

    def generate(self, question: str, results: list[RetrievedChunk]) -> Answer:
        context = "\n\n".join(f"[{i}] {r.chunk.title}: {r.chunk.text}" for i, r in enumerate(results, start=1))
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        try:
            data = retry(lambda: self._call(messages), attempts=3, **self._retry_kwargs)
        except (TransientError, httpx.HTTPError) as exc:
            logger.warning("LLM unavailable (%s); using extractive fallback", exc)
            answer = self.fallback.generate(question, results)
            answer.mode = "extractive-fallback"
            return answer

        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or count_tokens(self.SYSTEM_PROMPT + messages[1]["content"]))
        completion_tokens = int(usage.get("completion_tokens") or count_tokens(text))
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(results)})
        cost = prompt_tokens / 1000 * self.price_in + completion_tokens / 1000 * self.price_out
        return Answer(text, [_citation(i, results[i - 1]) for i in cited], self.name, prompt_tokens, completion_tokens, round(cost, 6))


def make_generator(settings: Settings) -> Generator:
    if settings.llm_base_url and settings.llm_api_key:
        return LLMGenerator(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_model,
            settings.llm_timeout_seconds,
            price_in=settings.price_per_1k_input_tokens,
            price_out=settings.price_per_1k_output_tokens,
        )
    return ExtractiveGenerator()
