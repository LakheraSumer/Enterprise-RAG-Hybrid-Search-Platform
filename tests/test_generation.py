import httpx
import pytest

from app.generation import FALLBACK_ANSWER, ExtractiveGenerator, LLMGenerator
from app.resilience import TransientError, retry


def test_extractive_answer_cites_sources(pipeline):
    results = pipeline.search("How long are silver tables retained?", top_k=3)
    answer = ExtractiveGenerator().generate("How long are silver tables retained?", results)
    assert "7 years" in answer.answer
    assert answer.citations and answer.citations[0]["doc_id"] == "policy-data-retention"
    assert "[1]" in answer.answer


def test_low_confidence_question_gets_fallback(pipeline):
    from dataclasses import replace

    pipeline.settings = replace(pipeline.settings, min_confidence=0.99)
    response = pipeline.query("What is the price of gold coins in Tokyo?")
    assert response["answer"] == FALLBACK_ANSWER
    assert response["answer_mode"] == "low-confidence-fallback"


def test_query_results_are_cached(pipeline):
    first = pipeline.query("What is the data quality target?")
    second = pipeline.query("What is the data quality target?")
    assert first["cached"] is False and second["cached"] is True
    assert first["answer"] == second["answer"]


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("503")
        return "ok"

    assert retry(flaky, attempts=3, sleep=lambda _: None) == "ok"
    assert calls["n"] == 3


def test_retry_gives_up():
    with pytest.raises(TransientError):
        retry(lambda: (_ for _ in ()).throw(TransientError("429")), attempts=2, sleep=lambda _: None)


def _llm(handler) -> LLMGenerator:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return LLMGenerator("https://llm.example/v1", "test-key", "test-model", client=client,
                        price_in=0.15, price_out=0.6, sleep=lambda _: None)


def test_llm_answer_citations_and_cost(pipeline):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Silver tables are kept for 7 years [1]."}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        })

    results = pipeline.search("silver retention", top_k=3)
    answer = _llm(handler).generate("How long are silver tables kept?", results)
    assert answer.mode == "llm"
    assert [c["index"] for c in answer.citations] == [1]
    assert answer.cost_usd == pytest.approx(0.15 + 0.06)


def test_llm_outage_falls_back_to_extractive(pipeline):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    results = pipeline.search("silver retention", top_k=3)
    answer = _llm(handler).generate("How long are silver tables kept?", results)
    assert calls["n"] == 3
    assert answer.mode == "extractive-fallback" and answer.citations
