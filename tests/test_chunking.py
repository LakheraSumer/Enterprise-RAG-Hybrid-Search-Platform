import pytest

from app.chunking import chunk_text
from app.text import count_tokens

TEXT = " ".join(f"Sentence number {i} talks about data pipelines and quality checks." for i in range(40))


def test_chunks_respect_token_budget():
    chunks = chunk_text(TEXT, max_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 50 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(TEXT, max_tokens=50, overlap_tokens=15)
    last_sentence_of_first = chunks[0].split(". ")[-1]
    assert last_sentence_of_first.rstrip(".") in chunks[1]


def test_no_text_is_lost():
    chunks = chunk_text(TEXT, max_tokens=50, overlap_tokens=0)
    assert " ".join(chunks) == " ".join(TEXT.split())


def test_long_sentence_is_split():
    long_sentence = " ".join(["word"] * 120)
    chunks = chunk_text(long_sentence, max_tokens=50, overlap_tokens=0)
    assert [count_tokens(c) for c in chunks] == [50, 50, 20]


def test_invalid_arguments():
    with pytest.raises(ValueError):
        chunk_text("x", max_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        chunk_text("x", max_tokens=0)
