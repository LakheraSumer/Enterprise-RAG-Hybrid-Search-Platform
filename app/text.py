"""Shared text helpers: tokenization, term extraction and sentence splitting."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+|[^\w\s]")
_WORD = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_CITATION = re.compile(r"\[\d+\]")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    may might of on or our should so such than that the their them then there these they this those to was we were
    what when where which who why will with would you your""".split()
)


def count_tokens(text: str) -> int:
    """Approximate token count (words and punctuation), used to size chunks and prompts."""
    return len(_TOKEN.findall(text))


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def terms(text: str) -> list[str]:
    """Lower-cased content words (stopwords removed) used by BM25 and answer scoring."""
    return [w for w in words(text) if w not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [" ".join(s.split()) for s in _SENTENCE.split(text) if s and s.strip()]


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text)


def fingerprint(text: str) -> str:
    """Whitespace/case-insensitive key used to drop duplicate chunks."""
    return " ".join(text.lower().split())
