"""Token-aware chunking: packs whole sentences into chunks of at most `max_tokens`, with sentence-level overlap."""
from __future__ import annotations

from .text import count_tokens, split_sentences


def _split_long(sentence: str, max_tokens: int) -> list[str]:
    """Break a sentence that alone exceeds the budget into word windows."""
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in sentence.split():
        n = count_tokens(word)
        if current and current_tokens + n > max_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(text: str, max_tokens: int = 200, overlap_tokens: int = 40) -> list[str]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be >= 0 and smaller than max_tokens")

    units: list[str] = []
    for sentence in split_sentences(text):
        if count_tokens(sentence) > max_tokens:
            units.extend(_split_long(sentence, max_tokens))
        else:
            units.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        n = count_tokens(unit)
        if current and current_tokens + n > max_tokens:
            chunks.append(" ".join(current))
            # carry trailing sentences forward as overlap, as long as they fit the overlap budget
            carry: list[str] = []
            carry_tokens = 0
            for previous in reversed(current):
                p = count_tokens(previous)
                if carry_tokens + p > overlap_tokens:
                    break
                carry.insert(0, previous)
                carry_tokens += p
            if carry_tokens + n > max_tokens:
                carry, carry_tokens = [], 0
            current, current_tokens = carry, carry_tokens
        current.append(unit)
        current_tokens += n
    if current:
        chunks.append(" ".join(current))
    return chunks
