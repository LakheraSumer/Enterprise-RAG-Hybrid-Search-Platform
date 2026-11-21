"""Retry with exponential backoff and jitter for calls to external services."""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


class TransientError(Exception):
    """A failure worth retrying (rate limit, 5xx, timeout)."""


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.25,
    max_delay: float = 4.0,
    retry_on: tuple[type[BaseException], ...] = (TransientError,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1)) * (0.5 + random.random() / 2)
            logger.warning("attempt %d/%d failed (%s); retrying in %.2fs", attempt, attempts, exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")
