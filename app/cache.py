"""Answer cache: Redis when REDIS_URL is set, otherwise an in-process TTL cache. Cache failures never fail a query."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Protocol

from .config import Settings

logger = logging.getLogger(__name__)


def cache_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return "rag:answer:" + hashlib.sha256(raw.encode()).hexdigest()


class Cache(Protocol):
    name: str

    def get(self, key: str) -> dict | None: ...

    def set(self, key: str, value: dict) -> None: ...


class MemoryTTLCache:
    name = "memory"

    def __init__(self, ttl_seconds: int = 600, max_items: int = 1024) -> None:
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.monotonic():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return json.loads(json.dumps(value))

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, json.loads(json.dumps(value)))
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)


class RedisCache:
    name = "redis"

    def __init__(self, url: str, ttl_seconds: int = 600) -> None:
        import redis

        self._errors = (redis.RedisError, OSError)
        self.ttl = ttl_seconds
        self._client = redis.Redis.from_url(url, socket_timeout=1.0, socket_connect_timeout=1.0)

    def get(self, key: str) -> dict | None:
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except self._errors as exc:
            logger.warning("redis get failed, continuing without cache: %s", exc)
            return None

    def set(self, key: str, value: dict) -> None:
        try:
            self._client.set(key, json.dumps(value), ex=self.ttl)
        except self._errors as exc:
            logger.warning("redis set failed, continuing without cache: %s", exc)


def make_cache(settings: Settings) -> Cache:
    if settings.redis_url:
        return RedisCache(settings.redis_url, settings.cache_ttl_seconds)
    return MemoryTTLCache(settings.cache_ttl_seconds)
