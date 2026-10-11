"""What pages share between requests and between instances: histories sent once under a
key, and plans kept between a page's plan and its rank.

RECOMMENDER_CACHE_URL names a Redis-compatible store (Dragonfly in the TypeType stack):
what is kept there survives a restart and is read by every instance. Left empty, the process
keeps everything itself, which is what the tests and a single local run want. A store that
fails must never take a page down: every error reads as "not kept".
"""

import logging
import threading
import time
from functools import cache
from typing import Protocol

from . import settings

log = logging.getLogger("recommender.state")


class State(Protocol):
    def get(self, key: str) -> bytes | None: ...

    def set(self, key: str, value: bytes, seconds: int) -> None: ...


class MemoryState:
    """Kept by the process itself, until its time is up."""

    def __init__(self):
        self._kept: dict[str, tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            kept = self._kept.get(key)
            if kept is None or kept[0] < time.monotonic():
                self._kept.pop(key, None)
                return None
            return kept[1]

    def set(self, key: str, value: bytes, seconds: int) -> None:
        with self._lock:
            self._kept[key] = (time.monotonic() + seconds, value)


class RedisState:
    def __init__(self, url: str):
        import redis

        self._client = redis.Redis.from_url(
            url, socket_timeout=settings.whole("RECOMMENDER_CACHE_TIMEOUT_MS", 2000) / 1000
        )
        self._warned = threading.Event()

    def _failed(self, error: Exception) -> None:
        if not self._warned.is_set():
            self._warned.set()
            log.warning("cache unavailable, going on without it: %s", error)

    def get(self, key: str) -> bytes | None:
        try:
            return self._client.get(key)
        except Exception as error:  # noqa: BLE001 - any failure reads as "not kept"
            self._failed(error)
            return None

    def set(self, key: str, value: bytes, seconds: int) -> None:
        try:
            self._client.set(key, value, ex=max(1, seconds))
        except Exception as error:  # noqa: BLE001
            self._failed(error)


@cache
def default_state() -> State | None:
    url = settings.secret("RECOMMENDER_CACHE_URL")
    return RedisState(url) if url else None
