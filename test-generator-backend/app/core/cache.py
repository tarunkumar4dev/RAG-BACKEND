"""
app/core/cache.py — High-Performance In-Memory TTL Cache
========================================================
Provides sub-millisecond in-memory caching for read-heavy static/semi-static data:
- NCERT chapter hierarchies (TTL: 1 hour)
- Chapter question stats (TTL: 10 minutes)
- Popular question queries (TTL: 3 minutes)

Designed to handle 10,000+ requests/minute with zero database / Supabase roundtrips.
Thread-safe, bounded memory with automatic expired-key pruning.
"""

import time
import threading
from typing import Any, Optional


class MemoryTTLCache:
    def __init__(self, max_size: int = 5000):
        self._max_size = max_size
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None
            val, expires_at = item
            if time.time() > expires_at:
                del self._cache[key]
                return None
            return val

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        expires_at = time.time() + ttl
        with self._lock:
            if len(self._cache) >= self._max_size:
                self._prune_expired_locked()
                # If still full, drop oldest 10%
                if len(self._cache) >= self._max_size:
                    to_remove = list(self._cache.keys())[: self._max_size // 10]
                    for k in to_remove:
                        self._cache.pop(k, None)
            self._cache[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def _prune_expired_locked(self) -> None:
        now = time.time()
        expired = [k for k, (_, exp) in self._cache.items() if now > exp]
        for k in expired:
            del self._cache[k]


# Global singleton instances for domain data
api_cache = MemoryTTLCache(max_size=5000)

