"""Cache manager for AI Gateway response caching."""

import hashlib
import logging
from typing import Optional, Dict

from core.interfaces.cache import ICacheManager
from core.models.cache_entry import CacheEntry

logger = logging.getLogger(__name__)


def hash_cache_key(version: str, documents_content: str, topic: str, model: str) -> str:
    """Build the cache key shared by every backend.

    `version` folds a cache-busting tag (the effective model name, or an
    explicit AI_GATEWAY_CACHE_VERSION) into the hash, so a model or prompt change
    yields a different key instead of serving stale output. Shared here so the
    in-memory and Supabase managers key identically.
    """
    key_material = f"{version}|{documents_content}|{topic}|{model}"
    return hashlib.sha256(key_material.encode()).hexdigest()


class CacheManager(ICacheManager):
    """Manages in-memory caching of LLM responses."""

    def __init__(self, ttl_seconds: int = 604800, version: str = ""):
        """Initialize cache manager.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 7 days).
            version: Cache-busting tag folded into every key (see hash_cache_key).
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._ttl_seconds = ttl_seconds
        self._version = version
        self._hits = 0
        self._misses = 0

    def generate_key(self, documents_content: str, topic: str, model: str) -> str:
        """Generate cache key from documents and topic.

        Args:
            documents_content: Concatenated document content.
            topic: The query topic.
            model: The LLM model name.

        Returns:
            SHA256 hash as cache key.
        """
        return hash_cache_key(self._version, documents_content, topic, model)

    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve cache entry if valid (not expired).

        Args:
            key: Cache key.

        Returns:
            CacheEntry if found and not expired, None otherwise.
        """
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]
        if entry.is_expired(self._ttl_seconds):
            logger.debug(f"Cache entry expired: {key}")
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        logger.debug(f"Cache hit: {key}")
        return entry

    def set(self, key: str, response: str, model: str, input_tokens: int, output_tokens: int) -> None:
        """Store cache entry.

        Args:
            key: Cache key.
            response: LLM response.
            model: The LLM model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        entry = CacheEntry(
            key=key,
            response=response,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._cache[key] = entry
        logger.debug(f"Cache set: {key}")

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_metrics(self) -> dict:
        """Get cache performance metrics.

        Returns:
            Dictionary with hit rate, size, and other metrics.
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2),
            "cache_size": len(self._cache),
            "total_requests": total,
        }
