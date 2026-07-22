"""Supabase-backed persistent LLM response cache."""

import logging
from datetime import datetime, timezone
from typing import Optional, cast

from postgrest.types import CountMethod
from supabase import Client

from core.implementations.llm.cache_manager import hash_cache_key
from core.interfaces.cache import ICacheManager
from core.models.cache_entry import CacheEntry, effective_ttl

logger = logging.getLogger(__name__)

TABLE = "llm_cache"


def _parse_ts(value: str) -> datetime:
    """Parse a Postgres timestamptz string into a tz-aware UTC datetime."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class SupabaseCacheManager(ICacheManager):
    """Persistent response cache in a Supabase table.

    Survives process restarts / Render cold starts and is shared across
    instances, unlike the in-memory CacheManager. All I/O is best-effort: a
    Supabase failure degrades to a cache miss (get) or a silent skip (set), so a
    cache problem can never break generation - the Gateway just calls the LLM.

    Methods are synchronous (matching ICacheManager); the async Gateway offloads
    them to a worker thread so the network round-trip never blocks the loop.
    """

    def __init__(self, client: Client, ttl_seconds: int = 604800, version: str = ""):
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._version = version
        self._hits = 0
        self._misses = 0

    def generate_key(self, documents_content: str, topic: str, model: str) -> str:
        return hash_cache_key(self._version, documents_content, topic, model)

    def get(self, key: str) -> Optional[CacheEntry]:
        try:
            resp = (
                self._client.table(TABLE).select("*").eq("key", key).limit(1).execute()
            )
        except Exception as e:
            # Best-effort: a read failure is a miss, not a request failure.
            logger.warning(f"llm_cache read failed for {key}: {e}")
            self._misses += 1
            return None

        rows = resp.data or []
        if not rows:
            self._misses += 1
            return None

        row = cast(dict, rows[0])
        created_at = _parse_ts(row["created_at"])
        # Compare tz-aware UTC on both sides (CacheEntry.is_expired uses a naive
        # now(), which would mismatch the stored tz-aware timestamp).
        age = (datetime.now(timezone.utc) - created_at).total_seconds()
        if age > effective_ttl(row["model"], self._ttl_seconds):
            self._evict(key)
            self._misses += 1
            return None

        self._hits += 1
        return CacheEntry(
            key=row["key"],
            response=row["response"],
            model=row["model"],
            input_tokens=row.get("input_tokens", 0),
            output_tokens=row.get("output_tokens", 0),
            created_at=created_at,
        )

    def set(
        self,
        key: str,
        response: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        row = {
            "key": key,
            "response": response,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._client.table(TABLE).upsert(row).execute()
        except Exception as e:
            logger.warning(f"llm_cache write failed for {key}: {e}")

    def clear(self) -> None:
        try:
            # neq on the PK matches every row (no key is the empty string).
            self._client.table(TABLE).delete().neq("key", "").execute()
        except Exception as e:
            logger.warning(f"llm_cache clear failed: {e}")

    def get_metrics(self) -> dict:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        try:
            resp = self._client.table(TABLE).select(
                "key", count=CountMethod.exact
            ).execute()
            cache_size = resp.count or 0
        except Exception:
            cache_size = 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2),
            "cache_size": cache_size,
            "total_requests": total,
        }

    def _evict(self, key: str) -> None:
        """Lazily drop an expired row so it doesn't linger."""
        try:
            self._client.table(TABLE).delete().eq("key", key).execute()
        except Exception as e:
            logger.warning(f"llm_cache evict failed for {key}: {e}")
