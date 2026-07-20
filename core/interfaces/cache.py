"""Abstract interface for the AI Gateway response cache."""

from abc import ABC, abstractmethod
from typing import Optional

from core.models.cache_entry import CacheEntry


class ICacheManager(ABC):
    """Stores and retrieves LLM responses keyed on their inputs.

    Two implementations back this: an in-memory CacheManager (per-process,
    cleared on restart) and a Supabase-backed SupabaseCacheManager (persistent,
    shared across instances). The Gateway depends only on this interface.
    """

    @abstractmethod
    def generate_key(self, documents_content: str, topic: str, model: str) -> str:
        """Return the stable cache key for these inputs."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Return the entry for `key`, or None if absent or expired."""

    @abstractmethod
    def set(
        self,
        key: str,
        response: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Store a response under `key`."""

    @abstractmethod
    def clear(self) -> None:
        """Drop all cached entries."""

    @abstractmethod
    def get_metrics(self) -> dict:
        """Return hit/miss/size counters for monitoring."""
