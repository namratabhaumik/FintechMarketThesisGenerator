"""Abstract interface for vector stores."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class IVectorStore(ABC):
    """Protocol for vector stores."""

    @abstractmethod
    def build(self, documents: List[Document]) -> VectorStore:
        """Build the store by embedding documents (the write path)."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        fetch_k: int,
        max_articles: int,
        window_days: Optional[int] = None,
        query_embedding: Optional[List[float]] = None,
        min_similarity: float = 0.0,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Document]:
        """Return up to `max_articles` distinct-article docs for `query`.

        The wide analytics pool: pull `fetch_k` chunk candidates by similarity,
        drop any below `min_similarity`, then DEDUPE BY URL (keeping the
        highest-similarity chunk per article) and cap to `max_articles`. MMR is
        NOT applied here so analytics/scoring see the full market weight while 
        only a few docs reach the LLM.

        Each returned doc carries its chunk `embedding` in metadata (under
        "embedding") so a later MMR pass can run without a second DB read; the
        persistence layer strips it before storing.

        When `window_days` is set, only articles published within that trailing
        window (anchored at query time) are considered; None or 0 searches the
        whole corpus. The retrieval service passes these from RetrievalConfig.

        `date_from`/`date_to`, when set, filter on the article's published
        date directly (an explicit range named in the query, e.g. "since
        March 2024"). The retrieval service sets only one of window_days or
        date_from/date_to per call, never both.

        `min_similarity` cosine floor applied to the candidates BEFORE dedup
        (default 0.0).

        `query_embedding`, when provided, is the already-computed vector for
        `query`: implementations reuse it instead of embedding again, so a
        caller that also needs the query vector elsewhere (e.g. episodic recall)
        embeds once. When None, the implementation embeds `query` itself.
        """
        pass
