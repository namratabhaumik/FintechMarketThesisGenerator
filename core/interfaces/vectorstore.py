"""Abstract interface for vector stores."""

from abc import ABC, abstractmethod
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
        k: int,
        fetch_k: int,
        lambda_mult: float,
        window_days: Optional[int] = None,
        query_embedding: Optional[List[float]] = None,
        min_similarity: float = 0.0,
    ) -> List[Document]:
        """Return up to `k` MMR-selected chunks for `query`.

        Pull `fetch_k` candidates by similarity, then select `k` by the MMR
        objective (`lambda_mult` trades relevance vs diversity). When
        `window_days` is set, only articles published within that trailing
        window (anchored at query time) are considered; None or 0 searches the
        whole corpus. The retrieval service passes these from RetrievalConfig.

        `min_similarity` cosine floor applied to the candidates BEFORE MMR 
        (default 0.0).

        `query_embedding`, when provided, is the already-computed vector for
        `query`: implementations reuse it instead of embedding again, so a
        caller that also needs the query vector elsewhere (e.g. episodic recall)
        embeds once. When None, the implementation embeds `query` itself.
        """
        pass
