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
    def open(self) -> VectorStore:
        """Open the existing persistent store for reading (the read path).

        Returns a retriever-ready handle over whatever is already persisted,
        without embedding or writing anything.
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        vectorstore: VectorStore,
        query: str,
        k: int,
        fetch_k: int,
        lambda_mult: float,
        window_days: Optional[int] = None,
    ) -> List[Document]:
        """Return up to `k` MMR-selected chunks for `query`.

        Pull `fetch_k` candidates by similarity, then select `k` by the MMR
        objective (`lambda_mult` trades relevance vs diversity). When
        `window_days` is set, only articles published within that trailing
        window (anchored at query time) are considered; None or 0 searches the
        whole corpus. The retrieval service passes these from RetrievalConfig.
        """
        pass
