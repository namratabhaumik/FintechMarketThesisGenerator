"""Abstract interface for vector stores."""

from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class IVectorStore(ABC):
    """Protocol for vector stores."""

    @abstractmethod
    def build(self, documents: List[Document]) -> VectorStore:
        """Build the store by embedding documents (the write path)."""
        pass

    def open(self) -> VectorStore:
        """Open the existing persistent store for reading (the read path).

        Returns a retriever-ready handle over whatever is already persisted,
        without embedding or writing anything. Default: unsupported - in-memory
        stores have nothing to open, so persistent stores override this.
        """
        raise NotImplementedError(
            "This vector store has no persistent store to open; build() it first."
        )

    @abstractmethod
    def as_retriever(
        self, vectorstore: VectorStore, k: int, fetch_k: int, lambda_mult: float
    ) -> Any:
        """Return an MMR retriever from the vectorstore.

        Retrieval is MMR: pull `fetch_k` candidates by similarity, then select
        `k` by the MMR objective (`lambda_mult` trades relevance vs diversity).
        The retrieval service passes these from RetrievalConfig.
        """
        pass
