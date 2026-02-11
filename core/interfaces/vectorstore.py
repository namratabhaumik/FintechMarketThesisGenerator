"""Abstract interface for vector stores."""

from abc import ABC, abstractmethod
from typing import Any, List

from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore


class IVectorStore(ABC):
    """Protocol for vector stores."""

    @abstractmethod
    def build(self, documents: List[Document]) -> VectorStore:
        """Build vectorstore from documents."""
        pass

    @abstractmethod
    def as_retriever(self, vectorstore: VectorStore, k: int) -> Any:
        """Return retriever from vectorstore."""
        pass
