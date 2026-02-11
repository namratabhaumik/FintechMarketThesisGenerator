"""Abstract interface for embedding models."""

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings


class IEmbeddingModel(ABC):
    """Protocol for embedding models."""

    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Return LangChain-compatible embeddings instance."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier."""
        pass
