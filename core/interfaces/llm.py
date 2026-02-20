"""Abstract interface for language models."""

from abc import ABC, abstractmethod
from typing import List

from langchain.docstore.document import Document


class ILanguageModel(ABC):
    """Protocol for LLM providers."""

    @abstractmethod
    def summarize(self, documents: List[Document]) -> str:
        """Generate summary from documents."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier."""
        pass
