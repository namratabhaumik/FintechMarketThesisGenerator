"""Abstract interface for language models."""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


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

    def refine(
        self,
        documents: List[Document],
        current_thesis_text: str,
        feedback_items: List[str],
    ) -> str:
        """Refine an existing thesis based on user feedback.

        Args:
            documents: List of source documents for context.
            current_thesis_text: The original thesis text to refine.
            feedback_items: List of predefined feedback strings from user.

        Returns:
            Refined thesis text.

        Raises:
            NotImplementedError: If the model doesn't support refinement.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support thesis refinement"
        )
