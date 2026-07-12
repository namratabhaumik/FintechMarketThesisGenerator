"""Abstract interface for language models."""

from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import List

from langchain_core.documents import Document

SOURCE_LLM = "llm"
SOURCE_LOCAL = "local"

# Per-call provenance of generated summary text. LocalSummarizerModel sets it
# to SOURCE_LOCAL on every call, so each path that can serve extractive text
# (wrapper outage fallback, gateway cost-limit fallback or routing, cache hits
# of a local response) marks it without callers having to know. The thesis
# service resets it before summarize and stores the outcome on the thesis, so
# a degraded no-LLM summary is visible to the API/UI instead of passing as an
# LLM one. A ContextVar (not instance state) keeps concurrent requests from
# seeing each other's value: each request task gets its own copy.
summary_source_var: ContextVar[str] = ContextVar("summary_source", default=SOURCE_LLM)


class ILanguageModel(ABC):
    """Protocol for LLM providers."""

    @abstractmethod
    async def summarize(self, documents: List[Document]) -> str:
        """Generate summary from documents."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier."""
        pass

    async def refine(
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
