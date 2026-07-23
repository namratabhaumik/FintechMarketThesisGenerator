"""Abstract interface for language models."""

from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import List, Optional

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
    async def summarize(self, documents: List[Document], topic: str = "") -> str:
        """Generate summary from documents, focused on the user's topic.

        Args:
            documents: Source documents to summarize.
            topic: The user's query. Empty means no topic focus (generic
                fintech summary).
        """
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
        prior_feedback: Optional[List[List[str]]] = None,
    ) -> str:
        """Refine an existing thesis based on user feedback.

        Args:
            documents: List of source documents for context.
            current_thesis_text: The original thesis text to refine.
            feedback_items: This round's predefined feedback strings from user.
            prior_feedback: Earlier rounds' feedback (oldest first), given to the
                model as already-satisfied constraints to preserve while
                addressing this round. None/empty on the first refinement.

        Returns:
            Refined thesis text.

        Raises:
            NotImplementedError: If the model doesn't support refinement.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support thesis refinement"
        )
