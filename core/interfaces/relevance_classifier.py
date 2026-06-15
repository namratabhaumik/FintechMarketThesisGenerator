"""Abstract interface for relevance classifiers."""

from abc import ABC, abstractmethod


class IRelevanceClassifier(ABC):
    """Protocol for deciding whether an article is relevant (e.g. fintech)."""

    @abstractmethod
    def is_relevant(self, title: str, description: str) -> bool:
        """Return True if the title/description is relevant to the domain."""
        pass
