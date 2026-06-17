"""Abstract interface for the validated article-content store"""

from abc import ABC, abstractmethod
from typing import List

from core.models.article import Article


class IArticleContentRepository(ABC):
    """Stores the validated, non-aggregated article record per URL.

    This is the durable source the embedding step transforms from, so embedding
    is replayable without re-scraping. Deduped by URL.
    """

    @abstractmethod
    def save(self, articles: List[Article]) -> int:
        """Persist validated articles, skipping any URL already stored.

        Returns:
            The number of articles newly stored.
        """
        pass

    @abstractmethod
    def fetch_all(self) -> List[Article]:
        """Return all stored validated articles."""
        pass