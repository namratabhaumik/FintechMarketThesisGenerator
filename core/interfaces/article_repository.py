"""Abstract interface for the raw article store"""

from abc import ABC, abstractmethod
from typing import List

from core.models.raw_article import RawArticle


class IArticleRepository(ABC):
    """Durable, append-only store of raw feed entries.

    Feed entries accumulate over time keyed by their publish date, so 
    later stages (Silver scrape + embed, Gold trend aggregation) have 
    a growing historical corpus instead of a single live snapshot.
    """

    @abstractmethod
    def save(self, articles: List[RawArticle]) -> int:
        """Append articles, skipping any already stored (deduped by URL).

        Args:
            articles: Raw feed entries to persist.

        Returns:
            The number of articles newly inserted (duplicates are not counted).
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of articles in the store."""
        pass