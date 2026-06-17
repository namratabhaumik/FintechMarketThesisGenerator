"""Abstract interface for the untagged-article capture store (Gold side-table)."""

from abc import ABC, abstractmethod
from typing import List

from core.models.raw_article import RawArticle


class IUntaggedRepository(ABC):
    """Stores fintech articles that matched no theme during Gold aggregation.

    A capture log for taxonomy gaps, not an input to any trend computation.
    Deduped by URL so repeated Gold runs do not pile up duplicates.
    """

    @abstractmethod
    def save(self, articles: List[RawArticle]) -> int:
        """Persist untagged articles, skipping any already recorded.

        Returns:
            The number of articles newly recorded.
        """
        pass
