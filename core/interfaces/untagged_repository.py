"""Abstract interface for the untagged-article capture store (Gold side-table)."""

from abc import ABC, abstractmethod
from typing import List

from core.models.article import Article


class IUntaggedRepository(ABC):
    """Stores fintech articles that matched no theme during Gold aggregation.

    A capture log for taxonomy gaps, not an input to any trend computation.
    Deduped by URL so repeated Gold runs do not pile up duplicates. Fed Silver
    `Article` records (the full-text source themes are matched against), so the
    capture reflects exactly what failed to tag.
    """

    @abstractmethod
    def save(self, articles: List[Article]) -> int:
        """Persist untagged articles, skipping any already recorded.

        Returns:
            The number of articles newly recorded.
        """
        pass
