"""Abstract interface for the untagged-article capture store (Gold side-table)."""

from abc import ABC, abstractmethod
from typing import List

from core.models.article import Article


class IUntaggedRepository(ABC):
    """Gold layer side-table: fintech articles that matched no theme.

    During Gold aggregation --> an accepted fintech article matches none of the
    known themes --> a copy is logged here instead of being thrown away.

    This is a diagnostic capture log for taxonomy gaps (articles the theme list
    failed to cover); it is NOT used in any trend count. It is fed the Silver
    `Article` records (the full text that themes are matched against), so the log
    reflects exactly what failed to tag. Deduped by URL so repeated Gold runs do
    not pile up duplicates.
    """

    @abstractmethod
    def save(self, articles: List[Article]) -> int:
        """Log untagged articles, skipping any URL already recorded.

        for each article --> if its URL is new, record it --> otherwise
        skip it.

        Returns:
            How many articles were newly recorded.
        """
        pass
