"""Abstract interface for the raw article store"""

from abc import ABC, abstractmethod
from typing import List

from core.models.raw_article import RawArticle


class IArticleRepository(ABC):
    """Bronze layer: the raw landing zone for RSS feed entries (articles_raw).

    This is where the pipeline starts. RSS feeds are read --> each entry is
    saved here exactly as it arrived, with no cleaning. The store is append-only
    and grows over time (keyed by publish date), so it becomes a historical
    corpus rather than just the latest snapshot of the feeds.

    Everything downstream reads from here: Silver pulls these raw entries to
    classify/scrape/embed them, and Gold later aggregates the results.
    """

    @abstractmethod
    def save(self, articles: List[RawArticle]) -> int:
        """Add raw entries, skipping any already stored (deduped by URL).

        for each entry --> if its URL is new, append it --> if the URL was
        seen in an earlier run, skip it so the same article is not stored twice.

        Args:
            articles: Raw feed entries to persist verbatim.

        Returns:
            How many entries were newly inserted (duplicates are not counted).
        """
        pass

    @abstractmethod
    def fetch_all(self) -> List[RawArticle]:
        """Return all stored raw articles, newest published first.

        The Silver layer reads these to enrich them (classify, scrape, embed).
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return how many raw articles are in the store in total."""
        pass